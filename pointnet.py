import torch
import torch.nn as nn
import torch.nn.functional as F


#辅助函数：计算平方距离（使用欧氏距离）
def squaredistance(src, dst):
    B, N, = src.shape
    _, M, = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

#最远点采样， 最远点采样 (FPS)
def farthestpointsample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batchindices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batchindices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# 索引点函数：根据索引从点集中提取点
def indexpoints(points, idx):
    device = points.device
    B = points.shape[0]
    viewshape = list(idx.shape)
    viewshape[1:] = [1] * (len(viewshape) - 1)
    repeatshape = list(idx.shape)
    repeatshape[0] = 1
    batchindices = torch.arange(B, dtype=torch.long).to(device).view(viewshape).repeat(repeatshape)
    return points[batchindices, idx:]

#球查询函数：在给定半径内查询点，不够时填充第一个有效点
def queryballpoint(radius, nsample, xyz, newxyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = newxyz.shape
    groupidx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat([B, S, 1])
    sqrdists = squaredistance(newxyz, xyz)
    groupidx[sqrdists > radius ** 2] = N
    groupidx = groupidx.sort(dim=-1)[0][:, :, :nsample]
    groupfirst = groupidx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = groupidx == N
    groupidx[mask] = groupfirst[mask]
    return groupidx

###########以下是核心模块(SA和FP)###########
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, inchannel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        #构架共享的MLP层
        layers = []
        lastchannel = inchannel + 3#用于拼接相对坐标
        for outchannel in mlp:
            layers.append(nn.Conv2d(lastchannel, outchannel, 1))#等价于全连接层，进行编码与解码的部分
            layers.append(nn.BatchNorm2d(outchannel))#归一化处理
            layers.append(nn.ReLU())#激活函数
            lastchannel = outchannel#下一层
        self.mlp = nn.Sequential(*layers)
################一层的定义




    def forward(self, xyz, points):
        B, N, C = xyz.shape# B=批大小, N=点数, C=坐标维度(通常为3)
        S = self.npoint
        #关键点采样
        fpsidx = farthestpointsample(xyz, S)
        newxyz = indexpoints(xyz, fpsidx)

        #领域查询
        idx = queryballpoint(self.radius, self.nsample, xyz, newxyz)
        groupedxyz = indexpoints(xyz, idx) - newxyz.unsqueeze(-2) # 计算相对坐标，转换为局部坐标系
        #特征拼接
        if points is not None:
            groupedpoints = indexpoints(points, idx)
            groupedpoints = torch.cat([groupedxyz, groupedpoints], dim=-1)
        else:
            groupedpoints = groupedxyz
        #特征提取
        groupedpoints = groupedpoints.permute(0, 3, 2, 1)
        newpoints = self.mlp(groupedpoints)
        #最大池化
        newpoints = torch.max(newpoints, 3)[0]
        return newxyz, newpoints

# 特征传播模块：将特征从粗到细传播
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, inchannel, mlp):
        super().__init__()
        layers = []#构建MLP层
        lastchannel = inchannel
        for outchannel in mlp:
            layers.append(nn.Conv1d(lastchannel, outchannel, 1))
            layers.append(nn.BatchNorm1d(outchannel))
            layers.append(nn.ReLU())
            lastchannel = outchannel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        if xyz1 is None:#全局特征直接传递
            return self.mlp(points2)
        #点间距离计算，3个最近邻插值
        dists = squaredistance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)#距离
        dists, idx = dists[:, :, :3], idx[:, :, :3]#插值
        distrecip = 1.0 / (dists + 1e-8)#距离倒数
        norm = torch.sum(distrecip, dim=2, keepdim=True)#归一化因子
        weight = distrecip / norm
        #加权求和
        interpolatedpoints = torch.sum(indexpoints(points2, idx) * weight.view(B, -1, 3, 1), dim=2)

        #特征拼接
        if points1 is not None:
            newpoints = torch.cat([points1, interpolatedpoints], dim=1)
        else:
            newpoints = interpolatedpoints
        #特征提取，特征变换
        newpoints = newpoints.unsqueeze(-1)
        newpoints = self.mlp(newpoints)
        newpoints = newpoints.squeeze(-1)
        return newpoints

#分类网络架构
class PointNet2Cls(nn.Module):
    def __init__(self, numclasses):
        super().__init__()
        #定义三个SA模块，下采样
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, inchannel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, inchannel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, inchannel=256 + 3,
                                          mlp=[256, 512, 1024])

        #分类头
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, numclasses)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        #特征提取层
        l1xyz, l1points = self.sa1(xyz, None)
        l2xyz, l2points = self.sa2(l1xyz, l1points)
        l3xyz, l3points = self.sa3(l2xyz, l2points)
        #全局特征
        x = l3points.view(-1, 1024)
        #分类头
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
#在每个分组内，PointNet++ 使用 PointNet 对局部区域的点云进行特征学习。
# PointNet 提取每个局部区域的特征，并通过对称函数（如最大池化）将这些局部特征聚合为一个全局特征，
# 生成每个质心点的特征表示。
#3层SA对应于PointNet++的三个Set Abstraction层，分别用于提取不同尺度的局部特征。




##########训练过程

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

# 定义设备
device = torch.device("cuda:0" if torch.cuda.isavailable() else "cpu")

# 超参数设置
batchsize=32
numepochs=50
learningrate=0.001
numclasses=40

# 加载数据集
traindataset = ModelNet40(root='data/modelnet40normalresampled', split='train')
trainloader = data.DataLoader(traindataset, batchsize=batchsize, shuffle=True, numworkers=4)
testdataset = ModelNet40(root='data/modelnet40normalresampled', split='test')
testloader = data.DataLoader(testdataset, batchsize=batchsize, shuffle=False, numworkers=4)

# 初始化模型、损失函数和优化器
model = PointNet2Cls(numclasses).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

# 训练模型
for epoch in range(numepochs):
    model.train()
    runningloss = 0.0
    for i, (points, labels) in enumerate(trainloader):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad()  # 修正：原代码中的zerograd应为zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        runningloss += loss.item()
    print(f'Epoch{epoch+1}/{numepochs},Loss:{runningloss/len(trainloader)})')


#####B，N，C
#一次处理的点云数量；单个点云中的总点数；点坐标维度
##D,S,K
#点特征维度;采样关键点数量(Set Abstraction层输出);每个关键点的邻域点数(nsample参数)

###########源码部分##########
#####SA
class PointNetSetAbstraction(nn.Module):
    '''
    如：npoint=128,radius=0.4,nsample=64,in_channle=128+3,mlp=[128,128,256],group_all=False
    128=npoint:points sampled in farthest point sampling
    0.4=radius:search radius in local region
    64=nsample:how many points inn each local region
    [128,128,256]=output size for MLP on each point
    '''
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        #nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
        # 可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，
        # 方法和 Python 自带的 list 一样，无非是 extend，append 等操作。
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]  # B=批大小, C=坐标维度(通常为3), N=点数
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)#将tensor的维度换位#（B,N,C）
        if points is not None:
            points = points.permute(0, 2, 1)#（B,N,D）

        if self.group_all: # 形成局部的group
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]

        # 以下是pointnet操作：
        # 对局部group中每一个点做MLP操作:
        # 利用1*1的2d卷积相当于把每个group当成一个通道，共npoint个通道
        # 对[C+D，nsample]的维度上做逐像素的卷积，结果相当于对单个c+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # print(new_points.shape)
        # 最后进行局部的最大池化，得到局部的全局特征
        # 对每个group做一个max pooling得到局部的全局特征,得到的new_points:[B,3+D,npoint]
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

##########MSG
class PointNetSetAbstractionMsg(nn.Module):
    # 例如：128,[0.2,0.4,0.8],[32,64,128],320,[[64,64,128],[128,128,256],[128,128,256]]
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint  # 最远采样点数
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []  # 将不同半径下点云特征保存在 new_points_list
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # 拼接点云特征数据和点坐标数据
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)  # 不同半径下点云特征的列表保存到new_points_list

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)  # 拼接不同半径下的点云特征的列表
        return new_xyz, new_points_concat


###################PointNet++上采样（Feature Propagation）
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:利用前一层的点对后面的点进行插值
            xyz1: input points position data, [B, C, N]  L层输出 xyz
            xyz2: sampled input points position data, [B, C, S]  L+1层输出 xyz
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:# 当点的个数只有一个的时候，采用repeat直接复制成N个点
            interpolated_points = points2.repeat(1, N, 1)
        else:# 当点的个数大于一个的时候，采用线性差值的方式进行上采样
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)#在最后一个维度进行排序 默认进行升序排序，也就是越靠前的位置说明 xyz1离xyz2距离较近
            #找到距离最近的三个邻居，这里的idx：B, N, 3的含义就是N个点与S个点距离最近的前三个点的索引，
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)#距离越远权值越小# 求距离的倒数
            norm = torch.sum(dist_recip, dim=2, keepdim=True)#对dist_recip的倒数求和 torch.sum   keepdim=True#也就是将距离最近的三个邻居的加起来
            weight = dist_recip / norm #这里的weight是计算权重  dist_recip中存放的是三个邻居的距离  norm中存放是距离的和
                                       #两者相除就是每个距离占总和的比重 也就是weight
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            #points2: B,S,C (S个点 C个特征)   idx B,N,3 （N个点中与S个点距离最近的三个点的索引）
            # index_points(points2, idx) 从高维特征（S个点）中找到对应低维特征（N个点） 对应距离最小的三个点的特征 B,S,3,C
            # 这个索引的含义比较重要，可以再看一下idx参数的解释，其实B,N,3,C中的N都是高维特征S个点组成的。
            # 例如 N中的第一个点 可能是由S中的第 1 2 3 组成的；第二个点可能是由2 3 4 三个点组成的
            #torch.sum dim=2 最后在第二个维度求和 取三个点的特征乘以权重后的和 也就完成了对特征点的上采样

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
