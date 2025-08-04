import torch
import torch.nn as nn
import torch.nn.functional as F
import model.unet.pytorch_utils as pt_utils
import numpy as np


class RandLANetUNet(nn.Module):
    def __init__(self, input_c, m, nPlanes, cfg):
        super().__init__()
        self.config = cfg
        self.input_c = input_c
        self.m = m
        self.nPlanes = nPlanes

        # 严格按照参考代码的配置 - 检查这些参数
        self.num_layers = getattr(cfg, 'num_layers', 5)  # 必须是5层
        self.d_out = getattr(cfg, 'd_out', [16, 64, 128, 256, 512])  # 必须是这个配置
        self.k_n = getattr(cfg, 'k_n', 16)  # KNN邻居数
        self.num_points = getattr(cfg, 'num_points', 65536)
        self.sub_sampling_ratio = getattr(cfg, 'sub_sampling_ratio', [4, 4, 4, 2, 2])

        print(f"[DEBUG] Network Config:")
        print(f"  num_layers: {self.num_layers}")
        print(f"  d_out: {self.d_out}")
        print(f"  k_n: {self.k_n}")
        print(f"  sub_sampling_ratio: {self.sub_sampling_ratio}")

        # 完全对标参考代码的初始化
        self.fc0 = nn.Linear(input_c, 8)
        self.fc0_acti = nn.LeakyReLU()
        self.fc0_bath = nn.BatchNorm1d(8, eps=1e-6, momentum=0.99)
        nn.init.constant_(self.fc0_bath.weight, 1.0)
        nn.init.constant_(self.fc0_bath.bias, 0)

        # 编码器 - 严格按照参考代码
        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.num_layers):
            d_out = self.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out
            print(f"  Encoder Layer {i}: {8 if i == 0 else 2 * self.d_out[i - 1]} -> {d_out} -> {2 * d_out}")

        # 中间层
        d_out = d_in  # 应该是 1024
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        print(f"  Middle Layer: {d_in} -> {d_out}")

        # 解码器 - 严格按照参考代码的维度计算
        self.decoder_blocks = nn.ModuleList()
        for j in range(self.num_layers):
            if j < self.num_layers - 1:
                d_in = d_out + 2 * self.d_out[-j - 2]
                d_out = 2 * self.d_out[-j - 2]
            else:
                d_in = 4 * self.d_out[-self.num_layers]  # 4 * 16 = 64
                d_out = 2 * self.d_out[-self.num_layers]  # 2 * 16 = 32
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))
            print(f"  Decoder Layer {j}: {d_in} -> {d_out}")

        # 输出层
        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, m, kernel_size=(1, 1), bn=False, activation=None)
        print(f"  Output Layers: {d_out} -> 64 -> 32 -> {m}")

    def forward(self, batch):
        """检查输入数据格式，并适配到参考代码的格式"""
        # 检查输入格式
        if self._is_raw_format(batch):
            # print("[DEBUG] Converting raw format to RandLA format...")
            end_points = self._convert_to_randla_format(batch)
        else:
            # print("[DEBUG] Input already in RandLA format")
            end_points = batch

        # 严格按照参考代码的前向传播
        return self._forward_randla(end_points)

    def _is_raw_format(self, batch):
        """检查是否是原始格式（只有xyz和features）"""
        return ('xyz' in batch and 'features' in batch and
                'neigh_idx' not in batch and 'sub_idx' not in batch)

    def _convert_to_randla_format(self, batch):
        """将原始格式转换为RandLA-Net需要的格式"""
        features = batch['features']  # (B, C, N)
        xyz = batch['xyz']  # (B, N, 3)

        B, C, N = features.shape
        device = features.device

        print(f"[DEBUG] Converting input: B={B}, C={C}, N={N}")

        # 构建end_points
        end_points = {}
        end_points['features'] = features.transpose(1, 2)  # (B, N, C)

        # 构建多层次数据结构
        xyz_list = []
        neigh_idx_list = []
        sub_idx_list = []
        interp_idx_list = []

        current_xyz = xyz
        current_n = N

        print(f"[DEBUG] Building multi-level structure...")

        for i in range(self.num_layers):
            print(f"  Level {i}: N={current_n}")
            xyz_list.append(current_xyz)

            # 创建KNN邻居索引
            neigh_idx = self._build_knn_index(current_xyz, self.k_n)
            neigh_idx_list.append(neigh_idx)

            # 创建下采样索引
            if i < self.num_layers - 1:
                next_n = current_n // self.sub_sampling_ratio[i]
                next_n = max(next_n, 64)  # 避免过度下采样

                # 使用随机采样创建sub_idx
                sub_idx = self._build_subsample_index(current_n, next_n, self.k_n, device, B)
                sub_idx_list.append(sub_idx)

                # 下采样xyz
                current_xyz = self._apply_subsample(current_xyz, sub_idx)
                current_n = next_n

        # 构建插值索引（从粗到细）
        for i in range(self.num_layers):
            if i == 0:
                # 最粗层到次粗层
                interp_idx = self._build_interp_index(xyz_list[-1], xyz_list[-2])
            else:
                # 其他层
                target_level = self.num_layers - 1 - i
                if target_level > 0:
                    interp_idx = self._build_interp_index(xyz_list[target_level], xyz_list[target_level - 1])
                else:
                    interp_idx = self._build_interp_index(xyz_list[1], xyz_list[0])
            interp_idx_list.append(interp_idx)

        end_points['xyz'] = xyz_list
        end_points['neigh_idx'] = neigh_idx_list
        end_points['sub_idx'] = sub_idx_list
        end_points['interp_idx'] = interp_idx_list

        print(f"[DEBUG] Built structure with {len(xyz_list)} levels")
        return end_points

    def _build_knn_index(self, xyz, k):
        """构建KNN邻居索引"""
        B, N, _ = xyz.shape

        # 使用CDist计算距离
        dist = torch.cdist(xyz, xyz)  # (B, N, N)

        # 获取最近的k个邻居
        _, indices = torch.topk(dist, k, dim=2, largest=False)

        return indices  # (B, N, k)

    def _build_subsample_index(self, current_n, next_n, k, device, B):
        """构建下采样索引"""
        # 创建随机采样索引
        sub_idx = torch.stack([torch.randperm(current_n, device=device)[:next_n] for _ in range(B)])

        # 扩展到k个邻居（模拟pooling操作）
        sub_idx = sub_idx.unsqueeze(-1).expand(-1, -1, k)  # (B, next_n, k)

        return sub_idx

    def _apply_subsample(self, xyz, sub_idx):
        """应用下采样"""
        B, _, _ = xyz.shape
        device = xyz.device

        # 取第一个邻居作为采样点
        sample_idx = sub_idx[:, :, 0]  # (B, next_n)

        # 使用gather采样
        batch_idx = torch.arange(B, device=device).view(B, 1).expand_as(sample_idx)
        sampled_xyz = xyz[batch_idx, sample_idx]

        return sampled_xyz

    def _build_interp_index(self, xyz_coarse, xyz_fine):
        """构建插值索引"""
        B, N_fine, _ = xyz_fine.shape

        # 计算距离
        dist = torch.cdist(xyz_fine, xyz_coarse)  # (B, N_fine, N_coarse)

        # 找最近邻
        _, interp_idx = torch.min(dist, dim=2)  # (B, N_fine)

        return interp_idx.unsqueeze(-1)  # (B, N_fine, 1)

    def _forward_randla(self, end_points):
        """严格按照参考代码的前向传播"""
        features = end_points['feats']  # (B, N, C)
        features = self.fc0(features)

        # 关键的三步变换
        features = self.fc0_acti(features)
        features = features.transpose(1, 2)  # (B, C, N)
        features = self.fc0_bath(features)

        features = features.unsqueeze(dim=3)  # (B, C, N, 1)

        # 编码器
        f_encoder_list = []
        for i in range(self.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            if i < len(end_points['sub_idx']):
                f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
                features = f_sampled_i
            else:
                features = f_encoder_i

            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(features)

        # 中间层
        features = self.decoder_0(f_encoder_list[-1])

        # 解码器
        for j in range(self.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))
            features = f_decoder_i

        # 输出层
        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)

        return features.squeeze(3)  # (B, m, N)

    @staticmethod
    def random_sample(feature, pool_idx):
        """完全复制参考代码"""
        feature = feature.squeeze(dim=3)
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """完全复制参考代码"""
        feature = feature.squeeze(dim=3)
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)
        return interpolated_features


# 完全复制参考代码的核心模块
class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)
        f_pc = self.lfa(xyz, f_pc, neigh_idx)
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)
        self.mlp2 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = f_xyz.permute((0, 3, 1, 2))
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg