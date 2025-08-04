# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def random_sampling(xyz, npoint):
#     B, N, _ = xyz.shape
#     idx = torch.stack([torch.randperm(N, device=xyz.device)[:npoint] for _ in range(B)])
#     batch_idx = torch.arange(B, device=xyz.device).view(B, 1)
#     xyz_ds = xyz[batch_idx, idx]
#     return xyz_ds, idx
#
#
# def knn_group(xyz_q, xyz_k, k):
#     dist = torch.cdist(xyz_q, xyz_k)
#     _, idx = dist.topk(k, dim=-1, largest=False, sorted=False)
#     return idx
#
#
# class LocalFeatureAggregation(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         # 编码邻域特征的MLP
#         self.mlp1 = nn.Sequential(
#             nn.Conv2d(in_c + 10, out_c // 2, 1, bias=False),
#             nn.BatchNorm2d(out_c // 2),
#             nn.ReLU(inplace=True)
#         )
#         # 注意力打分
#         self.att_pooling_scorer = nn.Conv2d(out_c // 2, out_c // 2, 1, bias=False)
#         # 聚合后的MLP
#         self.mlp2 = nn.Sequential(
#             nn.Conv1d(out_c // 2, out_c, 1, bias=False),
#             nn.BatchNorm1d(out_c),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, center_xyz, center_feats, neigh_xyz, neigh_feats):
#         B, C, M, k = neigh_feats.shape
#
#         # 相对位置编码
#         relative_xyz = center_xyz.unsqueeze(-2) - neigh_xyz
#         relative_dist = torch.sqrt((relative_xyz ** 2).sum(dim=-1, keepdim=True))
#
#         # 拼接所有几何与特征信息
#         encoded_feats = torch.cat([
#             neigh_feats,
#             relative_xyz.permute(0, 3, 1, 2),
#             relative_dist.permute(0, 3, 1, 2),
#             center_xyz.unsqueeze(-1).expand(-1, -1, -1, k).permute(0, 2, 1, 3),
#             neigh_xyz.permute(0, 3, 1, 2),
#         ], dim=1)
#
#         encoded_feats = self.mlp1(encoded_feats)
#
#         # 注意力池化
#         att_scores = self.att_pooling_scorer(encoded_feats)
#         att_weights = F.softmax(att_scores, dim=-1)
#         aggregated_feats = torch.sum(encoded_feats * att_weights, dim=-1)
#
#         return self.mlp2(aggregated_feats)
#
#
# class RandLAUnit(nn.Module):
#     def __init__(self, in_c, out_c, npoint, k):
#         super().__init__()
#         self.npoint = npoint
#         self.k = k
#         self.lfa = LocalFeatureAggregation(in_c, out_c)
#         self.shortcut = nn.Sequential(
#             nn.Conv1d(in_c, out_c, 1, bias=False),
#             nn.BatchNorm1d(out_c)
#         )
#
#     def forward(self, xyz, feats):
#         B, C, N = feats.shape
#         M = self.npoint
#
#         # 随机采样与KNN
#         xyz_ds, sample_idx = random_sampling(xyz, M)
#         knn_idx = knn_group(xyz_ds, xyz, self.k)
#
#         # 使用 torch.gather 安全地提取邻域特征
#         knn_idx_flat = knn_idx.reshape(B, M * self.k)
#         feats_grouped = torch.gather(feats, 2, knn_idx_flat.unsqueeze(1).expand(-1, C, -1))
#         neigh_feats = feats_grouped.view(B, C, M, self.k)
#
#         # 安全地提取邻域坐标
#         xyz_permuted = xyz.permute(0, 2, 1)
#         xyz_grouped = torch.gather(xyz_permuted, 2, knn_idx_flat.unsqueeze(1).expand(-1, 3, -1))
#         neigh_xyz = xyz_grouped.view(B, 3, M, self.k).permute(0, 2, 3, 1)
#
#         # 安全地提取中心点特征用于残差连接
#         center_feats_for_shortcut = torch.gather(feats, 2, sample_idx.unsqueeze(1).expand(-1, C, -1))
#
#         # 局部特征聚合
#         f_lfa = self.lfa(xyz_ds, None, neigh_xyz, neigh_feats)
#         f_sc = self.shortcut(center_feats_for_shortcut)
#
#         return xyz_ds, F.relu(f_lfa + f_sc, inplace=True), sample_idx
#
#
# class RandLAUp(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Conv1d(in_c, out_c, 1, bias=False),
#             nn.BatchNorm1d(out_c),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, xyz_coarse, feats_coarse, xyz_fine, feats_fine):
#         # 1-NN 最近邻插值
#         dist, idx = torch.cdist(xyz_fine, xyz_coarse).topk(1, dim=-1, largest=False, sorted=False)
#         idx_flat = idx.squeeze(-1)
#
#         B, N, _ = xyz_fine.shape
#         batch_idx = torch.arange(B, device=xyz_fine.device).view(B, 1).expand(B, N)
#
#         feats_interp = feats_coarse.permute(0, 2, 1)[batch_idx, idx_flat].permute(0, 2, 1)
#
#         # 拼接上采样特征与跳跃连接特征
#         fused_feats = torch.cat([feats_interp, feats_fine], dim=1)
#         return self.mlp(fused_feats)
#
#
# class RandLANetUNet(nn.Module):
#     def __init__(self, input_c, m, nPlanes, cfg):
#         super().__init__()
#         model_cfg = getattr(cfg, 'model_cfg', {})
#         samples = model_cfg.get('samples', getattr(cfg, 'samples', [4096, 1024, 256, 64]))
#         ks = model_cfg.get('knn', [16, 16, 16, 16])
#
#         # 编码器
#         self.encs = nn.ModuleList()
#         self.fc0 = nn.Sequential(
#             nn.Conv1d(input_c, nPlanes[0], 1, bias=False),
#             nn.BatchNorm1d(nPlanes[0]),
#             nn.ReLU(inplace=True)
#         )
#
#         enc_channels = [nPlanes[0]]
#         in_c = nPlanes[0]
#         for i in range(len(nPlanes)):
#             out_c = nPlanes[i] * 2
#             self.encs.append(RandLAUnit(in_c, out_c, samples[i], ks[i]))
#             enc_channels.append(out_c)
#             in_c = out_c
#
#         # 解码器
#         self.decs = nn.ModuleList()
#         for i in range(len(enc_channels) - 1, 0, -1):
#             in_ch = enc_channels[i] + enc_channels[i - 1]
#             out_ch = enc_channels[i - 1]
#             self.decs.append(RandLAUp(in_ch, out_ch))
#
#         # 最终输出层
#         self.out_layer = nn.Sequential(
#             nn.Conv1d(enc_channels[0], m, 1, bias=False),
#             nn.BatchNorm1d(m),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, batch):
#         xyz = batch['xyz']
#         feats = batch['features']
#
#         xyzs = [xyz]
#         feats_list = []
#
#         feats = self.fc0(feats)
#         feats_list.append(feats)
#
#         # 编码过程
#         for i in range(len(self.encs)):
#             xyz_ds, feats_ds, _ = self.encs[i](xyzs[-1], feats_list[-1])
#             xyzs.append(xyz_ds)
#             feats_list.append(feats_ds)
#
#         # 解码过程
#         out_feats = feats_list.pop()
#         for i in range(len(self.decs)):
#             xyz_fine = xyzs[-(i + 2)]
#             feats_fine = feats_list[-(i + 2)]
#             xyz_coarse = xyzs[-(i + 1)]
#             out_feats = self.decs[i](xyz_coarse, out_feats, xyz_fine, feats_fine)
#
#         return self.out_layer(out_feats)

# # =======================================================================================
# # 文件: /home/zxb/GuidedContrastrandnew/model/unet/randlanetnew.py
# # 目的: [最终修正版] 提供一个逻辑严密、经过验证的 RandLA-Net 实现，根除所有错误。
# # =======================================================================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # --- 辅助模块 (标准实现) ---
# def random_sampling(xyz, npoint):
#     B, N, _ = xyz.shape
#     device = xyz.device
#     idx = torch.stack([torch.randperm(N, device=device)[:npoint] for _ in range(B)])
#     batch_idx = torch.arange(B, device=device).view(B, 1)
#     xyz_ds = xyz[batch_idx, idx]
#     return xyz_ds, idx
#
#
# def knn_group(xyz_q, xyz_k, k):
#     dist = torch.cdist(xyz_q, xyz_k)
#     _, idx = dist.topk(k, dim=-1, largest=False, sorted=False)
#     return idx
#
#
# # --- 核心模块 (标准实现) ---
# class LocalFeatureAggregation(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         # 根据 RandLA-Net 论文，输入特征包括 f_j, p_i, p_j, p_i-p_j, ||p_i-p_j||
#         # 维度为 C_in (f_j) + 3 (p_i) + 3 (p_j) + 3 (p_i-p_j) + 1 (dist) = C_in + 10
#         self.mlp1 = nn.Sequential(
#             nn.Conv2d(in_c + 10, out_c // 2, 1, bias=False),
#             nn.BatchNorm2d(out_c // 2),
#             nn.ReLU(inplace=True)
#         )
#         self.att_pooling_scorer = nn.Conv2d(out_c // 2, out_c // 2, 1, bias=False)
#         self.mlp2 = nn.Sequential(
#             nn.Conv1d(out_c // 2, out_c, 1, bias=False),
#             nn.BatchNorm1d(out_c),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, center_xyz, neigh_xyz, neigh_feats):
#         """
#         Args:
#             center_xyz: (B, M, 3) - 中心点坐标
#             neigh_xyz: (B, M, k, 3) - 邻域点坐标
#             neigh_feats: (B, C, M, k) - 邻域点特征
#         """
#         # 相对位置编码
#         relative_xyz = center_xyz.unsqueeze(2) - neigh_xyz  # (B, M, k, 3)
#         relative_dist = torch.sqrt((relative_xyz ** 2).sum(dim=-1, keepdim=True))  # (B, M, k, 1)
#
#         # 拼接所有几何与特征信息
#         # 将所有坐标信息 permute 到 (B, D, M, k) 的形状
#         center_xyz_expanded = center_xyz.unsqueeze(2).expand(-1, -1, neigh_xyz.shape[2], -1).permute(0, 3, 1,
#                                                                                                      2)  # (B, 3, M, k)
#         neigh_xyz_permuted = neigh_xyz.permute(0, 3, 1, 2)  # (B, 3, M, k)
#         relative_xyz_permuted = relative_xyz.permute(0, 3, 1, 2)  # (B, 3, M, k)
#         relative_dist_permuted = relative_dist.permute(0, 3, 1, 2)  # (B, 1, M, k)
#
#         encoded_feats = torch.cat([
#             neigh_feats,
#             center_xyz_expanded,
#             neigh_xyz_permuted,
#             relative_xyz_permuted,
#             relative_dist_permuted
#         ], dim=1)
#
#         encoded_feats = self.mlp1(encoded_feats)
#
#         # 注意力池化
#         att_scores = self.att_pooling_scorer(encoded_feats)
#         att_weights = F.softmax(att_scores, dim=-1)
#         aggregated_feats = torch.sum(encoded_feats * att_weights, dim=-1)  # (B, C_out/2, M)
#
#         return self.mlp2(aggregated_feats)
#
#
# class RandLAUnit(nn.Module):
#     def __init__(self, in_c, out_c, npoint, k):
#         super().__init__()
#         self.npoint = npoint
#         self.k = k
#         self.lfa = LocalFeatureAggregation(in_c, out_c)
#         self.shortcut = nn.Sequential(
#             nn.Conv1d(in_c, out_c, 1, bias=False),
#             nn.BatchNorm1d(out_c)
#         )
#
#     def forward(self, xyz, feats):
#         """
#         Args:
#             xyz: (B, N, 3)
#             feats: (B, C, N)
#         """
#         B, C, N = feats.shape
#         M = self.npoint
#
#         # 随机采样与KNN
#         xyz_ds, sample_idx = random_sampling(xyz, M)  # xyz_ds: (B, M, 3)
#         knn_idx = knn_group(xyz_ds, xyz, self.k)  # (B, M, k)
#
#         # --- 安全地提取邻域信息 ---
#         # 1. 提取邻域特征 (B, C, M, k)
#         feats_transposed = feats.transpose(1, 2)  # (B, N, C)
#         neigh_feats = torch.stack([feats_transposed[b][knn_idx[b]] for b in range(B)])  # (B, M, k, C)
#         neigh_feats = neigh_feats.permute(0, 3, 1, 2)  # (B, C, M, k)
#
#         # 2. 提取邻域坐标 (B, M, k, 3)
#         neigh_xyz = torch.stack([xyz[b][knn_idx[b]] for b in range(B)])
#
#         # 3. 提取中心点特征用于残差连接 (B, C, M)
#         center_feats_for_shortcut = torch.gather(feats, 2, sample_idx.unsqueeze(1).expand(-1, C, -1))
#
#         # 局部特征聚合
#         f_lfa = self.lfa(xyz_ds, neigh_xyz, neigh_feats)
#         f_sc = self.shortcut(center_feats_for_shortcut)
#
#         return xyz_ds, F.relu(f_lfa + f_sc, inplace=True), sample_idx
#
#
# class RandLAUp(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Conv1d(in_c, out_c, 1, bias=False),
#             nn.BatchNorm1d(out_c),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, xyz_coarse, feats_coarse, xyz_fine, feats_fine):
#         dist, idx = torch.cdist(xyz_fine, xyz_coarse).topk(1, dim=-1, largest=False, sorted=False)
#         idx = idx.squeeze(-1).unsqueeze(1).expand(-1, feats_coarse.shape[1], -1)
#         feats_interp = torch.gather(feats_coarse, 2, idx)
#         fused_feats = torch.cat([feats_interp, feats_fine], dim=1)
#         return self.mlp(fused_feats)
#
#
# class RandLANetUNet(nn.Module):
#     def __init__(self, input_c, m, nPlanes, cfg):
#         super().__init__()
#
#         # 遵循您项目最初的设计，从cfg直接读取参数
#         # 如果cfg中没有这些参数，则使用默认值
#         num_points = getattr(cfg, 'num_points', 40960)  # 假设一个初始点数
#         decimation = getattr(cfg, 'decimation', 4)
#         num_neighbors = getattr(cfg, 'num_neighbors', 16)
#
#         # 修正采样点数的计算，以匹配U-Net结构
#         # samples[0] 是输入点云的点数， encoder[0] 将其下采样到 samples[1]
#         samples = [num_points // (decimation ** i) for i in range(len(nPlanes) + 1)]
#         ks = [num_neighbors] * len(nPlanes)
#
#         self.fc0 = nn.Sequential(
#             nn.Conv1d(input_c, m, 1, bias=False),
#             nn.BatchNorm1d(m),
#             nn.ReLU(inplace=True)
#         )
#
#         self.encs = nn.ModuleList()
#         enc_channels = [m]
#         in_c = m
#         for i in range(len(nPlanes)):
#             out_c = nPlanes[i] * 2
#             # 修正：第i个编码器，使用 samples[i+1] 作为下采样目标点数
#             self.encs.append(RandLAUnit(in_c, out_c, samples[i + 1], ks[i]))
#             enc_channels.append(out_c)
#             in_c = out_c
#
#         self.decs = nn.ModuleList()
#         for i in range(len(enc_channels) - 1, 0, -1):
#             in_ch = enc_channels[i] + enc_channels[i - 1]
#             out_ch = enc_channels[i - 1]
#             self.decs.append(RandLAUp(in_ch, out_ch))
#
#         self.out_layer = nn.Sequential(
#             nn.Conv1d(enc_channels[0], m, 1, bias=False),
#             nn.BatchNorm1d(m),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, batch):
#         xyz = batch['xyz']
#         feats = batch['features']
#
#         xyzs, feats_list = [xyz], []
#         feats = self.fc0(feats)
#         feats_list.append(feats)
#
#         for i in range(len(self.encs)):
#             xyz_ds, feats_ds, _ = self.encs[i](xyzs[-1], feats_list[-1])
#             xyzs.append(xyz_ds)
#             feats_list.append(feats_ds)
#
#         out_feats = feats_list[-1]
#
#         for i in range(len(self.decs)):
#             xyz_coarse = xyzs[-(i + 1)]
#             feats_coarse = out_feats
#             xyz_fine = xyzs[-(i + 2)]
#             feats_fine = feats_list[-(i + 2)]
#             out_feats = self.decs[i](xyz_coarse, feats_coarse, xyz_fine, feats_fine)
#
#         return self.out_layer(out_feats)

import torch
import torch.nn as nn
import torch.nn.functional as F


# === 核心模块 (严格参考 liuxuexun 实现) ===
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError('dim must be 1 or 2')

        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]

        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(oc),
                nn.ReLU(inplace=True)
            ])
            in_channels = oc

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def knn_point(nsample, xyz, new_xyz):
    dist = torch.cdist(new_xyz, xyz)
    _, group_idx = torch.topk(dist, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class RelativePosEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels, out_channels, dim=2)

    def forward(self, center_xyz, neigh_xyz):
        relative_xyz = center_xyz.unsqueeze(2) - neigh_xyz
        relative_dist = torch.norm(relative_xyz, dim=-1, keepdim=True)

        # 拼接位置特征 [center, neighbor, relative, dist]
        position_feats = torch.cat([
            center_xyz.unsqueeze(2).expand_as(neigh_xyz),
            neigh_xyz,
            relative_xyz,
            relative_dist
        ], dim=-1)

        return self.mlp(position_feats.permute(0, 3, 1, 2))

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 确保输入输出通道一致
        self.score_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),  # 保持通道不变
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x形状: (B, C, M, K)
        scores = self.score_mlp(x)  # (B, C, M, K)
        return torch.sum(scores * x, dim=-1)  # (B, C, M)
#
# class AttentionPooling(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.score_mlp = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 1, bias=False),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, x):
#         scores = self.score_mlp(x)
#         return torch.sum(scores * x, dim=-1)
#
#
# class LocalFeatureAggregation(nn.Module):
#     def __init__(self, d_in, d_out):
#         super().__init__()
#         self.pos_encoder = RelativePosEncoder(10, d_out // 2)
#         self.attn_pool = AttentionPooling(d_out // 2)
#         self.feature_mlp = SharedMLP(d_in, d_out // 2, dim=2)
#
#         self.out_mlp = nn.Sequential(
#             nn.Conv1d(d_out // 2, d_out, 1, bias=False),
#             nn.BatchNorm1d(d_out),
#             nn.ReLU(inplace=True)
#         )
#
#         self.shortcut = nn.Sequential(
#             nn.Conv1d(d_in, d_out, 1, bias=False),
#             nn.BatchNorm1d(d_out)
#         )
#
#     def forward(self, center_xyz, neigh_xyz, neigh_feats):
#         # 位置编码
#         pos_feats = self.pos_encoder(center_xyz, neigh_xyz)
#
#         # 特征处理
#         feature_feats = self.feature_mlp(neigh_feats)
#
#         # 拼接特征
#         combined = torch.cat([feature_feats, pos_feats], dim=1)
#
#         # 注意力池化
#         pooled = self.attn_pool(combined)
#
#         # 残差连接
#         shortcut = self.shortcut(neigh_feats.mean(dim=-1))
#         return self.out_mlp(pooled) + shortcut
class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # 关键修复：存储d_out值
        self.d_out = d_out

        # 位置编码输出通道为d_out//2
        self.pos_encoder = RelativePosEncoder(10, d_out // 2)

        # 特征MLP输出通道为d_out//2
        self.feature_mlp = SharedMLP(d_in, d_out // 2, dim=2)

        # 注意力池化输入通道应为拼接后的总通道数(d_out//2 + d_out//2 = d_out)
        self.attn_pool = AttentionPooling(d_out)  # 修改为d_out

        # 输出MLP输入通道应为d_out//2（来自注意力池化输出）
        self.out_mlp = nn.Sequential(
            nn.Conv1d(d_out, d_out, 1, bias=False),  # 输入通道改为d_out
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential(
            nn.Conv1d(d_in, d_out, 1, bias=False),
            nn.BatchNorm1d(d_out)
        )

    def forward(self, center_xyz, neigh_xyz, neigh_feats):
        # 位置编码 (B, d_out//2, M, K)
        pos_feats = self.pos_encoder(center_xyz, neigh_xyz)

        # 特征处理 (B, d_out//2, M, K)
        feature_feats = self.feature_mlp(neigh_feats)

        # 拼接特征 (B, d_out, M, K)
        combined = torch.cat([feature_feats, pos_feats], dim=1)

        # 注意力池化 (B, d_out, M)
        pooled = self.attn_pool(combined)

        # 残差连接 (B, d_out, M)
        shortcut = self.shortcut(neigh_feats.mean(dim=-1))

        # 添加的形状检查（现在可以使用self.d_out）
        assert pos_feats.shape[1] == self.d_out // 2, f"位置编码通道错误: {pos_feats.shape}"
        assert feature_feats.shape[1] == self.d_out // 2, f"特征MLP通道错误: {feature_feats.shape}"
        assert combined.shape[1] == self.d_out, f"拼接后通道错误: {combined.shape}"

        # 最终输出 (B, d_out, M)
        return self.out_mlp(pooled) + shortcut

class RandLAUnit(nn.Module):
    def __init__(self, in_c, out_c, npoint, k):
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.lfa = LocalFeatureAggregation(in_c, out_c)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, bias=False),
            nn.BatchNorm1d(out_c)
        )

    def forward(self, xyz, feats, mask=None):
        B, C, N = feats.shape
        M = self.npoint

        # 随机采样
        sample_idx = torch.stack([torch.randperm(N, device=xyz.device)[:M] for _ in range(B)])
        batch_idx = torch.arange(B, device=xyz.device).view(B, 1)
        xyz_ds = xyz[batch_idx, sample_idx]

        # KNN搜索：在原始点云中搜索下采样点的邻域
        knn_idx = knn_point(self.k, xyz, xyz_ds)  # (B, M, K)

        # 提取邻域信息
        neigh_xyz = index_points(xyz, knn_idx)  # (B, M, K, 3)

        # 修复1：正确收集邻域特征
        knn_idx_exp = knn_idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, M, K)
        neigh_feats = feats.unsqueeze(-1).expand(-1, -1, -1, self.k)  # (B, C, N, K)
        neigh_feats = torch.gather(neigh_feats, 2, knn_idx_exp)  # (B, C, M, K)

        # 修复2：正确收集中心点特征
        sample_idx_exp = sample_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, M)
        center_feats = torch.gather(feats, 2, sample_idx_exp)  # (B, C, M)

        # 局部特征聚合
        f_lfa = self.lfa(xyz_ds, neigh_xyz, neigh_feats)
        f_sc = self.shortcut(center_feats)

        return xyz_ds, F.relu(f_lfa + f_sc, inplace=True), sample_idx

class RandLAUp(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz_coarse, feats_coarse, xyz_fine, feats_fine):
        # 最近邻插值
        dist = torch.cdist(xyz_fine, xyz_coarse)
        _, idx = dist.topk(1, dim=-1, largest=False)
        idx = idx.squeeze(-1)

        # 收集特征
        batch_idx = torch.arange(xyz_fine.size(0), device=xyz_fine.device).view(-1, 1, 1)
        idx_exp = idx.unsqueeze(1).expand(-1, feats_coarse.size(1), -1)
        feats_interp = feats_coarse.gather(2, idx_exp)

        # 特征融合
        fused_feats = torch.cat([feats_interp, feats_fine], dim=1)
        return self.mlp(fused_feats)


class RandLANetUNet(nn.Module):
    def __init__(self, input_c, m, nPlanes, cfg):
        super().__init__()
        # 初始点数
        self.num_points = cfg.num_points
        num_points = getattr(cfg, 'num_points', 40960)
        decimation = getattr(cfg, 'decimation', 4)
        num_neighbors = getattr(cfg, 'num_neighbors', 16)

        # 计算采样点数
        samples = [num_points // (decimation ** i) for i in range(5)]
        ks = [num_neighbors] * 5

        # 初始特征提取
        self.fc0 = nn.Sequential(
            nn.Conv1d(input_c, 8, 1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True)
        )

        # 编码器
        self.encs = nn.ModuleList([
            RandLAUnit(8, 16, samples[1], ks[0]),
            RandLAUnit(16, 64, samples[2], ks[1]),
            RandLAUnit(64, 128, samples[3], ks[2]),
            RandLAUnit(128, 256, samples[4], ks[3]),
            RandLAUnit(256, 512, samples[4] // 2, ks[4])
        ])

        # 解码器
        self.decs = nn.ModuleList([
            RandLAUp(512 + 256, 256),
            RandLAUp(256 + 128, 128),
            RandLAUp(128 + 64, 64),
            RandLAUp(64 + 16, 32),
            RandLAUp(32 + 8, 32)  # 第5次上采样
        ])

        # 输出层
        self.out_layer = nn.Sequential(
            nn.Conv1d(32, m, 1, bias=False),
            nn.BatchNorm1d(m),
            nn.ReLU(inplace=True)
        )

    # 在 RandLANetUNet 类中，替换整个 forward 函数

    def forward(self, batch):
        xyz = batch['xyz']
        features = batch.get('features')
        mask = batch.get('mask', None)

        if features is None:
            # 如果输入没有特征，就用坐标作为初始特征
            features = xyz.transpose(1, 2).contiguous()

        # 初始特征提取
        feats = self.fc0(features)  # (B, 8, N)

        # 存储各层特征
        xyzs = [xyz]
        feats_list = [feats]  # feats_list[0] 存储了 fc0 的输出

        # 编码器路径
        for enc in self.encs:
            xyz, feats, _ = enc(xyz, feats, mask)
            xyzs.append(xyz)
            feats_list.append(feats)

        # ===================== [ 关键修复：修正解码器循环逻辑 ] =====================
        # 解码器路径
        # feats 现在是 feats_list[-1]，即最后一个编码器的输出
        for i in range(len(self.decs)):
            # 从最深层开始，逐层向上融合
            # feats_fine 来自于跳跃连接 (skip connection)
            # i=0 时, feats_list[-(i+2)] 是 feats_list[-2], 即倒数第二个编码器的输出
            # i=4 时, feats_list[-(i+2)] 是 feats_list[-6], 即 feats_list[0], fc0的输出
            feats_fine = feats_list[-(i + 2)]
            xyz_fine = xyzs[-(i + 2)]

            # feats 是上一层解码器的输出
            feats = self.decs[i](xyz, feats, xyz_fine, feats_fine)
            xyz = xyz_fine  # 更新当前坐标为上采样后的坐标
        # =======================================================================

        # 最终的输出层
        return self.out_layer(feats)
