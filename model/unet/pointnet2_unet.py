import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 你需要安装pointnet2_ops（https://github.com/erikwijmans/Pointnet2_PyTorch）或其它实现，
# 或者可用torch-points-kernels, torch-points3d等的相关模块。
# 这里以pointnet2_ops为例，若无可自行实现或替换。

# from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule


from model.unet.pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
_HAS_POINTNET2 = True

# try:
#     from model.unet.pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
# except ImportError:
#     PointnetSAModule = None
#     PointnetFPModule = None
#     print("Warning: Please install pointnet2_ops library for Pointnet++ modules.")

# ----------- PointNet++ Set Abstraction (SA) Block -----------
class PointNet2_SA_Block(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, use_bn=True):
        super().__init__()
        if PointnetSAModule is None:
            raise ImportError("PointnetSAModule not found. Please install pointnet2_ops.")
        self.sa_module = PointnetSAModule(
            npoint=npoint,
            radius=radius,
            nsample=nsample,
            mlp=[in_channel] + mlp,
            use_xyz=True,
            bn=use_bn,
            # group_all=group_all
        )

    def forward(self, xyz, features):
        # xyz: (B, N, 3), features: (B, C, N)
        new_xyz, new_features = self.sa_module(xyz, features)
        return new_xyz, new_features



# ----------- PointNet++ Feature Propagation (FP) Block -----------
class PointNet2_FP_Block(nn.Module):
    def __init__(self, in_channel, mlp, use_bn=True):
        super().__init__()
        if PointnetFPModule is None:
            raise ImportError("PointnetFPModule not found. Please install pointnet2_ops.")
        self.fp_module = PointnetFPModule(
            mlp=[in_channel] + mlp,
            bn=use_bn
        )

    def forward(self, xyz1, xyz2, feat1, feat2):
        # xyz1: (B, N, 3), xyz2: (B, S, 3), feat1: (B, C1, N), feat2: (B, C2, S)
        new_features = self.fp_module(xyz1, xyz2, feat1, feat2)
        return new_features

# ----------- PointNet++ U-Net-like Backbone -----------
class PointNet2UNet(nn.Module):
    def __init__(self, input_c, m, nPlanes, block_reps, block, norm_fn, model_cfg={}):
        """
        input_c: 输入特征通道数
        m: 第一层输出特征通道数
        nPlanes: 每个SA/FP阶段输出通道数列表（如 [32, 64, 128, 256]）
        block_reps, block, norm_fn: 兼容原spconv接口，无实际作用
        model_cfg: 额外配置
        """
        super().__init__()
        self.nPlanes = nPlanes
        self.SA_modules = nn.ModuleList()
        self.FP_modules = nn.ModuleList()

        # Set Abstraction (Encoder/Downsampling)
        # 假设输入为 (B, input_c, N)，坐标为 (B, N, 3)
        in_channels = input_c
        sa_cfg = model_cfg.get('sa_cfg', [
            # npoint, radius, nsample, [mlp...]
            [1024, 0.1, 32, [m]],
            [256, 0.2, 32, [nPlanes[1]]],
            [64, 0.4, 32, [nPlanes[2]]],
            [16, 0.8, 32, [nPlanes[3]]]
        ])
        num_stages = len(sa_cfg)
        for i in range(num_stages):
            npoint, radius, nsample, mlp = sa_cfg[i]
            self.SA_modules.append(
                PointNet2_SA_Block(npoint, radius, nsample, in_channels, mlp, use_bn=True)
            )
            in_channels = mlp[-1]

        # Feature Propagation (Decoder/Upsampling)
        fp_cfg = model_cfg.get('fp_cfg', [
            # [in_channel, [mlp...]]
            [nPlanes[3]+nPlanes[2], [nPlanes[2]]],
            [nPlanes[2]+nPlanes[1], [nPlanes[1]]],
            [nPlanes[1]+nPlanes[0], [nPlanes[0]]],
            [nPlanes[0]+input_c, [m]]
        ])
        for in_ch, mlp in fp_cfg:
            self.FP_modules.append(PointNet2_FP_Block(in_ch, mlp, use_bn=True))

        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(m), nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        pointcloud['xyz']: (B, N, 3)
        pointcloud['features']: (B, input_c, N) 或 None（如果只用xyz）
        """
        xyz = pointcloud['xyz']  # (B, N, 3)
        features = pointcloud.get('features', None)  # (B, input_c, N) or None

        # Encoder: Set Abstraction
        xyzs = [xyz]
        features_list = [features]
        for sa in self.SA_modules:
            xyz, features = sa(xyz, features)
            xyzs.append(xyz)
            features_list.append(features)

        # Decoder: Feature Propagation
        for i in range(len(self.FP_modules)):
            l = -i-1
            features = self.FP_modules[i](
                xyzs[l-1], xyzs[l], features_list[l-1], features
            )

        # Output layer
        output = self.output_layer(features)
        return output  # (B, m, N)

# --------- Example usage for replacement ---------
# self.backbone = UNet(input_c, m, nPlanes, block_reps, block, norm_fn, cfg)
#
# 替换为：
# self.backbone = PointNet2UNet(input_c, m, nPlanes, block_reps, block, norm_fn, cfg)
#
# 前向调用保持一致，但输入点云需传dict，包含'xyz'和'features'（后者可为None）。

