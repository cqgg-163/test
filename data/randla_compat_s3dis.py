import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from util.pointdata_process import dataAugment, elastic, get_xy_crop
from util import utils
import glob, os

class RandLACompatS3DIS(Dataset):
    def __init__(self, cfg, file_names, split='train'):
        self.file_names = file_names
        self.cfg = cfg
        self.split = split

        # 数据增强开关（仅训练）
        self.aug = (split == 'train')
        self.jit, self.flip, self.rot, self.elastic = True, True, True, True
        self.scale = cfg.scale
        self.crop_size = cfg.crop_size
        self.num_points = cfg.num_points

    def __getitem__(self, idx):
        fn = self.file_names[idx]
        data = torch.load(fn)
        xyz_origin, rgb, label = data[0], data[1], data[2].astype(np.int64)

        # 训练：随机裁剪 + 增强
        if self.split == 'train':
            while True:
                crop_idx = get_xy_crop(xyz_origin, self.crop_size)
                if len(crop_idx) > 3000:
                    break
            xyz_origin = xyz_origin[crop_idx]
            rgb = rgb[crop_idx]
            label = label[crop_idx]

        # 点数对齐（重复采样）
        n = len(xyz_origin)
        if n > self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)
        xyz_origin, rgb, label = xyz_origin[idx], rgb[idx], label[idx]

        # 数据增强
        if self.aug:
            xyz = dataAugment(xyz_origin, self.jit, self.flip, self.rot)
            xyz = xyz * self.scale
            if self.elastic:
                xyz = elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz -= xyz.min(0)
        else:
            xyz = xyz_origin - xyz_origin.min(0)

        # 归一化坐标
        xyz_float = xyz / self.scale
        rgb = (rgb / 127.5) - 1  # 匹配官方 [-1,1]

        # 构建 batch 格式
        return {
            'xyz': torch.from_numpy(xyz).float(),
            'xyz_float': torch.from_numpy(xyz_float).float(),
            'feats': torch.from_numpy(np.hstack([rgb, xyz_float])).float(),  # RGB+XYZ
            'labels': torch.from_numpy(label).long(),
            'item_fn': os.path.basename(fn),
        }

    def __len__(self):
        return len(self.file_names)