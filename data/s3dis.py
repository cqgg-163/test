#
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset
# import SharedArray as SA
# import os, sys, glob
# sys.path.append('../')
#
# # from lib.ops import voxelization_idx
# from util.pointdata_process import dataAugment, elastic, get_overlap_xy_crops, get_xy_crop
# from util import utils
#
#
# from sklearn.neighbors import NearestNeighbors
# def knn_search(support_pts, query_pts, k):
#     """
#     对批处理数据进行KNN搜索。
#     :param support_pts: (B, N1, 3) numpy array, 在这些点中搜索
#     :param query_pts: (B, N2, 3) numpy array, 为这些点查找邻居
#     :param k: int, 邻居数量
#     :return: (B, N2, k) numpy array, 邻居索引
#     """
#     all_neighbors = []
#     for i in range(support_pts.shape[0]):
#         nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(support_pts[i])
#         indices = nbrs.kneighbors(query_pts[i], return_distance=False)
#         all_neighbors.append(indices)
#     return np.stack(all_neighbors).astype(np.int32)
#
#
# class S3DIS(Dataset):
#
#     def __init__(self, cfg, file_names, aug_options, test=False, unlabeled=False):
#         super(S3DIS, self).__init__()
#         self.cache = cfg.cache
#         self.dist = cfg.dist
#         self.local_rank = cfg.local_rank
#
#         self.scale = cfg.scale
#         self.file_names = file_names
#         if not self.cache:
#             self.files = [torch.load(i) for i in self.file_names]
#         else:
#             num_gpus = 1 if not self.dist else torch.cuda.device_count()
#             rk = self.local_rank % num_gpus
#             utils.create_shared_memory(file_names[rk::num_gpus])
#
#         self.jit, self.flip, self.rot, self.elastic, self.crop, self.rgb_aug, self.subsample = aug_options
#         self.flipv, self.rotv = None, None
#
#         self.unlabeled = unlabeled
#         self.crop_size = cfg.crop_size
#         self.crop_max_iters = cfg.crop_max_iters
#         self.test = test
#
#         self.aug_options = aug_options
#
#         self.num_points = 65536
#
#     def item_process(self, xyz_origin, rgb, label=None):
#         # 参考实现的数据归一化
#         xyz = self.normalize_to_unit_cube(xyz_origin)
#
#         # 参考实现的数据增强
#         if self.rot:
#             # 仅绕Z轴旋转
#             theta = np.random.uniform(0, 2 * np.pi)
#             rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                 [np.sin(theta), np.cos(theta), 0],
#                                 [0, 0, 1]])
#             xyz = np.dot(xyz, rot_mat)
#
#         if self.flip:
#             # 仅XY平面翻转
#             if np.random.rand() > 0.5: xyz[:, 0] *= -1
#             if np.random.rand() > 0.5: xyz[:, 1] *= -1
#
#         if self.jit:
#             # 较小幅度的抖动
#             xyz += np.random.normal(0, 0.005, size=xyz.shape)
#
#         # 保持原始比例
#         xyz_float = xyz.copy()
#
#         item = {
#             'xyz': torch.from_numpy(xyz).float(),
#             'xyz_float': torch.from_numpy(xyz_float).float(),
#             'rgb': torch.from_numpy(rgb).float()
#         }
#
#         if label is not None:
#             item['label'] = torch.from_numpy(label).long()
#
#         return item
#
#     def process_single_view(self, xyz_in, rgb_in, label_in=None):
#         """ 辅助函数：对单个点云视图进行采样和归一化 """
#         # 1. 固定点数采样
#         if len(xyz_in) > self.num_points:
#             idxs = np.random.choice(len(xyz_in), self.num_points, replace=False)
#         else:
#             replace = not self.test  # 测试时不做重复采样
#             idxs = np.random.choice(len(xyz_in), self.num_points, replace=replace)
#
#         xyz, rgb = xyz_in[idxs], rgb_in[idxs]
#         label = None if label_in is None else label_in[idxs]
#
#         # 2. 正确的局部归一化：去中心化
#         xyz_normalized = xyz - np.mean(xyz, axis=0)
#
#         # 3. 数据增强 (您可以将原有的增强逻辑放在这里)
#         jit, flip, rot = self.aug_options[0], self.aug_options[1], self.aug_options[2]
#         if rot:
#             theta = np.random.uniform(0, 2 * np.pi)
#             rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
#                                 [np.sin(theta), np.cos(theta), 0],
#                                 [0, 0, 1]], dtype=np.float32)
#             xyz_normalized = np.dot(xyz_normalized, rot_mat)
#         # ... 其他 flip, jit 增强 ...
#
#         # 4. 准备返回字典 (只包含最核心的数据)
#         item = {
#             'xyz': torch.from_numpy(xyz_normalized).float(),
#             'rgb': torch.from_numpy(rgb).float(),
#         }
#         if label is not None:
#             item['label'] = torch.from_numpy(label).long()
#         return item
#     def __getitem__(self, id):
#         fn = self.file_names[id].split('/')[-1].split('.')[0]
#         if self.cache:
#             xyz_origin_full = SA.attach(f'shm://{fn}_xyz').copy()
#             rgb_full = SA.attach(f'shm://{fn}_rgb').copy()
#             label_full = None if self.test else SA.attach(f'shm://{fn}_label').copy()
#         else:
#             data = self.files[id]
#             xyz_origin_full, rgb_full = data[0], data[1]
#             label_full = None if self.test else data[2]
#
#         if self.test:
#             item = {
#                 # 注意：这里我们不进行任何处理，直接返回原始数据
#                 # 后续的处理将在投票评估函数中进行，以确保一致性
#                 'xyz_origin': torch.from_numpy(xyz_origin_full).float(),
#                 'rgb': torch.from_numpy(rgb_full).float(),
#             }
#             if label_full is not None:
#                 item['label'] = torch.from_numpy(label_full).long()
#
#             item['item_id'] = id
#             item['item_fn'] = fn
#
#             return item
#         if self.unlabeled:
#             # --- 无监督路径 ---
#             # 1. 创建两个重叠视图 (保留原有逻辑)
#             # 注意：原有的 get_overlap_xy_crops 是基于体素的，这里假设它返回点索引
#             # 您需要确保这里的 crop 逻辑是合理的。为了演示，我们假设它返回点索引。
#             # 这里我们用一个简化的随机crop来模拟
#             n_full = len(xyz_origin_full)
#             idx1 = np.random.choice(n_full, int(n_full * 0.8), replace=False)
#             idx2 = np.random.choice(n_full, int(n_full * 0.8), replace=False)
#
#             xyz1, rgb1 = xyz_origin_full[idx1], rgb_full[idx1]
#             xyz2, rgb2 = xyz_origin_full[idx2], rgb_full[idx2]
#
#             # 2. 独立处理每个视图
#             item1 = self.process_single_view(xyz1, rgb1)
#             item2 = self.process_single_view(xyz2, rgb2)
#
#             # 3. 组合成对比学习对
#             item = {}  # 'idx'和'idxo' 在此简化方案中省略，因为它们依赖于精确的crop逻辑
#             for k in item1.keys():
#                 item[k] = (item1[k], item2[k])
#         else:
#             # --- 监督与测试路径 ---
#             # 直接将整个场景（或测试时的crop）送去处理
#             item = self.process_single_view(xyz_origin_full, rgb_full, label_full)
#
#         item['item_id'] = id
#         item['item_fn'] = fn
#         return item
#
#     def __len__(self):
#         return len(self.file_names)
#
# class MyDataset:
#     def __init__(self, cfg, test=False):
#         self.cfg = cfg
#         self.data_root = cfg.data_root
#         self.dataset = cfg.dataset
#         self.test_area = cfg.test_area
#         self.train_area = cfg.train_area
#         self.batch_size = cfg.batch_size
#         self.train_workers = cfg.train_workers
#         self.val_workers = cfg.train_workers
#         self.dist = cfg.dist
#         self.train_flip = cfg.train_flip
#         self.train_rot = cfg.train_rot
#         self.train_jit = cfg.train_jit
#         self.train_elas = cfg.train_elas
#         self.train_subsample = cfg.train_subsample
#         self.full_scale = cfg.full_scale
#         self.labeled_ratio = cfg.labeled_ratio
#         if test:
#             self.test_workers = cfg.test_workers
#             cfg.batch_size = 1
#             self.batch_size = 1
#
#     def trainLoader(self):
#         self.train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 's3dis', self.train_area + '*.pth')))
#         if self.labeled_ratio < 1:
#             split_path = os.path.join(self.data_root, self.dataset, 'data_split')
#             split_fn = os.path.join(split_path, f'{int(self.labeled_ratio * 100)}.txt')
#             with open(split_fn) as f:
#                 l_fns_ = f.readlines()
#                 l_fns_ = [i.strip() for i in l_fns_]
#             self.labeled_fns = [os.path.join(self.data_root, self.dataset, 's3dis', f'{fn}.pth') for fn in l_fns_]
#             self.unlabeled_fns = [fn for fn in self.train_file_names if fn not in self.labeled_fns]
#         else:
#             self.labeled_fns, self.unlabeled_fns = self.train_file_names, self.train_file_names
#
#         l_train_set = S3DIS(self.cfg, self.labeled_fns,
#                             [self.train_jit, self.train_flip, self.train_rot, self.train_elas, True, True, self.train_subsample],
#                             unlabeled=False)
#         u_train_set = S3DIS(self.cfg, self.unlabeled_fns,
#                             [self.train_jit, self.train_flip, self.train_rot, self.train_elas, True, True, self.train_subsample],
#                             unlabeled=True)
#         self.l_train_sampler = torch.utils.data.distributed.DistributedSampler(
#             l_train_set, shuffle=True, drop_last=False) if self.dist else None
#         self.l_train_data_loader = DataLoader(l_train_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
#                                               num_workers=self.train_workers, shuffle=(self.l_train_sampler is None),
#                                               sampler=self.l_train_sampler, drop_last=False, pin_memory=True,
#                                               worker_init_fn=self._worker_init_fn_)
#         self.u_train_sampler = torch.utils.data.distributed.DistributedSampler(
#             u_train_set, shuffle=True, drop_last=False) if self.dist else None
#         self.u_train_data_loader = DataLoader(u_train_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
#                                               num_workers=self.train_workers, shuffle=(self.u_train_sampler is None),
#                                               sampler=self.u_train_sampler, drop_last=False, pin_memory=True,
#                                               worker_init_fn=self._worker_init_fn_)
#
#     def valLoader(self):
#         self.val_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 's3dis', self.test_area + '*.pth')))
#         val_set = S3DIS(self.cfg, self.val_file_names,
#                         [False, False, False, False, False, False, False], unlabeled=False)
#         self.val_sampler = torch.utils.data.distributed.DistributedSampler(
#             val_set, shuffle=False, drop_last=False) if self.dist else None
#         self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
#                                           num_workers=self.val_workers, shuffle=False, sampler=self.val_sampler,
#                                           drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)
#
#     def testLoader(self):
#         self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 's3dis', self.test_area + '*.pth')))
#         self.test_set = S3DIS(self.cfg, self.test_file_names,
#                               [False, True, True, False, False, False, False],
#                               test=False, unlabeled=False)
#         self.test_sampler = torch.utils.data.distributed.DistributedSampler(
#             self.test_set, shuffle=False, drop_last=False) if self.dist else None
#         self.test_data_loader = DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
#                                            num_workers=self.test_workers, shuffle=False, sampler=self.test_sampler,
#                                            drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)
#
#     def _worker_init_fn_(self, worker_id):
#         torch_seed = torch.initial_seed()
#         np_seed = torch_seed % 2 ** 32 - 1
#         np.random.seed(np_seed)
#
#     def tf_map(self, batch_xyz):
#         """
#         为批处理数据生成多层级的下采样和KNN索引。
#         (此函数为您已有的版本，无需改动)
#         """
#         # S3DIS的固定配置
#         num_layers = 5
#         sub_sampling_ratio = [4, 4, 4, 2, 2]
#         k_n = 16
#
#         input_points = []
#         input_neighbors = []
#         input_pools = []
#         input_up_samples = []
#
#         for i in range(num_layers):
#             neighbour_idx = knn_search(batch_xyz, batch_xyz, k_n)
#             sub_points = batch_xyz[:, :batch_xyz.shape[1] // sub_sampling_ratio[i], :]
#             pool_i = neighbour_idx[:, :batch_xyz.shape[1] // sub_sampling_ratio[i], :]
#             up_i = knn_search(sub_points, batch_xyz, 1)
#
#             input_points.append(batch_xyz)
#             input_neighbors.append(neighbour_idx)
#             input_pools.append(pool_i)
#             input_up_samples.append(up_i)
#             batch_xyz = sub_points
#
#         indices = {
#             'xyz': [torch.from_numpy(p).float() for p in input_points],
#             'neigh_idx': [torch.from_numpy(n).long() for n in input_neighbors],
#             'sub_idx': [torch.from_numpy(p).long() for p in input_pools],
#             'interp_idx': [torch.from_numpy(u).long() for u in input_up_samples]
#         }
#         return indices
#     def get_batch_data(self, item_list):
#         unlabeled = 'idxo' in item_list[0]
#         test = 'label' not in item_list[0]
#
#         # 辅助函数: 智能解包列表
#         def get_value_list(key):
#             values = []
#             for item in item_list:
#                 value = item.get(key)
#                 if value is None: continue
#                 if isinstance(value, tuple):
#                     values.extend(list(value))
#                 else:
#                     values.append(value)
#             return values
#
#         # 1. 解包核心数据
#         xyz_list = get_value_list('xyz')
#         rgb_list = get_value_list('rgb')
#
#         batch_xyz_tensor = torch.stack(xyz_list)
#         batch_rgb_tensor = torch.stack(rgb_list)
#
#         # 2. 生成RandLA-Net的结构化输入
#         randla_indices = self.tf_map(batch_xyz_tensor.numpy())
#         batch_data = randla_indices
#
#         # 3. 组合6D特征
#         features_6d = torch.cat([batch_data['xyz'][0], batch_rgb_tensor], dim=-1)
#         batch_data['feats'] = features_6d.contiguous()
#
#         # 4. 处理标签 (仅监督模式)
#         if not test and not unlabeled:
#             batch_data['labels'] = torch.stack(get_value_list('label'))
#
#
#         batch_offsets = torch.arange(0, batch_xyz_tensor.shape[0] + 1) * batch_xyz_tensor.shape[1]
#         batch_data['offsets'] = batch_offsets.to(torch.int32)  # 类型保持一致
#         # 5. [关键恢复] 完整保留无监督元数据的处理逻辑
#         if unlabeled:
#             # a. 计算 batch_offsets，这是调整索引所必需的
#             batch_offsets = [0]
#             for value in xyz_list:
#                 batch_offsets.append(batch_offsets[-1] + value.shape[0])
#             batch_offsets = torch.tensor(batch_offsets, dtype=torch.int32)
#             batch_data['offsets'] = batch_offsets  # 将 offsets 也加入 batch_data
#
#             # b. 处理 idxos
#             idxos = get_value_list('idxo')
#             idxos_adjusted = [idx + int(batch_offsets[i]) for i, idx in enumerate(idxos)]
#
#             idxos1 = torch.cat([torch.from_numpy(i).long() for i in idxos_adjusted[0::2]], 0)
#             idxos2 = torch.cat([torch.from_numpy(i).long() for i in idxos_adjusted[1::2]], 0)
#             batch_data['idxos'] = (idxos1, idxos2)
#
#             # c. 处理 point_cnts_o
#             point_cnts_o = torch.tensor([i.shape[0] for i in idxos[0::2]], dtype=torch.int32)
#             batch_data['point_cnts_o'] = point_cnts_o
#         batch_data['file_names'] = get_value_list('item_fn')
#         batch_data['id'] = get_value_list('item_id')
#         return batch_data
#
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import SharedArray as SA
import os, sys, glob

from util.pointdata_process import elastic, get_overlap_xy_crops, get_xy_crop
from util import utils

from sklearn.neighbors import NearestNeighbors


def knn_search(support_pts, query_pts, k):
    all_neighbors = []
    for i in range(support_pts.shape[0]):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(support_pts[i])
        indices = nbrs.kneighbors(query_pts[i], return_distance=False)
        all_neighbors.append(indices)
    return np.stack(all_neighbors).astype(np.int32)


class S3DIS(Dataset):
    def __init__(self, cfg, file_names, aug_options, test=False, unlabeled=False):
        super(S3DIS, self).__init__()
        self.cache = cfg.cache
        self.dist = cfg.dist
        self.local_rank = cfg.local_rank

        self.scale = cfg.scale
        # 保持与 GuidedContrast 一致的配置
        self.crop_size = cfg.crop_size if hasattr(cfg, 'crop_size') else [4.0, 4.0]
        self.crop_max_iters = cfg.crop_max_iters if hasattr(cfg, 'crop_max_iters') else 50
        self.file_names = file_names

        if self.cache:
            num_gpus = 1 if not self.dist else torch.cuda.device_count()
            rk = self.local_rank % num_gpus
            utils.create_shared_memory(file_names[rk::num_gpus])

        # 保持原有 aug_options 处理方式
        self.jit, self.flip, self.rot, self.elastic = aug_options[:4]
        self.rgb_aug = aug_options[5] if len(aug_options) > 5 else False
        self.subsample = aug_options[6] if len(aug_options) > 6 else False

        self.unlabeled = unlabeled
        self.test = test
        # 使用配置中的 num_points
        self.num_points = cfg.num_points if hasattr(cfg, 'num_points') else 40960

    def geometric_augment(self, xyz):
        """几何增强：保持与 GuidedContrast 一致"""
        if self.rot:
            theta = np.random.uniform(0, 2 * np.pi)
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
            xyz = np.dot(xyz, rot_mat)

        if self.flip:
            if np.random.rand() > 0.5: xyz[:, 0] *= -1
            if np.random.rand() > 0.5: xyz[:, 1] *= -1

        if self.jit:
            xyz += np.random.normal(0, 0.005, size=xyz.shape)

        return xyz

    def item_process(self, xyz_origin, rgb, label=None):
        """处理单个视图的数据 - 保持与 GuidedContrast 一致"""
        # 弹性变形（仅在训练时）
        if self.elastic and not self.test:
            xyz_origin = elastic(xyz_origin, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz_origin = elastic(xyz_origin, 20 * self.scale // 50, 160 * self.scale / 50)

        # 几何增强
        xyz_middle = self.geometric_augment(xyz_origin)

        # 去中心化（局部归一化）
        xyz = xyz_middle - xyz_middle.mean(0)

        # 点数控制 - 保持原有逻辑
        n = xyz.shape[0]
        if n > self.num_points:
            idxs = np.random.choice(n, self.num_points, replace=False)
        else:
            # 测试时不做重复采样
            replace = not self.test if label is not None else False
            idxs = np.random.choice(n, self.num_points, replace=replace)

        xyz, rgb = xyz[idxs], rgb[idxs]
        if label is not None:
            label = label[idxs]

        item = {
            'xyz': torch.from_numpy(xyz).float(),
            'rgb': torch.from_numpy(rgb).float()
        }
        if label is not None:
            item['label'] = torch.from_numpy(label).long()

        return item

    def __getitem__(self, id):
        fn = self.file_names[id].split('/')[-1].split('.')[0]
        if self.cache:
            xyz_origin_full = SA.attach(f'shm://{fn}_xyz').copy()
            rgb_full = SA.attach(f'shm://{fn}_rgb').copy()
            label_full = None if self.test else SA.attach(f'shm://{fn}_label').copy()
        else:
            data = torch.load(self.file_names[id])
            xyz_origin_full, rgb_full = data[0], data[1]
            label_full = None if self.test else data[2]

        # 测试模式保持原有逻辑
        if self.test:
            item = {
                'xyz_origin': torch.from_numpy(xyz_origin_full).float(),
                'rgb': torch.from_numpy(rgb_full).float(),
            }
            if label_full is not None:
                item['label'] = torch.from_numpy(label_full).long()
            item['item_id'] = id
            item['item_fn'] = fn
            return item

        if self.unlabeled:
            # 无监督模式：生成重叠视图
            # 保持与 GuidedContrast 完全一致的逻辑
            while True:
                result = get_overlap_xy_crops(xyz_origin_full, self.crop_size, self.crop_max_iters)
                if result is not None and len(result) >= 5 and result[3].shape[0] > 1000:
                    idx1, idx2, idxo, idxo1, idxo2 = result
                    break

            xyz1, rgb1 = xyz_origin_full[idx1], rgb_full[idx1]
            xyz2, rgb2 = xyz_origin_full[idx2], rgb_full[idx2]

            # 独立处理每个视图
            item1 = self.item_process(xyz1, rgb1)
            item2 = self.item_process(xyz2, rgb2)

            # 组合成对比学习对
            item = {}
            for k in item1.keys():
                item[k] = (item1[k], item2[k])
            # 保留重叠索引信息
            item['idxo'] = (idxo1, idxo2)
        else:
            # 监督模式：生成单个视图
            while True:
                idx = get_xy_crop(xyz_origin_full, self.crop_size)
                if idx.shape[0] > 3000:
                    break

            xyz_crop, rgb_crop = xyz_origin_full[idx], rgb_full[idx]
            label_crop = label_full[idx] if label_full is not None else None
            item = self.item_process(xyz_crop, rgb_crop, label_crop)

        item['item_id'] = id
        item['item_fn'] = fn
        return item

    def __len__(self):
        return len(self.file_names)


class MyDataset:
    def __init__(self, cfg, test=False):
        self.cfg = cfg
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.test_area = cfg.test_area
        self.train_area = cfg.train_area
        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers
        self.dist = cfg.dist
        self.train_flip = cfg.train_flip
        self.train_rot = cfg.train_rot
        self.train_jit = cfg.train_jit
        self.train_elas = cfg.train_elas
        self.train_subsample = cfg.train_subsample
        self.full_scale = cfg.full_scale
        self.labeled_ratio = cfg.labeled_ratio

        if test:
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1
            self.batch_size = 1

    def trainLoader(self):
        self.train_file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, 's3dis', self.train_area + '*.pth')))

        if self.labeled_ratio < 1:
            split_path = os.path.join(self.data_root, self.dataset, 'data_split')
            split_fn = os.path.join(split_path, f'{int(self.labeled_ratio * 100)}.txt')
            with open(split_fn) as f:
                l_fns_ = [i.strip() for i in f.readlines()]
            self.labeled_fns = [os.path.join(self.data_root, self.dataset, 's3dis', f'{fn}.pth') for fn in l_fns_]
            self.unlabeled_fns = [fn for fn in self.train_file_names if fn not in self.labeled_fns]
        else:
            self.labeled_fns, self.unlabeled_fns = self.train_file_names, self.train_file_names

        # 保持与原有项目一致的增强选项传递
        aug_options = [self.train_jit, self.train_flip, self.train_rot, self.train_elas, True, True,
                       self.train_subsample]

        l_train_set = S3DIS(self.cfg, self.labeled_fns, aug_options, unlabeled=False)
        u_train_set = S3DIS(self.cfg, self.unlabeled_fns, aug_options, unlabeled=True)

        self.l_train_sampler = torch.utils.data.distributed.DistributedSampler(
            l_train_set, shuffle=True, drop_last=False) if self.dist else None
        self.l_train_data_loader = DataLoader(l_train_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                              num_workers=self.train_workers, shuffle=(self.l_train_sampler is None),
                                              sampler=self.l_train_sampler, drop_last=False, pin_memory=True,
                                              worker_init_fn=self._worker_init_fn_)

        self.u_train_sampler = torch.utils.data.distributed.DistributedSampler(
            u_train_set, shuffle=True, drop_last=False) if self.dist else None
        self.u_train_data_loader = DataLoader(u_train_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                              num_workers=self.train_workers, shuffle=(self.u_train_sampler is None),
                                              sampler=self.u_train_sampler, drop_last=False, pin_memory=True,
                                              worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        self.val_file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, 's3dis', self.test_area + '*.pth')))
        # 验证集关闭所有增强
        aug_options = [False, False, False, False, False, False, False]
        val_set = S3DIS(self.cfg, self.val_file_names, aug_options, unlabeled=False)

        self.val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, shuffle=False, drop_last=False) if self.dist else None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                          num_workers=self.val_workers, shuffle=False, sampler=self.val_sampler,
                                          drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)

    def testLoader(self):
        self.test_file_names = sorted(
            glob.glob(os.path.join(self.data_root, self.dataset, 's3dis', self.test_area + '*.pth')))
        # 测试时通常只开启基础增强
        aug_options = [False, True, True, False, False, False, False]
        self.test_set = S3DIS(self.cfg, self.test_file_names, aug_options, test=False, unlabeled=False)

        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.test_set, shuffle=False, drop_last=False) if self.dist else None
        self.test_data_loader = DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.get_batch_data,
                                           num_workers=self.test_workers, shuffle=False, sampler=self.test_sampler,
                                           drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)

    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed % 2 ** 32 - 1
        np.random.seed(np_seed)

    def get_batch_data(self, item_list):
        """保持与原有项目完全一致的 batch_data 构造逻辑"""
        if not item_list:
            return {}

        # 判断数据类型 - 保持原有逻辑
        unlabeled = 'idxo' in item_list[0]
        test = 'label' not in item_list[0]

        # 辅助函数: 智能解包列表 - 保持原有逻辑
        def get_value_list(key):
            values = []
            for item in item_list:
                value = item.get(key)
                if value is None: continue
                if isinstance(value, tuple):
                    values.extend(list(value))
                else:
                    values.append(value)
            return values

        # 1. 解包核心数据 - 保持原有逻辑
        xyz_list = get_value_list('xyz')
        rgb_list = get_value_list('rgb')

        if not xyz_list:
            return {}

        batch_xyz_tensor = torch.stack(xyz_list)
        batch_rgb_tensor = torch.stack(rgb_list)

        # 2. 生成RandLA-Net的结构化输入 - 保持原有优秀设计
        randla_indices = self.tf_map(batch_xyz_tensor.numpy())
        batch_data = randla_indices

        # 3. 组合6D特征 - 保持原有逻辑
        features_6d = torch.cat([batch_data['xyz'][0], batch_rgb_tensor], dim=-1)
        batch_data['feats'] = features_6d.contiguous()

        # 4. 处理标签 (仅监督模式) - 保持原有逻辑
        if not test and not unlabeled:
            labels_list = get_value_list('label')
            if labels_list:
                batch_data['labels'] = torch.stack(labels_list)

        # 5. [关键恢复] 完整保留无监督元数据的处理逻辑
        if unlabeled:
            # a. 计算 batch_offsets，这是调整索引所必需的
            # 保持与原有项目一致的计算方式
            batch_offsets = [0]
            for value in xyz_list:
                batch_offsets.append(batch_offsets[-1] + value.shape[0])
            batch_offsets = torch.tensor(batch_offsets, dtype=torch.int32)
            batch_data['offsets'] = batch_offsets

            # b. 处理 idxos - 保持原有项目的索引调整逻辑
            idxos = get_value_list('idxo')
            idxos_adjusted = [idx + int(batch_offsets[i]) for i, idx in enumerate(idxos)]

            idxos1 = torch.cat([torch.from_numpy(i).long() for i in idxos_adjusted[0::2]], 0)
            idxos2 = torch.cat([torch.from_numpy(i).long() for i in idxos_adjusted[1::2]], 0)
            batch_data['idxos'] = (idxos1, idxos2)

            # c. 处理 point_cnts_o - 保持原有逻辑
            point_cnts_o = torch.tensor([i.shape[0] for i in idxos[0::2]], dtype=torch.int32)
            batch_data['point_cnts_o'] = point_cnts_o

        batch_data['file_names'] = get_value_list('item_fn')
        batch_data['id'] = get_value_list('item_id')
        return batch_data

    def tf_map(self, batch_xyz):
        """保持原有项目的 tf_map 实现"""
        # S3DIS的固定配置
        num_layers = 5
        sub_sampling_ratio = [4, 4, 4, 2, 2]
        k_n = 16

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(num_layers):
            neighbour_idx = knn_search(batch_xyz, batch_xyz, k_n)
            # 确保下采样后至少有1个点
            sub_count = max(1, batch_xyz.shape[1] // sub_sampling_ratio[i])
            sub_points = batch_xyz[:, :sub_count, :]
            pool_i = neighbour_idx[:, :sub_count, :]
            up_i = knn_search(sub_points, batch_xyz, 1)

            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        indices = {
            'xyz': [torch.from_numpy(p).float() for p in input_points],
            'neigh_idx': [torch.from_numpy(n).long() for n in input_neighbors],
            'sub_idx': [torch.from_numpy(p).long() for p in input_pools],
            'interp_idx': [torch.from_numpy(u).long() for u in input_up_samples]
        }
        return indices
