import torch
import numpy as np
import random
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import subprocess

import util.utils as utils
from model import build_network
from data import build_dataset
from util import commu_utils


def init_dist_pytorch(backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    assert cfg.batch_size % num_gpus == 0, f'Batch size should be matched with GPUS: ({cfg.batch_size}, {num_gpus})'
    cfg.batch_size = cfg.batch_size // num_gpus

    cfg.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(cfg.local_rank)

    print(f'[PID {os.getpid()}] rank: {cfg.local_rank} world_size: {num_gpus}')
    dist.init_process_group(backend=backend)


def init_dist_slurm(backend='nccl'):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_PORT'] = str(cfg.tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    assert cfg.batch_size % total_gpus == 0, f'Batch size should be matched with GPUS: ({cfg.batch_size}, {total_gpus})'
    cfg.batch_size = cfg.batch_size // total_gpus

    cfg.local_rank = dist.get_rank()


def init():
    cfg.task = 'test'
    cfg.result_dir = os.path.join(cfg.exp_path, 'result', f'iter{cfg.test_iter}', cfg.split)

    backup_dir = os.path.join(cfg.result_dir, 'backup_files')
    if cfg.local_rank == 0:
        os.makedirs(backup_dir, exist_ok=True)
        os.system(f'cp test.py {backup_dir}')
        os.system(f'cp {cfg.model_dir} {backup_dir}')
        os.system(f'cp {cfg.dataset_dir} {backup_dir}')
        os.system(f'cp {cfg.config} {backup_dir}')

    global logger
    from util.log import get_logger
    if cfg.local_rank == 0:
        logger = get_logger(cfg)
        logger.info(cfg)

    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


# In test.py
# 请确保文件顶部有这些导入
import os
import torch
import numpy as np
import util.utils as utils
from util.pointdata_process import get_xy_crop, dataAugment


def test(model, model_fn, dataset, dataloader, step):
    if cfg.local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Final Multi-Run TTA Voting Evaluation (Corrected) >>>>>>>>>>>>>>>>')

    collate_fn = dataset.get_batch_data
    file_list = dataset.test_file_names
    semantic_label_idx, semantic_names = utils.get_semantic_names(cfg.dataset)

    model.eval()

    test_reps = 3

    room_pred_scores_sum = {}
    room_labels = {}

    with torch.no_grad():
        for rep in range(test_reps):
            if cfg.local_rank == 0:
                logger.info(f'Running TTA Round {rep + 1}/{test_reps}...')

            for i, file_path in enumerate(file_list):
                cur_file_name = os.path.basename(file_path).split('.')[0]

                data = torch.load(file_path)
                xyz_origin, rgb_origin = data[0], data[1]
                labels_origin = data[2] if len(data) > 2 else None

                if rep == 0:
                    num_room_points = len(xyz_origin)
                    room_pred_scores_sum[cur_file_name] = np.zeros((num_room_points, cfg.classes), dtype=np.float32)
                    if labels_origin is not None:
                        room_labels[cur_file_name] = labels_origin

                # --- 正确的 TTA 逻辑 ---
                xyz_augmented = xyz_origin.copy()
                if rep == 1:  # 第2轮: 绕Z轴旋转180度
                    rot_matrix = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                                           [np.sin(np.pi), np.cos(np.pi), 0],
                                           [0, 0, 1]])
                    xyz_augmented[:, :3] = np.dot(xyz_augmented[:, :3], rot_matrix)
                elif rep == 2:  # 第3轮: XY平面翻转
                    xyz_augmented[:, 0] = -xyz_augmented[:, 0]
                    xyz_augmented[:, 1] = -xyz_augmented[:, 1]
                # -------------------------

                room_pred_scores_rep = np.zeros_like(room_pred_scores_sum[cur_file_name])
                room_vote_counts_rep = np.zeros(len(xyz_origin), dtype=np.int32)

                for _ in range(cfg.crop_max_iters):
                    idx_crop = get_xy_crop(xyz_augmented, crop_size=cfg.crop_size)
                    if len(idx_crop) < 1000: continue

                    xyz_crop, rgb_crop = xyz_augmented[idx_crop], rgb_origin[idx_crop]

                    if len(xyz_crop) > cfg.num_points:
                        choice = np.random.choice(len(xyz_crop), cfg.num_points, replace=False)
                    else:
                        choice = np.random.choice(len(xyz_crop), cfg.num_points, replace=True)

                    xyz_sampled, rgb_sampled = xyz_crop[choice], rgb_crop[choice]
                    xyz_normalized = xyz_sampled - np.mean(xyz_sampled, axis=0)

                    item = {'xyz': torch.from_numpy(xyz_normalized).float(),
                            'rgb': torch.from_numpy(rgb_sampled).float()}
                    input_batch = collate_fn([item])

                    for k, v in input_batch.items():
                        if isinstance(v, list):
                            input_batch[k] = [item.cuda() for item in v if torch.is_tensor(item)]
                        elif torch.is_tensor(v):
                            input_batch[k] = v.cuda()

                    ret = model(input_batch)
                    semantic_scores = ret['semantic_scores_l'].cpu().numpy()

                    global_indices_for_preds = idx_crop[choice]
                    np.add.at(room_pred_scores_rep, global_indices_for_preds, semantic_scores)
                    np.add.at(room_vote_counts_rep, global_indices_for_preds, 1)

                valid_mask = room_vote_counts_rep > 0
                room_pred_scores_sum[cur_file_name][valid_mask] += room_pred_scores_rep[valid_mask] / \
                                                                   room_vote_counts_rep[valid_mask][:, np.newaxis]

                if cfg.local_rank == 0:
                    logger.info(f"  Room {cur_file_name} (Round {rep + 1}) finished.")

        # ==================== 关键修改点 ====================
        # 将下面的最终评估逻辑整体缩进，使其位于 with torch.no_grad(): 代码块内部
        if cfg.local_rank == 0 and len(room_labels) > 0:
            logger.info('<<<<<<<<<<<<<<<<< Final TTA Voting Result >>>>>>>>>>>>>>>>>')
            intersection_total, union_total, target_total = [], [], []
            for file_name, scores in room_pred_scores_sum.items():
                final_preds = np.argmax(scores, axis=1)
                true_labels = room_labels[file_name]

                i, u, t = utils.intersectionAndUnion(final_preds, true_labels, cfg.classes, cfg.ignore_label)
                intersection_total.append(i)
                union_total.append(u)
                target_total.append(t)

            i_all = np.sum(intersection_total, axis=0)
            u_all = np.sum(union_total, axis=0)
            t_all = np.sum(target_total, axis=0)

            iou_class = i_all / (u_all + 1e-10)
            accuracy_class = i_all / (t_all + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = np.sum(i_all) / (np.sum(t_all) + 1e-10)

            utils.print_iou_acc_class(iou_class, accuracy_class, semantic_names, logger)
            logger.info(f'mIoU/mAcc/allAcc {mIoU * 100:.2f}/{mAcc * 100:.2f}/{allAcc * 100:.2f}.')
        # ====================================================
    

if __name__ == '__main__':
    # config
    global cfg
    from util.config import get_parser
    cfg = get_parser()

    # init
    if cfg.launcher == 'pytorch':
        init_dist_pytorch(backend='nccl')
        cfg.dist = True
    elif cfg.launcher == 'slurm':
        init_dist_slurm(backend='nccl')
        cfg.dist = True
    else:
        cfg.dist = False
    init()

    # get model and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    cfg.model_name = exp_name.split('_')[0]
    cfg.data_name = exp_name.split('_')[-1]

    # model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')
        logger.info(f'Classes: {cfg.classes}')

    cfg.pretrain_path = None
    model, model_fn = build_network(cfg, test=True)

    use_cuda = torch.cuda.is_available()
    if cfg.local_rank == 0:
        logger.info(f'cuda available: {use_cuda}')
    assert use_cuda
    model = model.cuda()

    if cfg.dist:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.local_rank % num_gpus
        if cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if cfg.local_rank == 0:
        logger.info(f'#model parameters: {sum([x.nelement() for x in model.parameters()])}')

    # load model
    _, f = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], cfg.test_iter,
                                    f=cfg.pretrain, dist=cfg.dist, gpu=cfg.local_rank % torch.cuda.device_count())
    if cfg.local_rank == 0:
        logger.info(f'Restore from {f}')

    # data
    dataset = build_dataset(cfg, test=True)
    dataset.testLoader()
    dataloader = dataset.test_data_loader
    if cfg.local_rank == 0:
        logger.info(f'Testing samples ({cfg.split}): {dataset.test_file_names.__len__()}')

    # evaluate
    test(model, model_fn, dataset, dataloader, cfg.test_iter)
