import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import numpy as np
import time, os, random, subprocess

from torch import nn

import util.utils as utils
from model import build_network
from data import build_dataset
from util.lr import initialize_scheduler


def init_dist_pytorch(backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    assert cfg.batch_size % num_gpus == 0, f'Batch size should be matched with GPUS: ({cfg.batch_size}, {num_gpus})'
    cfg.batch_size = cfg.batch_size // num_gpus

    cfg.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(cfg.local_rank)
#读取当前进程的local_rank环境变量，并指定使用哪块GPU。
    print(f'[PID {os.getpid()}] rank: {cfg.local_rank} world_size: {num_gpus}')
    dist.init_process_group(backend=backend)
#多处指定了卡0。设置进程启动方式

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
    # 设置主端口与地址，
    total_gpus = dist.get_world_size()
    assert cfg.batch_size % total_gpus == 0, f'Batch size should be matched with GPUS: ({cfg.batch_size}, {total_gpus})'
    cfg.batch_size = cfg.batch_size // total_gpus
    # 初始化分布式通信组，调整batch_size保证每个GPU分到的数据量一致。
    cfg.local_rank = dist.get_rank()


def init():
    # backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    if cfg.local_rank == 0:
        os.makedirs(backup_dir, exist_ok=True)
        os.system(f'cp train.py {backup_dir}')
        os.system(f'cp {cfg.model_dir} {backup_dir}')
        os.system(f'cp {cfg.dataset_dir} {backup_dir}')
        os.system(f'cp {cfg.config} {backup_dir}')
        os.system(f'cp /home/zxb/GuidedContrastrandnew/model/unet/randlanetnew.py {backup_dir}')

    if cfg.local_rank == 0:
        # logger
        global logger
        from util.log import get_logger
        logger = get_logger(cfg)
        logger.info(cfg)

        # summary writer
        global writer
        writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)
# 全局的初始化与备份，只有rank为0的主进程会把主要的代码和配置文件复制到一个backup文件夹，便于以后复现实验和排查实验问题。
# 最后依旧是设置随机数种子，保证实验的可复现性。

def get_batch_data(dataloader, sampler, data_iterator=None, epoch=0, it_in_epoch=0, dist=False):
    if data_iterator is None or it_in_epoch == 0:
        if dist:
            sampler.set_epoch(epoch)
        data_iterator = iter(dataloader)

    batch = next(data_iterator)

    it_in_epoch = (it_in_epoch + 1) % (dataloader.__len__())
    epoch = epoch + int(it_in_epoch == 0)

    return batch, data_iterator, epoch, it_in_epoch


# 请将此函数完整复制并替换您 train.py 中的 evaluate 函数

# def evaluate(cfg, model, model_fn, dataloader, it):
#     model.eval()
#
#     if cfg.local_rank == 0:
#         logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
#     am_dict = {}
#
#     with torch.no_grad():
#         start_epoch = time.time()
#         for i, batch in enumerate(dataloader):
#             # 直接进行前向推理，不在评估时下采样
#             loss, preds, visual_dict, meter_dict = model_fn(batch, model, it)
#
#             # 多卡GPU数据合并
#             if cfg.dist:
#                 for k, v in visual_dict.items():  # losses
#                     count = meter_dict[k][1]
#
#                     v = v * count
#                     count = loss.new_tensor([count], dtype=torch.long)
#                     dist.all_reduce(v), dist.all_reduce(count)
#                     count = count.item()
#                     v = v / count
#
#                     visual_dict[k] = v
#                     meter_dict[k] = (float(v), count)
#
#             # 更新各项指标的平均值
#             for k, v in meter_dict.items():
#                 if k not in am_dict.keys():
#                     am_dict[k] = utils.AverageMeter()
#                 if cfg.dist and k in ['intersection', 'union', 'target']:
#                     cnt_list = torch.from_numpy(v[0]).cuda()
#                     dist.all_reduce(cnt_list)
#                     am_dict[k].update(cnt_list.cpu().numpy(), v[1])
#                 else:
#                     am_dict[k].update(v[0], v[1])
#
#             # 打印进度信息
#             if cfg.local_rank == 0:
#                 print(f"{i + 1}/{dataloader.__len__()} loss: {am_dict['loss'].val:.4f}({am_dict['loss'].avg:.4f})")
#
#         # 记录到 TensorBoard / logger
#         if cfg.local_rank == 0:
#             print_info = f"iter: {it}/{cfg.iters}, val loss: {am_dict['loss'].avg:.4f}, " \
#                          f"time: {time.time() - start_epoch:.4f}s"
#
#             for k in am_dict.keys():
#                 if k in visual_dict.keys():
#                     writer.add_scalar(k + '_eval', am_dict[k].avg, it)
#
#             if 'intersection' in am_dict:
#                 miou = (am_dict['intersection'].sum / (am_dict['union'].sum + 1e-10)).mean()
#                 macc = (am_dict['intersection'].sum / (am_dict['target'].sum + 1e-10)).mean()
#                 allacc = (am_dict['intersection'].sum).sum() / ((am_dict['target'].sum).sum() + 1e-10)
#                 writer.add_scalar('miou_eval', miou, it)
#                 writer.add_scalar('macc_eval', macc, it)
#                 writer.add_scalar('allacc_eval', allacc, it)
#                 print_info += f', miou: {miou:.4f}, macc: {macc:.4f}, allacc: {allacc:.4f}'
#
#             logger.info(print_info)

# ======================================================================
# 请在 train.py 文件中，只替换下面的 evaluate 函数 (最终修正版)
# ======================================================================
# ======================================================================
# 请在 train.py 文件中，只替换下面的 evaluate 函数 (最终修正版)
# ======================================================================
#
# def evaluate(cfg, model, model_fn, dataloader, it):
#     model.eval()
#
#     if cfg.local_rank == 0:
#         logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
#
#     # AverageMeter 用于累加和计算平均指标
#     am_dict = {}
#
#     with torch.no_grad():
#         start_epoch = time.time()
#         for i, batch in enumerate(dataloader):
#             # model_fn 已经封装了设备转移、前向传播、损失和指标计算
#             # 我们只需要像训练时一样调用它即可。
#             # 注意：不再需要滑窗，因为dataloader的每个batch都已经是适合模型的大小。
#             loss, preds, visual_dict, meter_dict = model_fn(batch, model, it)
#
#             # 累加每个batch的评估结果
#             for k, v in meter_dict.items():
#                 if k not in am_dict:
#                     am_dict[k] = utils.AverageMeter()
#                 am_dict[k].update(v[0], v[1])
#
#         # 在所有验证batch结束后，计算并打印最终的平均指标
#         if cfg.local_rank == 0:
#             writer = SummaryWriter(cfg.exp_path)
#             print_info = f"iter: {it}/{cfg.iters}, val loss: {am_dict['loss'].avg:.4f}, " \
#                          f"time: {time.time() - start_epoch:.4f}s"
#
#             writer.add_scalar('loss_eval', am_dict['loss'].avg, it)
#             if 'intersection' in am_dict:
#                 iou_per_class = am_dict['intersection'].sum / (am_dict['union'].sum + 1e-10)
#                 miou = iou_per_class.mean()
#
#                 # 打印每个类的IoU，方便详细分析
#                 for i, iou in enumerate(iou_per_class):
#                     logger.info(f"class_{i} IoU: {iou:.4f}")
#
#                 writer.add_scalar('miou_eval', miou, it)
#                 print_info += f', miou: {miou:.4f}'
#
#             logger.info(print_info)
#
#     # 切换回训练模式
#     model.train()
# ======================================================================
# 最终解决方案：实现“多样本随机评估”的 evaluate 函数
# 该方案旨在平衡评估的“速度”与“可靠性”
# ======================================================================
# ======================================================================
# 最终解决方案：实现“多样本随机评估”的 evaluate 函数
# 此版本在解决评估不稳定问题的同时，完整保留了您原有的输出格式
# ======================================================================
# ======================================================================
# 最终解决方案：实现“多样本随机评估”的 evaluate 函数
# 此版本严格遵从您指定的输出格式：静默执行，最后输出类别IoU和总结报告
# ======================================================================
import time
import numpy as np
import torch
import util.utils as utils


def evaluate(cfg, model, model_fn, dataloader, it):
    model.eval()

    if cfg.local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    # ########################## 可调参数 ########################## #
    # 设置每个场景的随机采样次数。数值越高，结果越可靠，但评估时间越长。
    num_eval_samples = 3
    # ############################################################# #

    # AverageMeter 用于累加和计算所有样本和所有场景的总指标
    am_dict = {}

    with torch.no_grad():
        start_epoch = time.time()
        # 遍历数据加载器中的每一个 batch
        for i, batch in enumerate(dataloader):

            # --- 多样本评估核心逻辑 ---
            batch_intersections, batch_unions = [], []
            batch_loss_sum = 0.0

            for k in range(num_eval_samples):
                loss, preds, visual_dict, meter_dict = model_fn(batch, model, it)

                if 'intersection' in meter_dict:
                    batch_intersections.append(meter_dict['intersection'][0])
                    batch_unions.append(meter_dict['union'][0])
                batch_loss_sum += loss.item()

            # --- 聚合多样本结果并更新 AverageMeter ---
            if 'loss' not in am_dict: am_dict['loss'] = utils.AverageMeter()
            am_dict['loss'].update(batch_loss_sum / num_eval_samples, batch['labels'].size(0))

            if batch_intersections:
                if 'intersection' not in am_dict: am_dict['intersection'] = utils.AverageMeter()
                if 'union' not in am_dict: am_dict['union'] = utils.AverageMeter()

                am_dict['intersection'].update(np.sum(batch_intersections, axis=0), 1)
                am_dict['union'].update(np.sum(batch_unions, axis=0), 1)

            # 【已移除】：此处没有任何 per-batch 的 print 语句，评估过程保持静默。

        # ====================================================================
        # 在所有验证batch结束后，计算并打印您指定的最终报告
        # ====================================================================
        if cfg.local_rank == 0:
            writer = SummaryWriter(cfg.exp_path)

            # 准备最终的总结信息字符串
            print_info = f"iter: {it}/{cfg.iters}, val loss: {am_dict['loss'].avg:.4f}, " \
                         f"time: {time.time() - start_epoch:.4f}s"

            # 写入 loss 到 Tensorboard
            writer.add_scalar('loss_eval', am_dict['loss'].avg, it)

            # 仅在有交并集结果时，才计算和打印IoU相关信息
            if 'intersection' in am_dict:
                iou_per_class = am_dict['intersection'].sum / (am_dict['union'].sum + 1e-10)
                miou = iou_per_class.mean()

                # 【遵从要求】：逐行打印每个类的IoU
                logger.info("--- Per-Class IoU ---")
                for class_idx, iou in enumerate(iou_per_class):
                    logger.info(f"class_{class_idx} IoU: {iou:.4f}")
                logger.info("---------------------")

                # 写入 miou 到 Tensorboard
                writer.add_scalar('miou_eval', miou, it)

                # 将 miou 追加到最终的总结信息中
                print_info += f', miou: {miou:.4f}'

            # 【遵从要求】：打印最终的总结信息行
            logger.info(print_info)

    # 切换回训练模式
    model.train()

def train_iter(cfg, model, model_fn, optimizer, scheduler, dataset, data_iterator_l, data_iterator_u,
               it, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u,  # it from 1, epoch from 0, it_in_epoch from 0
               am_dict, train_with_unlabeled=False):
    end = time.time()

    # data
    batch_l, data_iterator_l, epoch_l, it_in_epoch_l = get_batch_data(
        dataset.l_train_data_loader, dataset.l_train_sampler, data_iterator_l, epoch_l, it_in_epoch_l, cfg.dist)
    batch = batch_l
    if train_with_unlabeled and it > cfg.prepare_iter:
        batch_u, data_iterator_u, epoch_u, it_in_epoch_u = get_batch_data(
            dataset.u_train_data_loader, dataset.u_train_sampler, data_iterator_u, epoch_u, it_in_epoch_u, cfg.dist)
        batch = (batch_l, batch_u)
    am_dict['data_time'].update(time.time() - end)
    # 从有标签数据加载器中获取一个批次的数据，并更新迭代器和epoch。batch_l是有标签数据，batch_u是无标签数据。
    #if后是获取无标签数据的部分，train_with_unlabeled是一个布尔值，表示是否使用无标签数据进行训练。，batch变为tuple，包含有标签和无标签数据。
    #

    # forward,
    # 重点部分向前推理，visal_dict是可视化字典，用于tensorboard可视化；meter_dict是度量字典，记录loss,miou等
    loss, _, visual_dict, meter_dict = model_fn(batch, model, it)#函数内虽然没有定义模型相关，但是将数据传入到模型内也是在这里进行的，model_fn是一个函数，负责处理模型的前向推理、损失计算和可视化等任务。封装了向前与损失计算函数等。

    # backward
    # 梯度清零，反向传播，更新模型参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # adjust learning rate
    # 获取当前学习率，执行学习率调度器的步进操作
    lrs = scheduler.get_last_lr()
    scheduler.step()

    # meter dict
    # 遍历度量值并同步更新
    for k, v in meter_dict.items():
        if k not in am_dict.keys():
            am_dict[k] = utils.AverageMeter()
        am_dict[k].update(v[0], v[1])

    # summary writer
    # 打印日志的略
    if cfg.local_rank == 0:
        writer.add_scalar('lr_train', lrs[0], it)
        if epoch_l > 0 and it_in_epoch_l == 0:
            print_info = f'iter: {it}/{cfg.iters}'
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_train', am_dict[k].avg, it)
                    print_info += f', {k}: {am_dict[k].avg:.4f}'
            if 'intersection' in am_dict:
                miou = (am_dict['intersection'].sum / (am_dict['union'].sum + 1e-10)).mean()
                macc = (am_dict['intersection'].sum / (am_dict['target'].sum + 1e-10)).mean()
                allacc = (am_dict['intersection'].sum).sum() / ((am_dict['target'].sum).sum() + 1e-10)
                writer.add_scalar('miou_train', miou, it)
                writer.add_scalar('macc_train', macc, it)
                writer.add_scalar('allacc_train', allacc, it)
                print_info += f', miou: {miou:.4f}, macc: {macc:.4f}, allacc: {allacc:.4f}'
            logger.info(print_info)

    # save checkpoint，保存模型
    if cfg.local_rank == 0:
        f = utils.checkpoint_save(model, optimizer, cfg.exp_path, cfg.config.split('/')[-1][:-5], it, cfg.iters,
                                  save_freq=cfg.save_freq, keep_freq=cfg.keep_freq, keep_last_ratio=cfg.keep_last_ratio)
        if f is not None:
            logger.info(f'iter: {it}/{cfg.iters}, Saving {f}')

    # infos
    am_dict['iter_time'].update(time.time() - end)

    remain_iter = cfg.iters - it
    remain_time = time.strftime('%d:%H:%M:%S', time.gmtime(remain_iter * am_dict['iter_time'].avg))
    remain_time = f'{int(remain_time[:2]) - 1:02d}{remain_time[2:]}'

    if cfg.local_rank == 0:
        logger.info(
            f"iter: {it}/{cfg.iters}, lr: {lrs[0]:.4e} "
            f"loss: {am_dict['loss'].val:.4f}({am_dict['loss'].avg:.4f}) "
            f"data_time: {am_dict['data_time'].val:.2f}({am_dict['data_time'].avg:.2f}) "
            f"iter_time: {am_dict['iter_time'].val:.2f}({am_dict['iter_time'].avg:.2f}) "
            f"remain_time: {remain_time}")

    # reset meter_dict，每个新的epoch都要重置度量字典，避免跨epoch累加。
    if epoch_l > 0 and it_in_epoch_l == 0:
        for k in am_dict.keys():
            if k in visual_dict.keys():
                am_dict[k].reset()
        if 'intersection' in am_dict:
            am_dict['intersection'].reset(), am_dict['union'].reset(), am_dict['target'].reset()

    return data_iterator_l, data_iterator_u, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u

# 上为单步训练流程，


#model_fn是一个函数，负责处理模型的前向推理、损失计算和可视化等任务。封装了向前与损失计算函数
# 最后一个参数为从第几步开始支持断点继续。
def train(cfg, model, model_fn, optimizer, scheduler, dataset, start_iter=0):
    model.train()
    data_iterator_l, data_iterator_u = None, None
    epoch_l, it_in_epoch_l = divmod(start_iter, dataset.l_train_data_loader.__len__())#第几个epoch编号以及第几个内的步数,epoch内的第几个batch,断点恢复
    epoch_u, it_in_epoch_u = divmod(max(start_iter - cfg.prepare_iter, 0), dataset.u_train_data_loader.__len__())
    # 初始化指数统计表，方便估算时间以及性能分析######################################
    am_dict = {}
    am_dict['iter_time'] = utils.AverageMeter()
    am_dict['data_time'] = utils.AverageMeter()
    # 主训练循环，每次循环就是一个"iteration"（一次前向+反向+优化）。
    for it in range(start_iter, cfg.iters):  # start from 0
        data_iterator_l, data_iterator_u, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u = train_iter(
            cfg, model, model_fn, optimizer, scheduler, dataset, data_iterator_l, data_iterator_u,
            it + 1, epoch_l, epoch_u, it_in_epoch_l, it_in_epoch_u, am_dict, train_with_unlabeled=cfg.semi)
        #调用单步训练函数，每次迭代都调用进行训练，train_with_unlabeled=cfg.semi 表示是否使用半监督（有无标签数据一起训练）
        if cfg.validation and (
                utils.is_multiple(it + 1, cfg.eval_freq) or
                (utils.is_last(it + 1, cfg.iters, cfg.eval_last_ratio) and utils.is_multiple(it + 1, cfg.save_freq))):
            evaluate(cfg, model, model_fn, dataset.val_data_loader, it + 1)
            model.train()


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
    # 分布式训练初始化
    # get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    cfg.model_name = exp_name.split('_')[0]
    cfg.data_name = exp_name.split('_')[-1]

    # model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')

    model, model_fn = build_network(cfg)#####################################################

    use_cuda = torch.cuda.is_available()
    if cfg.local_rank == 0:
        logger.info(f'cuda available: {use_cuda}')
    assert use_cuda
    model = model.cuda()
    # 检查是否存在GPU，并将模型转移到GPU上。
    if cfg.dist:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.local_rank % num_gpus
        if cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    if cfg.local_rank == 0:
        logger.info(f'#model parameters: {sum([x.nelement() for x in model.parameters()])}')


    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(params, lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError

    # dataset
    dataset = build_dataset(cfg)

    if cfg.dataset == 's3dis':
        if cfg.local_rank == 0:
            logger.info(f'Training area: {cfg.train_area}')
            logger.info(f'Validation area: {cfg.test_area}')
    dataset.trainLoader()
    dataset.valLoader()
    if cfg.local_rank == 0:
        logger.info(f'Training samples: {dataset.train_file_names.__len__()}')
        logger.info(f'Validation samples: {dataset.val_file_names.__len__()}')

    # resume
    start_iter, f = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], dist=cfg.dist,
                                             gpu=cfg.local_rank % torch.cuda.device_count(), optimizer=optimizer)
    if cfg.local_rank == 0:
        logger.info(f'Restore from {f}' if len(f) > 0 else f'Start from iteration {start_iter}')

    # lr_scheduler
    # 根据配置和起始迭代初始化学习率调度器
    scheduler = initialize_scheduler(optimizer, cfg, last_step=start_iter - 1)
    # 调用前面分析过的train()主循环，正式开始训练
    train(cfg, model, model_fn, optimizer, scheduler, dataset, start_iter=start_iter)  # start_iter from 0

