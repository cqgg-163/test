

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import functools
import sys, os
sys.path.append('../../')

from util.spconv_utils import spconv
from lib.ops import voxelization
from util import utils
# from model.unet.unet import UNet, ResidualBlock, VGGBlock
# from model.unet.pointnet2_unet import PointNet2UNet
from model.unet.randlanetnew import RandLANetUNet
from model.unet.helper_tool import DataProcessing as  DP


from util.pointdata_process import get_pos_sample_idx, get_neg_sample, split_embed
#
#
# class SemSeg(nn.Module):
#     def __init__(self, cfg):#模型初始化与结构定义
#         super().__init__()
#
#         input_c = cfg.input_channel + 3#输入通道数
#         classes = cfg.classes#
#
#         m = cfg.m                  #基础通道数，
#         embed_m = cfg.embed_m       #嵌入向量维度
#
#         self.pretrain_path = cfg.pretrain_path#预训练权重
#         self.pretrain_module = cfg.pretrain_module#预训练模块名列表
#
#         # backbone 主干网络
#         # nPlanes = [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m]#每层通道数，
#         # block_reps = cfg.block_reps                            #每层重复块数
#         # block = ResidualBlock if cfg.block_residual else VGGBlock
#         # norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
#         # self.backbone = UNet(input_c, m, nPlanes, block_reps, block, norm_fn, cfg)
#         nPlanes = [m, 2 * m, 3 * m, 4 * m]
#         # self.backbone = PointNet2UNet(input_c, m, nPlanes, None, None, None, cfg)
#         self.backbone = RandLANetUNet(input_c, m, nPlanes, cfg)
#
#         # projector 特征投影层
#         self.projector = nn.Linear(m, embed_m)
#
#         # classifier 分类头
#         self.classifier = nn.Linear(m, classes)
#
#         # memory bank  两组记忆队列，（用于对比学习
#         self.register_buffer('embed_queue1', torch.randn(classes, cfg.bank_length, embed_m))
#         self.embed_queue1 = F.normalize(self.embed_queue1, p=2, dim=-1)
#         self.register_buffer('index_queue1', torch.zeros((classes, cfg.bank_length), dtype=torch.long))
#         self.register_buffer('queue_pointer1', torch.zeros(classes, dtype=torch.long))
#         self.register_buffer('valid_pointer1', torch.zeros(classes, dtype=torch.long))
#
#         self.register_buffer('embed_queue2', torch.randn(classes, cfg.bank_length, embed_m))
#         self.embed_queue2 = F.normalize(self.embed_queue2, p=2, dim=-1)
#         self.register_buffer('index_queue2', torch.zeros((classes, cfg.bank_length), dtype=torch.long))
#         self.register_buffer('queue_pointer2', torch.zeros(classes, dtype=torch.long))
#         self.register_buffer('valid_pointer2', torch.zeros(classes, dtype=torch.long))
#         # BN初始化
#         self.apply(self.set_bn_init)
#
#         # load pretrain weights  预训练权重加载
#         module_map = {'backbone': self.backbone, 'classifier': self.classifier, 'projector': self.projector}
#         if self.pretrain_path is not None:
#             map_location = {'cuda:0': f'cuda:{cfg.local_rank % torch.cuda.device_count()}'} if cfg.local_rank > 0 else None
#             state = torch.load(self.pretrain_path, map_location=map_location)
#             pretrain_dict = state if not 'state_dict' in state else state['state_dict']
#             if 'module.' in list(pretrain_dict.keys())[0]:
#                 pretrain_dict = {k[len('module.'):]: v for k, v in pretrain_dict.items()}
#             for m in self.pretrain_module:
#                 n1, n2 = utils.load_model_param(module_map[m], pretrain_dict, prefix=m)
#                 if cfg.local_rank == 0:
#                     print(f'[PID {os.getpid()}] Load pretrained {m}: {n1}/{n2}')
#
#     @staticmethod
#     def set_bn_init(m):#BN初始化的辅助
#         classname = m.__class__.__name__
#         if classname.find('BatchNorm') != -1:
#             m.weight.data.fill_(1.0)
#             m.bias.data.fill_(0.0)
#     # 数据编码流程，将点云批量数据体素化、聚合，再通过主干网络编码，输出点特征。
#     # def encoder(self, batch):
#     #     voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
#     #     p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
#     #     v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda
#     #
#     #     coords_float = batch['locs_float'].cuda()  # (N, 3), float, cuda
#     #     feats = batch['feats'].cuda()              # (N, C), float, cuda
#     #     feats = torch.cat((feats, coords_float), 1)
#     #     voxel_feats = voxelization(feats, v2p_map, 4)  # (M, C), float, cuda
#     #
#     #     spatial_shape = batch['spatial_shape']
#     #
#     #     batch_size = len(batch['offsets']) - 1
#     #
#     #     inp = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
#     #     out = self.backbone(inp)
#     #     out = out.features[p2v_map.long()]
#     #
#     #     return out
#
#     def encoder(self, batch):
#         coords_float = batch['locs_float'].cuda()  # (N_total, 3)
#         feats = batch['feats'].cuda()  # (N_total, C_in)
#         offsets = batch['offsets'].cuda()  # (B+1,)
#
#         # Combine original features with coordinate features
#         feats = torch.cat((feats, coords_float), 1)
#
#         B = len(offsets) - 1
#
#         # Prepare lists for batching
#         xyz_list = []
#         feats_list = []
#
#         for b in range(B):
#             start = offsets[b]
#             end = offsets[b + 1]
#
#             # --- KEY MODIFICATION: Per-scan normalization ---
#             # This is crucial for training stability with RandLA-Net
#             xyz_b = coords_float[start:end]
#
#             # Center the point cloud
#             xyz_center = xyz_b.mean(0)
#             xyz_b -= xyz_center
#
#             # Normalize to a unit sphere
#             # This scale factor is important for consistent distance metrics inside the network
#             scale = (1 / torch.sqrt((xyz_b ** 2).sum(dim=1)).max()) * 0.99999
#             xyz_b *= scale
#
#             xyz_list.append(xyz_b)
#             feats_list.append(feats[start:end])
#
#         # Pad to max_num to create a dense batch for RandLA-Net
#         max_num = max([x.shape[0] for x in xyz_list])
#
#         xyz_pad = torch.zeros(B, max_num, 3, device=coords_float.device)
#         feats_pad = torch.zeros(B, feats.shape[1], max_num, device=feats.device)
#         mask = torch.zeros(B, max_num, dtype=torch.bool, device=xyz_pad.device)
#
#         for b in range(B):
#             n = xyz_list[b].shape[0]
#             xyz_pad[b, :n] = xyz_list[b]
#             feats_pad[b, :, :n] = feats_list[b].transpose(0, 1)
#             mask[b, :n] = 1
#
#         # Prepare batch dictionary for the new network
#         net_batch = {'xyz': xyz_pad, 'features': feats_pad, 'mask': mask}
#
#         # Pass through the new backbone
#         out_feats_padded = self.backbone(net_batch)  # (B, m, max_num)
#
#         # Un-pad the results to get features for original points
#         outs = []
#         for b in range(B):
#             n = xyz_list[b].shape[0]
#             outs.append(out_feats_padded[b, :, :n].transpose(0, 1))
#
#         return torch.cat(outs, 0)  # (N_total, m)
#
#     #支持有标签/无标签（半监督）输入，输出各自的特征、分类分数和对比学习用的嵌入向量。
#     def forward(self, batch_l, batch_u=None):
#         ret= {}
#
#         output_feats_l = self.encoder(batch_l)                # (Nl, C)
#         semantic_scores_l = self.classifier(output_feats_l)   # (Nl, nClass)
#         ret['semantic_scores_l'] = semantic_scores_l
#         ret['semantic_features_l'] = output_feats_l
#
#         if batch_u is not None:
#             output_feats_u = self.encoder(batch_u)              # (Nu, C)
#             semantic_scores_u = self.classifier(output_feats_u) # (Nu, nClass)
#             ret['semantic_scores_u'] = semantic_scores_u
#             ret['semantic_features_u'] = output_feats_u
#
#             output_feats_u = self.projector(output_feats_u)        # (Nu, C), Nu = Nu11 + Nu12 + ... + NuB1 + NuB2
#             embed_u = F.normalize(output_feats_u, p=2, dim=1)
#             ret['embed_u'] = embed_u
#
#         return ret


class SemSeg(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 输入通道数 = 基础RGB通道数(3) + 坐标特征(3) = 6
        # 这里的 cfg.input_channel 应该在您的 yaml 文件中被设置为 3
        input_c = cfg.input_channel + 3
        classes = cfg.classes

        m = cfg.m
        embed_m = cfg.embed_m

        self.pretrain_path = cfg.pretrain_path
        self.pretrain_module = cfg.pretrain_module

        # nPlanes = [m, 2 * m, 3 * m, 4 * m]
        # 使用您提供的 RandLANetUNet
        # self.backbone = RandLANetUNet(input_c, m, nPlanes, cfg)
        nPlanes = [m, 2*m, 3*m, 4*m,5*m]  # 保持原有配置
        self.backbone = RandLANetUNet(input_c, m, nPlanes, cfg)

        # 增强分类头
        self.projector = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, embed_m)
        )
        self.classifier = nn.Sequential(
            nn.Linear(m, m),
            nn.ReLU(),
            nn.Linear(m, classes)
        )

        # self.projector = nn.Linear(m, embed_m)
        # self.classifier = nn.Linear(m, classes)

        # memory bank 和其他初始化保持不变
        self.register_buffer('embed_queue1', torch.randn(classes, cfg.bank_length, embed_m))
        self.embed_queue1 = F.normalize(self.embed_queue1, p=2, dim=-1)
        self.register_buffer('index_queue1', torch.zeros((classes, cfg.bank_length), dtype=torch.long))
        self.register_buffer('queue_pointer1', torch.zeros(classes, dtype=torch.long))
        self.register_buffer('valid_pointer1', torch.zeros(classes, dtype=torch.long))

        self.register_buffer('embed_queue2', torch.randn(classes, cfg.bank_length, embed_m))
        self.embed_queue2 = F.normalize(self.embed_queue2, p=2, dim=-1)
        self.register_buffer('index_queue2', torch.zeros((classes, cfg.bank_length), dtype=torch.long))
        self.register_buffer('queue_pointer2', torch.zeros(classes, dtype=torch.long))
        self.register_buffer('valid_pointer2', torch.zeros(classes, dtype=torch.long))

        # self.apply(self.set_bn_init)

        # 预训练权重加载保持不变
        module_map = {'backbone': self.backbone, 'classifier': self.classifier, 'projector': self.projector}
        if self.pretrain_path is not None:
            map_location = {
                'cuda:0': f'cuda:{cfg.local_rank % torch.cuda.device_count()}'} if cfg.local_rank > 0 else None
            state = torch.load(self.pretrain_path, map_location=map_location)
            pretrain_dict = state if not 'state_dict' in state else state['state_dict']
            if 'module.' in list(pretrain_dict.keys())[0]:
                pretrain_dict = {k[len('module.'):]: v for k, v in pretrain_dict.items()}
            for m_ in self.pretrain_module:
                n1, n2 = utils.load_model_param(module_map[m_], pretrain_dict, prefix=m_)
                if cfg.local_rank == 0:
                    print(f'[PID {os.getpid()}] Load pretrained {m_}: {n1}/{n2}')

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def encoder(self, batch):
        # # 1. 数据已经由 get_batch_data 准备好，直接使用
        # coords_float = batch['locs_float'].cuda()  # (N_total, 3)
        # input_features = batch['feats'].cuda()  # (N_total, 6) <--- 已经是6维了！
        #
        # # 2. 高效地准备送入 backbone 的数据
        # batch_size = len(batch['offsets']) - 1
        # # 从模型获取期望点数，例如 40960
        # # 您需要确保在 RandLANetUNet 的 __init__ 中有 self.num_points = num_points
        # num_points = self.backbone.num_points
        #
        # # 将坐标重塑为 (B, N, 3)
        # xyz = coords_float.view(batch_size, num_points, 3)
        #
        # # 将6维特征重塑为 (B, C, N) 以匹配 Conv1d 的输入
        # features_6d = input_features.view(batch_size, num_points, 6).permute(0, 2, 1)
        #
        # # 创建一个干净的字典送入 backbone
        # net_batch = {'xyz': xyz, 'features': features_6d}
        #
        # # 3. 调用 backbone
        # out_feats_padded = self.backbone(net_batch)  # (B, C_out, 40960)
        #
        # # 4. 将批处理的输出重新变回平铺的张量
        # output_features = out_feats_padded.permute(0, 2, 1).contiguous().view(-1, out_feats_padded.shape[1])
        #
        # return output_features
        # batch 字典由 collate_fn 准备好，直接送入 backbone
        # backbone 输出 (B, m, N)
        output_features_bmn = self.backbone(batch)

        # 将输出调整为 (B*N, m) 以适应下游
        B, M, N = output_features_bmn.shape
        return output_features_bmn.permute(0, 2, 1).contiguous().view(-1, M)

    def forward(self, batch_l, batch_u=None):
        ret = {}
        output_feats_l = self.encoder(batch_l)
        semantic_scores_l = self.classifier(output_feats_l)
        ret['semantic_scores_l'] = semantic_scores_l
        ret['semantic_features_l'] = output_feats_l

        if batch_u is not None:
            output_feats_u = self.encoder(batch_u)
            semantic_scores_u = self.classifier(output_feats_u)
            ret['semantic_scores_u'] = semantic_scores_u
            ret['semantic_features_u'] = output_feats_u
            output_feats_u = self.projector(output_feats_u)
            embed_u = F.normalize(output_feats_u, p=2, dim=1)
            ret['embed_u'] = embed_u

        return ret

def model_fn_decorator(cfg, test=False):
    # criterion = nn.CrossEntropyLoss(
    #     ignore_index=cfg.ignore_label,
    #     weight=(None if not hasattr(cfg, 'loss_weights_classes')
    #             else torch.tensor(cfg.loss_weights_classes).float())
    # ).cuda()
    class_weights = torch.from_numpy(DP.get_class_weights('S3DIS')).squeeze().float().cuda()

    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.ignore_label,
        weight=class_weights
    ).cuda()
    # def model_fn(batch, model, step):
    #     # 1) 设备对齐 + 拆分有/无标签
    #     device = next(model.parameters()).device
    #     if isinstance(batch, tuple):
    #         batch_l, batch_u = batch
    #     else:
    #         batch_l, batch_u = batch, None
    #     for d in (batch_l,) + ((batch_u,) if batch_u is not None else ()):
    #         if d is None: continue
    #         for k, v in d.items():
    #             if v is None:
    #                 continue
    #             # if isinstance(v, list):
    #             #     # 如果值是一个列表, 遍历列表并将每个张量移到GPU
    #             #     d[k] = [item.to(device) for item in v]
    #             # elif torch.is_tensor(v):
    #             #     # 如果值是单个张量, 直接移动
    #             #     d[k] = v.to(device)
    #             if isinstance(v, list):
    #                 # 遍历列表中的每个元素，如果是张量则移动，否则保持不变
    #                 d[k] = [item.to(device) if torch.is_tensor(item) else item for item in v]
    #             elif torch.is_tensor(v):
    #                 # 如果值是单个张量, 直接移动
    #                 d[k] = v.to(device)
    #     # 2) 前向
    #     ret = model(batch_l, batch_u)
    #     semantic_scores_l = ret['semantic_scores_l']  # (Nl, C)
    #
    #     # 3) 构造 loss_inp
    #     loss_inp = {}
    #     if not test:
    #         # -- 有监督分支 --
    #         labels_l = batch_l['labels']  # (Nl,)
    #         loss_inp['sup_loss'] = (semantic_scores_l, labels_l)
    #
    #         # -- 半监督对比分支 --
    #         if batch_u is not None and step > cfg.prepare_iter:
    #             semantic_scores_u = ret['semantic_scores_u']  # (Nu, C)
    #             embed_u = ret['embed_u']  # (Nu, D)
    #             pseudo_labels = ret['pseudo_labels']  # (Nu,)
    #
    #             idxos1, idxos2 = batch_u['idxos']  # each (Nuo,)
    #             point_cnts_o = batch_u['point_cnts_o']  # [Nuo1, ..., NuoB]
    #             offsets_u = batch_u['offsets']  # (2B+1,)
    #
    #             # 3.1) 切分两视图 embedding
    #             embed1, embed2 = split_embed(embed_u, offsets_u)
    #
    #             # 3.2) 正样本采样
    #             pos_idx = get_pos_sample_idx(
    #                 point_cnts_o, cfg.num_pos_sample,
    #                 pseudo_labels, cfg.classes
    #             )
    #             assert pos_idx.numel() > 0, (
    #                 f"[ERROR] step {step}: pos_idx is empty, counts={point_cnts_o.tolist()}"
    #             )
    #
    #             # 3.3) 负样本采样
    #             all_idx = torch.cat([idxos1, idxos2], dim=0)
    #             neg_embed, neg_labels, neg_idx = get_neg_sample(
    #                 embed_u, all_idx, pseudo_labels,
    #                 model.embed_queue, model.index_queue,
    #                 model.queue_pointer, model.valid_pointer,
    #                 cfg
    #             )
    #             if cfg.dist:
    #                 neg_embed = commu_utils.concat_all_gather(neg_embed)
    #                 neg_labels = commu_utils.concat_all_gather(neg_labels)
    #
    #             loss_inp['unsup_loss'] = (
    #                 embed1, embed2, pos_idx, neg_embed, neg_labels
    #             )
    #
    #     # 4) 计算总 loss
    #     loss, loss_out, infos = loss_fn(
    #         loss_inp, step,
    #         model=model.module if hasattr(model, 'module') else model
    #     )
    #
    #     # 5) 统计 & 返回
    #     with torch.no_grad():
    #         preds = {'semantic': semantic_scores_l}
    #         if test:
    #             return preds
    #
    #         p = semantic_scores_l.argmax(dim=1).cpu().numpy()
    #         # gt = batch_l['labels'].cpu().numpy()
    #         gt = batch_l['labels'].view(-1).cpu().numpy()  # 添加 .view(-1)
    #         i, u, t = utils.intersectionAndUnion(
    #             p, gt, cfg.classes, cfg.ignore_label
    #         )
    #         visual = {'loss': loss}
    #         meter = {'loss': (loss.item(), gt.shape[0]),
    #                  'intersection': (i, 1),
    #                  'union': (u, 1),
    #                  'target': (t, 1)}
    #         for k, (v, cnt) in loss_out.items():
    #             visual[k] = v
    #             meter[k] = (v.item(), cnt)
    #         return loss, preds, visual, meter
    def model_fn(batch, model, step):
        # 1) 设备对齐 + 拆分有/无标签
        device = next(model.parameters()).device
        if isinstance(batch, tuple):
            batch_l, batch_u = batch
        else:
            batch_l, batch_u = batch, None

        for d in (batch_l,) + ((batch_u,) if batch_u is not None else ()):
            if d is None: continue
            for k, v in d.items():
                if v is None: continue
                if isinstance(v, list):
                    d[k] = [item.to(device) if torch.is_tensor(item) else item for item in v]
                elif torch.is_tensor(v):
                    d[k] = v.to(device)

        # 2) 前向传播
        # 在测试模式下，只处理 batch_l，不传入 batch_u
        ret = model(batch_l, batch_u if not test else None)
        semantic_scores_l = ret['semantic_scores_l']  # (Nl, C)

        # 3) 测试模式下直接返回预测
        if test:
            with torch.no_grad():  # 确保在测试模式下没有梯度计算
                preds = {'semantic': semantic_scores_l}
                return preds

        # ----------------------------------------------------
        # 以下是训练模式下的代码，测试模式会跳过
        # ----------------------------------------------------

        # 4) 构造 loss_inp (仅在训练模式下需要)
        loss_inp = {}

        # -- 有监督分支 --
        if 'labels' in batch_l and torch.is_tensor(batch_l['labels']):
            labels_l = batch_l['labels']  # (Nl,)
            loss_inp['sup_loss'] = (semantic_scores_l, labels_l)
        else:
            raise RuntimeError("Training mode requires 'labels' in batch_l for 'sup_loss'.")

        # -- 半监督对比分支 --
        # 只有在有无标签数据且过了准备阶段才计算无监督损失
        if batch_u is not None and step > cfg.prepare_iter:
            # 确保模型返回了无标签数据的特征和分数
            if 'semantic_scores_u' in ret and 'embed_u' in ret:
                # 传递原始的无标签数据相关的输出和输入，loss_fn内部再进行采样和计算
                loss_inp['unsup_loss'] = {
                    'embed_u': ret['embed_u'],
                    'semantic_scores_u': ret['semantic_scores_u'],
                    'idxos': batch_u['idxos'],  # (idxos1, idxos2)
                    'point_cnts_o': batch_u['point_cnts_o'],
                    'offsets': batch_u['offsets']
                }
            else:
                if step > cfg.prepare_iter:
                    print(
                        f"[WARN] Step {step}: Skipping unsupervised loss as model did not return 'semantic_scores_u' or 'embed_u' for batch_u.")

        # 5) 计算总 loss
        loss, loss_out, infos = loss_fn(
            loss_inp, step,
            model=model.module if hasattr(model, 'module') else model
        )

        # 6) 统计 & 返回 (训练模式下)
        with torch.no_grad():
            preds = {'semantic': semantic_scores_l}
            p = semantic_scores_l.argmax(dim=1).cpu().numpy()
            gt = batch_l['labels'].view(-1).cpu().numpy()
            i, u, t = utils.intersectionAndUnion(
                p, gt, cfg.classes, cfg.ignore_label
            )
            visual = {'loss': loss}
            meter = {'loss': (loss.item(), gt.shape[0]),
                     'intersection': (i, 1),
                     'union': (u, 1),
                     'target': (t, 1)}
            for k, (v, cnt) in loss_out.items():
                visual[k] = v
                meter[k] = (v.item(), cnt)
            return loss, preds, visual, meter

    def sup_loss_fn(scores, labels, step):
        if cfg.get('use_ce_thresh', False):
            def get_ce_thresh(cur_step):
                if cfg.get('use_descending_thresh', False):
                    start_thresh = 1.1
                    thresh = max(
                        cfg.ce_thresh,
                        start_thresh - (start_thresh - cfg.ce_thresh) * cur_step / cfg.ce_steps
                    )
                    return thresh
                elif cfg.get('use_ascending_thresh', False):
                    start_thresh = 1.0 / cfg.classes
                    thresh = min(
                        cfg.ce_thresh,
                        start_thresh + (cfg.ce_thresh - start_thresh) * cur_step / cfg.ce_steps
                    )
                else:
                    thresh = cfg.ce_thresh
                return thresh

            thresh = get_ce_thresh(step)
            temp_scores, _ = F.softmax(scores.detach(), dim=-1).max(1)
            mask = (temp_scores < thresh)
            if mask.sum() == 0:
                mask[torch.nonzero(labels >= 0)[0]] = True
            scores, labels = scores[mask], labels[mask]
        sup_loss = criterion(scores, labels.view(-1))
        # sup_loss = criterion(scores, labels)
        return sup_loss

    def unsup_loss_fn(embed_pos1, embed_pos2, embed_neg, pseudo_labels_pos1, pseudo_labels_neg, pconfs_pos2,
                      pos_idx1, pos_idx2, neg_idx):
        """loss on embed_pos1"""
        def run(embed, embed_neg, pseudo_labels, pseudo_labels_neg, pos_idx1, pos_idx2, neg_idx):
            neg = (embed @ embed_neg.T) / cfg.temp
            mask = torch.ones((embed.shape[0], embed_neg.shape[0]), dtype=torch.float32, device=embed.device)
            mask *= ((neg_idx.unsqueeze(0) != pos_idx1.unsqueeze(-1)).float() *
                     (neg_idx.unsqueeze(0) != pos_idx2.unsqueeze(-1)).float())
            pseudo_label_guidance = (pseudo_labels.unsqueeze(-1) != pseudo_labels_neg.unsqueeze(0)).float()
            mask *= pseudo_label_guidance
            neg = (torch.exp(neg) * mask).sum(-1)
            return neg

        pos = (embed_pos1 * embed_pos2.detach()).sum(-1, keepdim=True) / cfg.temp
        pos = torch.exp(pos).squeeze(-1)

        N = embed_neg.size(0)
        b = cfg.mem_batch_size
        neg = 0
        for i in range((N - 1) // b + 1):
            cur_embed_neg = embed_neg[i * b: (i + 1) * b]
            cur_pseudo_labels_neg = pseudo_labels_neg[i * b: (i + 1) * b]
            cur_neg_idx = neg_idx[i * b: (i + 1) * b]

            cur_neg = checkpoint.checkpoint(run, embed_pos1, cur_embed_neg, pseudo_labels_pos1, cur_pseudo_labels_neg,
                                            pos_idx2, pos_idx1, cur_neg_idx)
            neg += cur_neg

        eps = 1e-10
        unsup_loss = -torch.log(torch.clip(pos / torch.clip(pos + neg, eps), eps))

        confidence_guidance = (pconfs_pos2 >= cfg.conf_thresh).float()
        unsup_loss = (unsup_loss * confidence_guidance).sum() / torch.clip(confidence_guidance.sum(), eps)

        return unsup_loss
    # def loss_fn(loss_inp, step, model=None):
    #     """
    #     Compute supervised and unsupervised contrastive losses with robust bounds checking.
    #     Ensures total_loss is always defined before return.
    #     """
    #     loss_out = {}
    #     infos = {}
    #
    #     # --- 1) Supervised loss ---
    #     semantic_scores_l, semantic_labels_l = loss_inp['sup_loss']
    #     sup_loss = sup_loss_fn(semantic_scores_l, semantic_labels_l, step)
    #     loss_out['sup_loss'] = (sup_loss, semantic_scores_l.size(0))
    #
    #     # Initialize total_loss with supervised component
    #     total_loss = cfg.loss_weight[0] * sup_loss
    #
    #     # --- 2) Unsupervised (contrastive) loss ---
    #     if 'unsup_loss' in loss_inp:
    #         embed_u, semantic_scores_u, idxos1, idxos2, cnts_o, offsets_u = loss_inp['unsup_loss']
    #         device = embed_u.device
    #         N_total = embed_u.size(0)
    #
    #         # 2.1) Pseudo-labels & confidences
    #         confs, pseudo_labels = F.softmax(semantic_scores_u, dim=1).max(dim=1)
    #
    #         # 2.2) Filter invalid idx pairs
    #         L = pseudo_labels.size(0)
    #         valid_mask = (idxos1 >= 0) & (idxos1 < L) & (idxos2 >= 0) & (idxos2 < L)
    #         if valid_mask.numel() != int(valid_mask.sum()):
    #             bad = valid_mask.numel() - int(valid_mask.sum())
    #             print(f"[WARN] Dropping {bad} invalid idxos entries (pool size={L})")
    #         idxos1 = idxos1[valid_mask]
    #         idxos2 = idxos2[valid_mask]
    #
    #         # If no valid pairs remain, skip unsup_loss
    #         if idxos1.numel() == 0:
    #             infos['unsup_skipped'] = True
    #         else:
    #             # 2.3) Positive sampling on CPU
    #             cnts_list = cnts_o.detach().cpu().tolist() if isinstance(cnts_o, torch.Tensor) else list(cnts_o)
    #             pseudo1 = pseudo_labels[idxos1].detach().cpu().tolist()
    #             pos_idx = get_pos_sample_idx(cnts_list, cfg.num_pos_sample, pseudo1, num_classes=cfg.classes)
    #             pos_idx = torch.as_tensor(pos_idx, dtype=torch.long, device=device)
    #
    #             # 2.4) Filter sampled positions
    #             M = idxos1.numel()
    #             valid_pos = (pos_idx >= 0) & (pos_idx < M)
    #             if valid_pos.numel() != int(valid_pos.sum()):
    #                 bad = valid_pos.numel() - int(valid_pos.sum())
    #                 print(f"[WARN] Dropping {bad} invalid pos_idx entries (block size={M})")
    #             pos_idx = pos_idx[valid_pos]
    #
    #             # 2.5) Map to global and clamp
    #             pos_global1 = idxos1[pos_idx].clamp(0, N_total - 1)
    #             pos_global2 = idxos2[pos_idx].clamp(0, N_total - 1)
    #
    #             # 2.6) Gather positives
    #             embed_pos1 = embed_u[pos_global1]
    #             embed_pos2 = embed_u[pos_global2]
    #             plabels_pos1 = pseudo_labels[pos_global1]
    #             plabels_pos2 = pseudo_labels[pos_global2]
    #             confs_pos1 = confs[pos_global1]
    #             confs_pos2 = confs[pos_global2]
    #
    #             # 2.7) Negative sampling (unchanged)
    #             embed_u1, embed_u2 = split_embed(embed_u, offsets_u)
    #             plabels1, plabels2 = split_embed(pseudo_labels, offsets_u)
    #             idx_u1, idx_u2 = split_embed(torch.arange(N_total, device=device), offsets_u)
    #             model.index_queue1[:] = -1
    #             model.index_queue2[:] = -1
    #             embed_neg1, plabels_neg1, neg_idx1 = get_neg_sample(
    #                 embed_u1, idx_u1, plabels1,
    #                 model.embed_queue1, model.index_queue1,
    #                 model.queue_pointer1, model.valid_pointer1, cfg
    #             )
    #             embed_neg2, plabels_neg2, neg_idx2 = get_neg_sample(
    #                 embed_u2, idx_u2, plabels2,
    #                 model.embed_queue2, model.index_queue2,
    #                 model.queue_pointer2, model.valid_pointer2, cfg
    #             )
    #
    #             # 2.8) Compute unsupervised losses
    #             unsup_loss1 = unsup_loss_fn(
    #                 embed_pos1, embed_pos2, embed_neg2,
    #                 plabels_pos1, plabels_neg2,
    #                 confs_pos2, pos_global1, pos_global2, neg_idx2
    #             )
    #             unsup_loss2 = unsup_loss_fn(
    #                 embed_pos2, embed_pos1, embed_neg1,
    #                 plabels_pos2, plabels_neg1,
    #                 confs_pos1, pos_global2, pos_global1, neg_idx1
    #             )
    #             loss_out['unsup_loss'] = (unsup_loss1 + unsup_loss2, embed_pos1.size(0))
    #
    #             # Add weighted unsupervised component
    #             total_loss = total_loss + cfg.loss_weight[1] * (unsup_loss1 + unsup_loss2)
    #
    #     # --- 3) Return ---
    #     return total_loss, loss_out, infos
    def loss_fn(loss_inp, step, model=None):
        """
        Compute supervised and unsupervised contrastive losses with robust bounds checking.
        Ensures total_loss is always defined before return.
        """
        loss_out = {}
        infos = {}

        # --- 1) Supervised loss ---
        semantic_scores_l, semantic_labels_l = loss_inp['sup_loss']
        sup_loss = sup_loss_fn(semantic_scores_l, semantic_labels_l, step)
        loss_out['sup_loss'] = (sup_loss, semantic_scores_l.size(0))

        # Initialize total_loss with supervised component
        total_loss = cfg.loss_weight[0] * sup_loss

        # --- 2) Unsupervised (contrastive) loss ---
        if 'unsup_loss' in loss_inp:
            # 从字典中解包参数
            unsup_data = loss_inp['unsup_loss']
            embed_u = unsup_data['embed_u']
            semantic_scores_u = unsup_data['semantic_scores_u']
            idxos1, idxos2 = unsup_data['idxos']
            point_cnts_o = unsup_data['point_cnts_o']
            offsets_u = unsup_data['offsets']

            device = embed_u.device
            N_total = embed_u.size(0)

            # 2.1) Pseudo-labels & confidences
            confs, pseudo_labels = F.softmax(semantic_scores_u, dim=1).max(dim=1)

            # 2.2) Filter invalid idx pairs
            L = pseudo_labels.size(0)
            valid_mask = (idxos1 >= 0) & (idxos1 < L) & (idxos2 >= 0) & (idxos2 < L)
            if valid_mask.numel() != int(valid_mask.sum()):
                bad = valid_mask.numel() - int(valid_mask.sum())
                print(f"[WARN] Dropping {bad} invalid idxos entries (pool size={L})")
            idxos1 = idxos1[valid_mask]
            idxos2 = idxos2[valid_mask]

            # If no valid pairs remain, skip unsup_loss
            if idxos1.numel() == 0:
                infos['unsup_skipped'] = True
            else:
                # 2.3) Positive sampling on CPU
                cnts_list = point_cnts_o.detach().cpu().tolist() if isinstance(point_cnts_o, torch.Tensor) else list(
                    point_cnts_o)
                # pseudo1 应该是在 idxos1 索引下的伪标签
                pseudo1 = pseudo_labels[idxos1].detach().cpu().tolist()
                pos_idx = get_pos_sample_idx(cnts_list, cfg.num_pos_sample, pseudo1, num_classes=cfg.classes)
                pos_idx = torch.as_tensor(pos_idx, dtype=torch.long, device=device)

                # 2.4) Filter sampled positions
                M = idxos1.numel()
                valid_pos = (pos_idx >= 0) & (pos_idx < M)
                if valid_pos.numel() != int(valid_pos.sum()):
                    bad = valid_pos.numel() - int(valid_pos.sum())
                    print(f"[WARN] Dropping {bad} invalid pos_idx entries (block size={M})")
                pos_idx = pos_idx[valid_pos]

                # If no valid positive pairs remain after sampling, skip unsup_loss
                if pos_idx.numel() == 0:
                    infos['unsup_skipped'] = True
                else:
                    # 2.5) Map to global and clamp
                    pos_global1 = idxos1[pos_idx].clamp(0, N_total - 1)
                    pos_global2 = idxos2[pos_idx].clamp(0, N_total - 1)

                    # 2.6) Gather positives
                    embed_pos1 = embed_u[pos_global1]
                    embed_pos2 = embed_u[pos_global2]
                    plabels_pos1 = pseudo_labels[pos_global1]
                    plabels_pos2 = pseudo_labels[pos_global2]
                    confs_pos1 = confs[pos_global1]
                    confs_pos2 = confs[pos_global2]

                    # 2.7) Negative sampling
                    # 根据 offsets_u 拆分 embed_u 为两个视图的原始 embed
                    # 这与 get_neg_sample 的预期一致，它处理的是单个场景或视图的特征
                    embed_u1, embed_u2 = split_embed(embed_u, offsets_u)
                    plabels1, plabels2 = split_embed(pseudo_labels, offsets_u)
                    # idx_u1, idx_u2 是原始点在当前视图中的局部索引，用于 get_neg_sample 内部判断
                    idx_u1, idx_u2 = split_embed(torch.arange(N_total, device=device), offsets_u)

                    # 每次调用 get_neg_sample 时，重置相应队列的 index_queue
                    # 这是关键，因为 get_neg_sample 内部会更新 index_queue 和 queue_pointer
                    model.index_queue1[:] = -1
                    model.index_queue2[:] = -1

                    embed_neg1, plabels_neg1, neg_idx1 = get_neg_sample(
                        embed_u1, idx_u1, plabels1,
                        model.embed_queue1, model.index_queue1, model.queue_pointer1, model.valid_pointer1, cfg
                    )
                    embed_neg2, plabels_neg2, neg_idx2 = get_neg_sample(
                        embed_u2, idx_u2, plabels2,
                        model.embed_queue2, model.index_queue2, model.queue_pointer2, model.valid_pointer2, cfg
                    )

                    # 分布式收集负样本
                    if cfg.dist:
                        # 确保只在 gather 之前对这些样本进行操作，否则数据会翻倍
                        embed_neg1 = commu_utils.concat_all_gather(embed_neg1)
                        plabels_neg1 = commu_utils.concat_all_gather(plabels_neg1)
                        neg_idx1 = commu_utils.concat_all_gather(neg_idx1)  # 负样本的全局索引也要 gather

                        embed_neg2 = commu_utils.concat_all_gather(embed_neg2)
                        plabels_neg2 = commu_utils.concat_all_gather(plabels_neg2)
                        neg_idx2 = commu_utils.concat_all_gather(neg_idx2)  # 负样本的全局索引也要 gather

                    # 2.8) Compute unsupervised losses
                    unsup_loss1 = unsup_loss_fn(
                        embed_pos1, embed_pos2, embed_neg2,
                        plabels_pos1, plabels_neg2,
                        confs_pos2, pos_global1, pos_global2, neg_idx2
                    )
                    unsup_loss2 = unsup_loss_fn(
                        embed_pos2, embed_pos1, embed_neg1,
                        plabels_pos2, plabels_neg1,
                        confs_pos1, pos_global2, pos_global1, neg_idx1
                    )
                    unsup_loss = unsup_loss1 + unsup_loss2
                    loss_out['unsup_loss'] = (unsup_loss, embed_pos1.size(0))

                    # Add weighted unsupervised component
                    total_loss = total_loss + cfg.loss_weight[1] * unsup_loss

        # --- 3) Return ---
        return total_loss, loss_out, infos
    return model_fn
