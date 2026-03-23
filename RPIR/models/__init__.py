import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

import os.path as osp
import itertools
import random

from RPIR.utils.utils import load_model, load_DDPModel
from RPIR.utils.log_helper import init_log

from RPIR.models.RoPE_ND_ViT import rope_nd_vit, RoPE_Layer_scale_init_Block, RoPE_HGNN_Layer_scale_init_Block, RoPEAttention, RoPEAttention_HGNN
from RPIR.models.RoPE_ND_ViT import vit, Layer_scale_init_Block, Attention
from timm.models.vision_transformer import Mlp
from functools import partial

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, GINConv, GPSConv, LEConv, TransformerConv
from RPIR.utils.utils import fully_connected_edge_index

init_log('group')
logger = logging.getLogger('group')

# early fusion
# class ModelBuilder(nn.Module):
#     def __init__(self, config):
#         super(ModelBuilder, self).__init__()
#         print("early fusion！")
#
#         self.config = config
#         num_features_joint = config.structure.joint_embedding.num_features
#         self.ind_coord_axis = config.structure.ind_rope.coord_axis
#         if config.dataset.name == 'volleyball':
#             activities_num_classes = config.dataset.volleyball.activities_num_classes
#             actions_num_classes = config.dataset.volleyball.actions_num_classes
#         elif config.dataset.name == 'nba':
#             activities_num_classes = config.dataset.nba.activities_num_classes
#             actions_num_classes = config.dataset.nba.actions_num_classes
#
#         self.joint_embedding = nn.Sequential(nn.Linear(2, num_features_joint),
#                                              # nn.LayerNorm(num_features_joint),  # 新增
#                                             nn.LeakyReLU())
#
#         if config.structure.joint_rope.use == 'rope_attn_hgnn':
#             self.joint_rope = rope_nd_vit(block_layers=RoPE_HGNN_Layer_scale_init_Block, coor_ndim=2, rope_theta=10.0,
#                                           embed_dim=num_features_joint,
#                                           num_heads=config.structure.joint_rope.num_heads,
#                                           depth=config.structure.joint_rope.depth, mlp_ratio=4.,
#                                           qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
#                                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                                           act_layer=nn.GELU, Attention_block=RoPEAttention_HGNN, Mlp_block=Mlp,
#                                           init_scale=1e-4)
#             print("关节点间推理使用 rope + attn + hgnn 。")
#         elif config.structure.joint_rope.use == 'rope_attn':
#             self.joint_rope = rope_nd_vit(block_layers=RoPE_Layer_scale_init_Block, coor_ndim=2, rope_theta=10.0,
#                                                  embed_dim=num_features_joint,
#                                                  num_heads=config.structure.joint_rope.num_heads,
#                                                  depth=config.structure.joint_rope.depth, mlp_ratio=4.,
#                                                  qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
#                                                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                                                  act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
#             print("关节点间推理使用 rope + attn 。")
#         elif config.structure.joint_rope.use == 'attn':
#             self.joint_rope = vit(block_layers=Layer_scale_init_Block, embed_dim=num_features_joint,
#                                   num_heads=config.structure.joint_rope.num_heads, depth=config.structure.joint_rope.depth,
#                                   mlp_ratio=4., qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
#                                   norm_layer=nn.LayerNorm, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4)
#             print("关节点间推理使用 attn 。")
#         else:
#             print("关节点间不进行任何推理。")
#
#
#         if config.structure.ind_rope.use == 'rope_attn_hgnn':
#             self.ind_rope = rope_nd_vit(block_layers=RoPE_HGNN_Layer_scale_init_Block, coor_ndim=len(self.ind_coord_axis),
#                                         rope_theta=10.0,
#                                         embed_dim=num_features_joint,
#                                         num_heads=config.structure.ind_rope.num_heads,
#                                         depth=config.structure.ind_rope.depth, mlp_ratio=4.,
#                                         qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
#                                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                                         act_layer=nn.GELU, Attention_block=RoPEAttention_HGNN, Mlp_block=Mlp,
#                                         init_scale=1e-4)
#             print("个体骨架间推理使用 rope + attn + hgnn 。")
#         elif config.structure.ind_rope.use == 'rope_attn':
#             self.ind_rope = rope_nd_vit(block_layers=RoPE_Layer_scale_init_Block, coor_ndim=len(self.ind_coord_axis), rope_theta=10.0,
#                                           embed_dim=num_features_joint,
#                                           num_heads=config.structure.ind_rope.num_heads,
#                                           depth=config.structure.ind_rope.depth, mlp_ratio=4.,
#                                           qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
#                                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                                           act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
#             print("个体骨架间推理使用 rope + attn 。")
#         elif config.structure.ind_rope.use == 'attn':
#             self.ind_rope = vit(block_layers=Layer_scale_init_Block, embed_dim=num_features_joint,
#                                   num_heads=config.structure.ind_rope.num_heads, depth=config.structure.ind_rope.depth,
#                                   mlp_ratio=4., qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
#                                   norm_layer=nn.LayerNorm, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4)
#             print("个体骨架间推理使用 attn 。")
#         else:
#             print("个体骨架间不进行任何推理。")
#
#         self.classifier_activity = nn.Linear(num_features_joint, activities_num_classes)
#         # self.classifier_activity = nn.Sequential(nn.LayerNorm(num_features_joint),
#         #                                          nn.Linear(num_features_joint, activities_num_classes))
#
#         if config.train.action_loss:
#             self.classifier_action = nn.Linear(num_features_joint, actions_num_classes)
#             print("初始化个体动作分类器。")
#
#         # 新增
#         self.norm1 = nn.LayerNorm(num_features_joint)
#         self.norm2 = nn.LayerNorm(num_features_joint)
#
#         self.load_checkpoint(config)
#
#     def load_checkpoint(self, config):
#         if config.checkpoint is not None:
#             assert osp.exists(config.checkpoint), 'checkpoint file does not exist'
#             logger.info("load check point: " + str(config.checkpoint))
#             load_DDPModel(self, str(config.checkpoint))
#             print("加载模型：{}".format(str(config.checkpoint)))
#
#     def forward(self, top_coordinates, orientations, poses, tracks):
#         output = {}
#
#         # neck = (poses[:, :, 5, :] + poses[:, :, 6, :]) / 2.0
#         # neck = neck.unsqueeze(dim=2)
#         # nose = poses[:, :, 0, :].unsqueeze(dim=2)
#         # poses = torch.cat([nose, neck, poses[:, :, 1:, :]], dim=2)
#
#         B, T, N = tracks.shape[0:3]
#         N_point = poses.shape[2]
#
#         # 不要删除一下代码！！！！：人为生成ID-Switch，压力测试
#         # N_sample_IDSW = int(15 * B / 16)
#         # poses = torch.reshape(poses, (B, T, N, N_point, -1))
#         # sample_index = random.sample(range(B), N_sample_IDSW)   # 随机选择样本
#         # for b in sample_index:
#         #     # 随机选取时间区间
#         #     switch_len = random.choices(population=[1, 2, 3, 4], weights=[0.25, 0.25, 0.25, 0.25])[0]   # 1. 采样长度
#         #     t1 = random.randint(1, T - switch_len - 1)  # 2. 采样起点（避开边界）
#         #     t2 = t1 + switch_len  # [t1, t2)
#         #     # 选取时间区间内最近的两人
#         #     t_ref = (t1 + t2) // 2
#         #     pos = (tracks[b, t_ref, :, :2] + tracks[b, t_ref, :, 2:]) / 2.0  # [N, 2]
#         #     dist = torch.cdist(pos, pos)  # [N, N]
#         #     dist.fill_diagonal_(float('inf'))
#         #     flat_idx = torch.argmin(dist)
#         #     i = flat_idx // dist.size(1)
#         #     j = flat_idx % dist.size(1)
#         #     # skeleton
#         #     poses[b, t1:t2, i], poses[b, t1:t2, j] = poses[b, t1:t2, j].clone(), poses[b, t1:t2, i].clone()
#         #     # trajectory
#         #     tracks[b, t1:t2, i], tracks[b, t1:t2, j] = tracks[b, t1:t2, j].clone(), tracks[b, t1:t2, i].clone()
#         # poses = torch.reshape(poses, (B, T * N, N_point, -1))
#
#         # joint_embedding
#         joint_embedding = self.joint_embedding(poses)
#         joint_features = torch.reshape(joint_embedding, [B * T * N, N_point, -1])
#
#         # joint rope
#         if self.config.structure.joint_rope.use == 'rope_attn_hgnn' or self.config.structure.joint_rope.use == 'rope_attn':
#             joint_coor = torch.reshape(poses, [B * T * N, N_point, 2])
#             joint_features_02 = self.joint_rope(joint_features, joint_coor)
#             # joint_features_02 = self.norm1(joint_features_02)   # 新增
#             ind_features = joint_features_02[:, 0]
#             ind_features = torch.reshape(ind_features, [B, T, N, -1])
#             ind_features = torch.mean(ind_features, dim=1)
#             joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
#             joint_features_mean = torch.mean(torch.mean(joint_features, dim=3), dim=1)
#             # ind_features = torch.cat([ind_features, joint_features_mean], dim=-1)
#             ind_features = (ind_features + joint_features_mean) / 2.0
#         elif self.config.structure.joint_rope.use == 'attn':
#             joint_features_02 = self.joint_rope(joint_features)
#             ind_features = joint_features_02[:, 0]
#             ind_features = torch.reshape(ind_features, [B, T, N, -1])
#             ind_features = torch.mean(ind_features, dim=1)
#             joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
#             joint_features_mean = torch.mean(torch.mean(joint_features, dim=3), dim=1)
#             # ind_features = torch.cat([ind_features, joint_features_mean], dim=-1)
#             ind_features = (ind_features + joint_features_mean) / 2.0
#         else:
#             ind_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
#             ind_features = torch.mean(torch.mean(ind_features, dim=3), dim=1)
#
#         # 处理个体坐标 以及  ind rope
#         if self.config.structure.ind_rope.use == 'rope_attn_hgnn' or self.config.structure.ind_rope.use == 'rope_attn':
#             center = (tracks[:, :, :, :2] + tracks[:, :, :, 2:]) / 2.0
#             ind_coord_x = center[:, int(T/2), :, 0]
#             ind_coord_y = center[:, int(T/2), :, 1]
#             ind_coord_topx = top_coordinates[:, :, 0]
#             ind_coord_topy = top_coordinates[:, :, 1]
#             _, ind_coord_o = torch.max(orientations, dim=2)
#             ind_coord_o = ind_coord_o / 8.0
#             ind_coord = []
#             if 'x' in self.ind_coord_axis:
#                 ind_coord.append(ind_coord_x)
#             if 'y' in self.ind_coord_axis:
#                 ind_coord.append(ind_coord_y)
#             if 'topx' in self.ind_coord_axis:
#                 ind_coord.append(ind_coord_topx)
#             if 'topy' in self.ind_coord_axis:
#                 ind_coord.append(ind_coord_topy)
#             if 'o' in self.ind_coord_axis:
#                 ind_coord.append(ind_coord_o)
#             ind_coord = torch.stack(ind_coord, dim=-1).to(device=poses.device)
#             ind_features_02 = self.ind_rope(ind_features, ind_coord)
#             # ind_features_02 = self.norm2(ind_features_02)   # 新增
#             group_features = ind_features_02[:, 0]
#             ind_features_mean = torch.mean(ind_features, dim=1)
#             # group_features = torch.cat([group_features, ind_features_mean], dim=1)
#             group_features = (group_features + ind_features_mean) / 2.0
#         elif self.config.structure.ind_rope.use == 'attn':
#             ind_features_02 = self.ind_rope(ind_features)
#             group_features = ind_features_02[:, 0]
#             ind_features_mean = torch.mean(ind_features, dim=1)
#             # group_features = torch.cat([group_features, ind_features_mean], dim=1)
#             group_features = (group_features + ind_features_mean) / 2.0
#         else:
#             group_features = torch.mean(ind_features, dim=1)
#
#         output['ind_features'] = ind_features
#         output['group_features'] = group_features
#
#         activities_scores = self.classifier_activity(group_features)
#         output['activities_scores'] = activities_scores
#
#         if self.config.train.action_loss:
#             actions_scores = self.classifier_action(ind_features)
#             output['actions_scores'] = actions_scores
#
#         return output


# late fusion
class ModelBuilder(nn.Module):
    def __init__(self, config):
        super(ModelBuilder, self).__init__()
        print("late fusion！")

        self.config = config
        num_features_joint = config.structure.joint_embedding.num_features
        self.ind_coord_axis = config.structure.ind_rope.coord_axis
        if config.dataset.name == 'volleyball':
            activities_num_classes = config.dataset.volleyball.activities_num_classes
            actions_num_classes = config.dataset.volleyball.actions_num_classes
        elif config.dataset.name == 'nba':
            activities_num_classes = config.dataset.nba.activities_num_classes
            actions_num_classes = config.dataset.nba.actions_num_classes

        self.joint_embedding = nn.Sequential(nn.Linear(2, num_features_joint),
                                             # nn.LayerNorm(num_features_joint),  # 新增
                                            nn.LeakyReLU())

        if config.structure.joint_rope.use == 'rope_attn_hgnn':
            self.joint_rope = rope_nd_vit(block_layers=RoPE_HGNN_Layer_scale_init_Block, coor_ndim=2, rope_theta=10.0,
                                          embed_dim=num_features_joint,
                                          num_heads=config.structure.joint_rope.num_heads,
                                          depth=config.structure.joint_rope.depth, mlp_ratio=4.,
                                          qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          act_layer=nn.GELU, Attention_block=RoPEAttention_HGNN, Mlp_block=Mlp,
                                          init_scale=1e-4)
            print("关节点间推理使用 rope + attn + hgnn 。")
        elif config.structure.joint_rope.use == 'rope_attn':
            self.joint_rope = rope_nd_vit(block_layers=RoPE_Layer_scale_init_Block, coor_ndim=2, rope_theta=10.0,
                                                 embed_dim=num_features_joint,
                                                 num_heads=config.structure.joint_rope.num_heads,
                                                 depth=config.structure.joint_rope.depth, mlp_ratio=4.,
                                                 qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                 act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
            print("关节点间推理使用 rope + attn 。")
        elif config.structure.joint_rope.use == 'attn':
            self.joint_rope = vit(block_layers=Layer_scale_init_Block, embed_dim=num_features_joint,
                                  num_heads=config.structure.joint_rope.num_heads, depth=config.structure.joint_rope.depth,
                                  mlp_ratio=4., qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                  norm_layer=nn.LayerNorm, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4)
            print("关节点间推理使用 attn 。")
        elif config.structure.joint_rope.use == 'gcn':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(GCNConv(num_features_joint, num_features_joint))
            print("关节点间推理使用 gcn 。")
        elif config.structure.joint_rope.use == 'GraphSAGE':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(SAGEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("关节点间推理使用 GraphSAGE 。")
        elif config.structure.joint_rope.use == 'gat':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(GATConv(in_channels=num_features_joint,
                                               out_channels=num_features_joint,
                                               heads=4,
                                               concat=False,
                                               dropout=0.4))
            print("关节点间推理使用 gat 。")
        elif config.structure.joint_rope.use == 'gin':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(GINConv(nn.Sequential(nn.Linear(num_features_joint, num_features_joint),
                                                             nn.ReLU(),
                                                             nn.Linear(num_features_joint, num_features_joint))))
            print("关节点间推理使用 gin 。")
        elif config.structure.joint_rope.use == 'LightGCN':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(LEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("关节点间推理使用 LightGCN 。")
        elif config.structure.joint_rope.use == 'GraphTransformer':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(TransformerConv(in_channels=num_features_joint,
                                                       out_channels=num_features_joint,
                                                       heads=4, concat=False, dropout=0.4))
            print("关节点间推理使用 GraphTransformer 。")
        elif config.structure.joint_rope.use == 'gatv2':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(GATv2Conv(in_channels=num_features_joint,
                                                 out_channels=num_features_joint,
                                                 heads=4,
                                                 concat=False,
                                                 dropout=0.4))
            print("关节点间推理使用 gatv2 。")
        elif config.structure.joint_rope.use == 'GraphGPS':
            self.joint_rope = nn.ModuleList()
            for _ in range(config.structure.joint_rope.GNN_num_layers):
                self.joint_rope.append(GPSConv(channels=num_features_joint,
                                               conv=GCNConv(num_features_joint, num_features_joint),
                                               heads=4, dropout=0.4))
            print("关节点间推理使用 GraphGPS 。")
        else:
            print("关节点间不进行任何推理。")


        if config.structure.ind_rope.use == 'rope_attn_hgnn':
            self.ind_rope = rope_nd_vit(block_layers=RoPE_HGNN_Layer_scale_init_Block, coor_ndim=len(self.ind_coord_axis),
                                        rope_theta=10.0,
                                        embed_dim=num_features_joint,
                                        num_heads=config.structure.ind_rope.num_heads,
                                        depth=config.structure.ind_rope.depth, mlp_ratio=4.,
                                        qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                        act_layer=nn.GELU, Attention_block=RoPEAttention_HGNN, Mlp_block=Mlp,
                                        init_scale=1e-4)
            print("个体骨架间推理使用 rope + attn + hgnn 。")
        elif config.structure.ind_rope.use == 'rope_attn':
            self.ind_rope = rope_nd_vit(block_layers=RoPE_Layer_scale_init_Block, coor_ndim=len(self.ind_coord_axis), rope_theta=10.0,
                                          embed_dim=num_features_joint,
                                          num_heads=config.structure.ind_rope.num_heads,
                                          depth=config.structure.ind_rope.depth, mlp_ratio=4.,
                                          qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
            print("个体骨架间推理使用 rope + attn 。")
        elif config.structure.ind_rope.use == 'attn':
            self.ind_rope = vit(block_layers=Layer_scale_init_Block, embed_dim=num_features_joint,
                                  num_heads=config.structure.ind_rope.num_heads, depth=config.structure.ind_rope.depth,
                                  mlp_ratio=4., qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                  norm_layer=nn.LayerNorm, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4)
            print("个体骨架间推理使用 attn 。")
        elif config.structure.ind_rope.use == 'gcn':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(GCNConv(num_features_joint, num_features_joint))
            print("个体骨架间推理使用 gcn 。")
        elif config.structure.ind_rope.use == 'GraphSAGE':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(SAGEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("个体骨架间推理使用 GraphSAGE 。")
        elif config.structure.ind_rope.use == 'gat':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(GATConv(in_channels=num_features_joint,
                                               out_channels=num_features_joint,
                                               heads=4,
                                               concat=False,
                                               dropout=0.4))
            print("个体骨架间推理使用 gat 。")
        elif config.structure.ind_rope.use == 'gin':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(GINConv(nn.Sequential(nn.Linear(num_features_joint, num_features_joint),
                                                             nn.ReLU(),
                                                             nn.Linear(num_features_joint, num_features_joint))))
            print("个体骨架间推理使用 gin 。")
        elif config.structure.ind_rope.use == 'LightGCN':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(LEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("个体骨架间推理使用 LightGCN 。")
        elif config.structure.ind_rope.use == 'GraphTransformer':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(TransformerConv(in_channels=num_features_joint,
                                                       out_channels=num_features_joint,
                                                       heads=4, concat=False, dropout=0.4))
            print("个体骨架间推理使用 GraphTransformer 。")
        elif config.structure.ind_rope.use == 'gatv2':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(GATv2Conv(in_channels=num_features_joint,
                                                 out_channels=num_features_joint,
                                                 heads=4,
                                                 concat=False,
                                                 dropout=0.4))
            print("个体骨架间推理使用 gatv2 。")
        elif config.structure.ind_rope.use == 'GraphGPS':
            self.ind_rope = nn.ModuleList()
            for _ in range(config.structure.ind_rope.GNN_num_layers):
                self.ind_rope.append(GPSConv(channels=num_features_joint,
                                               conv=GCNConv(num_features_joint, num_features_joint),
                                               heads=4, dropout=0.4))
            print("个体骨架间推理使用 GraphGPS 。")
        else:
            print("个体骨架间不进行任何推理。")

        self.classifier_activity = nn.Linear(num_features_joint, activities_num_classes)
        # self.classifier_activity = nn.Sequential(nn.LayerNorm(num_features_joint),
        #                                          nn.Linear(num_features_joint, activities_num_classes))

        if config.train.action_loss:
            self.classifier_action = nn.Linear(num_features_joint, actions_num_classes)
            print("初始化个体动作分类器。")

        # 新增
        self.norm1 = nn.LayerNorm(num_features_joint)
        self.norm2 = nn.LayerNorm(num_features_joint)

        self.load_checkpoint(config)

    def load_checkpoint(self, config):
        if config.checkpoint is not None:
            assert osp.exists(config.checkpoint), 'checkpoint file does not exist'
            logger.info("load check point: " + str(config.checkpoint))
            load_DDPModel(self, str(config.checkpoint))
            print("加载模型：{}".format(str(config.checkpoint)))

    def forward(self, top_coordinates, orientations, poses, tracks):
        output = {}

        B, T, N = tracks.shape[0:3]
        N_point = poses.shape[2]

        # 不要删除一下代码！！！！：人为生成ID-Switch，压力测试
        N_sample_IDSW = int(16 * B / 16)    # 按比例（最后一个batch不足16）
        poses = torch.reshape(poses, (B, T, N, N_point, -1))
        sample_index = random.sample(range(B), N_sample_IDSW)   # 随机选择样本
        for b in sample_index:
            # 随机选取时间区间
            switch_len = random.choices(population=[1, 2, 3, 4], weights=[0.25, 0.25, 0.25, 0.25])[0]   # 1. 采样长度
            t1 = random.randint(1, T - switch_len - 1)  # 2. 采样起点（避开边界）
            t2 = t1 + switch_len  # [t1, t2)
            # 选取时间区间内最近的两人
            t_ref = (t1 + t2) // 2
            pos = (tracks[b, t_ref, :, :2] + tracks[b, t_ref, :, 2:]) / 2.0  # [N, 2]
            dist = torch.cdist(pos, pos)  # [N, N]
            # print(dist.shape)
            dist.fill_diagonal_(float('inf'))
            flat_idx = torch.argmin(dist)
            i = flat_idx // dist.size(1)
            j = flat_idx % dist.size(1)
            # skeleton
            poses[b, t1:t2, i], poses[b, t1:t2, j] = poses[b, t1:t2, j].clone(), poses[b, t1:t2, i].clone()
            # trajectory
            tracks[b, t1:t2, i], tracks[b, t1:t2, j] = tracks[b, t1:t2, j].clone(), tracks[b, t1:t2, i].clone()
        poses = torch.reshape(poses, (B, T * N, N_point, -1))

        # joint_embedding
        joint_embedding = self.joint_embedding(poses)
        joint_features = torch.reshape(joint_embedding, [B * T * N, N_point, -1])

        # joint rope
        if self.config.structure.joint_rope.use == 'rope_attn_hgnn' or self.config.structure.joint_rope.use == 'rope_attn':
            joint_coor = torch.reshape(poses, [B * T * N, N_point, 2])
            joint_features_02 = self.joint_rope(joint_features, joint_coor)
            # joint_features_02 = self.norm1(joint_features_02)   # 新增
            ind_features = joint_features_02[:, 0]
            ind_features = torch.reshape(ind_features, [B, T, N, -1])
            # ind_features = torch.mean(ind_features, dim=1)
            joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
            joint_features_mean = torch.mean(joint_features, dim=3)
            # ind_features = torch.cat([ind_features, joint_features_mean], dim=-1)
            ind_features = (ind_features + joint_features_mean) / 2.0
        elif self.config.structure.joint_rope.use == 'attn':
            joint_features_02 = self.joint_rope(joint_features)
            ind_features = joint_features_02[:, 0]
            ind_features = torch.reshape(ind_features, [B, T, N, -1])
            # ind_features = torch.mean(ind_features, dim=1)
            joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
            joint_features_mean = torch.mean(joint_features, dim=3)
            # ind_features = torch.cat([ind_features, joint_features_mean], dim=-1)
            ind_features = (ind_features + joint_features_mean) / 2.0
        elif self.config.structure.joint_rope.use in ['gcn', 'GraphSAGE', 'gat', 'gin', 'LightGCN', 'GraphTransformer', 'gatv2', 'GraphGPS']:
            edge_index_one = fully_connected_edge_index(N_point).to(device=joint_features.device)
            # 为每个 batch 样本创建偏移量：0, N_joint, 2*N_joint, ... (B-1)*N_joint
            offset = torch.arange(joint_features.shape[0], device=edge_index_one.device).repeat_interleave(edge_index_one.shape[1]) * N_point
            # 扩展维度 edge_index
            edge_index_batch = edge_index_one.repeat(1, joint_features.shape[0]) + offset.unsqueeze(0)
            joint_features_02 = torch.reshape(joint_features, [B * T * N * N_point, -1])
            for conv in self.joint_rope:
                joint_features_02 = F.relu(conv(joint_features_02, edge_index_batch))
            joint_features_02 = torch.reshape(joint_features_02, [B, T, N, N_point, -1])
            ind_features = torch.mean(joint_features_02, dim=3)
            joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
            joint_features_mean = torch.mean(joint_features, dim=3)
            ind_features = (ind_features + joint_features_mean) / 2.0
        else:
            ind_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
            ind_features = torch.mean(ind_features, dim=3)

        # 处理个体坐标 以及  ind rope
        if self.config.structure.ind_rope.use == 'rope_attn_hgnn' or self.config.structure.ind_rope.use == 'rope_attn':
            center = (tracks[:, :, :, :2] + tracks[:, :, :, 2:]) / 2.0
            ind_coord_x = center[:, :, :, 0]
            ind_coord_y = center[:, :, :, 1]
            ind_coord_topx = top_coordinates[:, :, 0]
            ind_coord_topx = torch.cat([ind_coord_topx.unsqueeze(dim=1)] * T, dim=1)
            ind_coord_topy = top_coordinates[:, :, 1]
            ind_coord_topy = torch.cat([ind_coord_topy.unsqueeze(dim=1)] * T, dim=1)
            _, ind_coord_o = torch.max(orientations, dim=2)
            ind_coord_o = ind_coord_o / 8.0
            ind_coord_o = torch.cat([ind_coord_o.unsqueeze(dim=1)] * T, dim=1)
            ind_coord = []
            if 'x' in self.ind_coord_axis:
                ind_coord.append(ind_coord_x)
            if 'y' in self.ind_coord_axis:
                ind_coord.append(ind_coord_y)
            if 'topx' in self.ind_coord_axis:
                ind_coord.append(ind_coord_topx)
            if 'topy' in self.ind_coord_axis:
                ind_coord.append(ind_coord_topy)
            if 'o' in self.ind_coord_axis:
                ind_coord.append(ind_coord_o)
            ind_coord = torch.stack(ind_coord, dim=-1).to(device=poses.device)
            # pdb.set_trace()
            ind_coord = torch.reshape(ind_coord, [B * T, N, len(self.ind_coord_axis)])
            ind_features = torch.reshape(ind_features, [B * T, N, -1])
            ind_features_02 = self.ind_rope(ind_features, ind_coord)
            # ind_features_02 = self.norm2(ind_features_02)   # 新增
            group_features = ind_features_02[:, 0]
            group_features = torch.reshape(group_features, [B, T, -1])
            ind_features = torch.reshape(ind_features, [B, T, N, -1])
            ind_features_mean = torch.mean(ind_features, dim=2)
            # group_features = torch.cat([group_features, ind_features_mean], dim=1)
            group_features = (group_features + ind_features_mean) / 2.0
            # group_features = torch.mean(group_features, dim=1)
        elif self.config.structure.ind_rope.use == 'attn':
            ind_features = torch.reshape(ind_features, [B * T, N, -1])
            ind_features_02 = self.ind_rope(ind_features)
            group_features = ind_features_02[:, 0]
            group_features = torch.reshape(group_features, [B, T, -1])
            ind_features = torch.reshape(ind_features, [B, T, N, -1])
            ind_features_mean = torch.mean(ind_features, dim=2)
            # group_features = torch.cat([group_features, ind_features_mean], dim=1)
            group_features = (group_features + ind_features_mean) / 2.0
            # group_features = torch.mean(group_features, dim=1)
        elif self.config.structure.ind_rope.use in ['gcn', 'GraphSAGE', 'gat', 'gin', 'LightGCN', 'GraphTransformer', 'gatv2', 'GraphGPS']:
            edge_index_one = fully_connected_edge_index(N).to(device=poses.device)
            # 为每个 batch 样本创建偏移量：0, num_box, 2*num_box, ... (B-1)*num_box
            offset = torch.arange(B * T, device=edge_index_one.device).repeat_interleave(edge_index_one.shape[1]) * N
            # 扩展维度 edge_index
            edge_index_batch = edge_index_one.repeat(1, B * T) + offset.unsqueeze(0)
            ind_features_02 = torch.reshape(ind_features, [B * T * N, -1])
            for conv in self.ind_rope:
                ind_features_02 = F.relu(conv(ind_features_02, edge_index_batch))
            ind_features_02 = torch.reshape(ind_features_02, [B, T, N, -1])
            group_features = torch.mean(ind_features_02, dim=2)
            ind_features_mean = torch.mean(ind_features, dim=2)
            group_features = (group_features + ind_features_mean) / 2.0
        else:
            group_features = torch.mean(ind_features, dim=2)
            # group_features = torch.mean(group_features, dim=1)

        # temporal fusion
        ind_features = torch.mean(ind_features, dim=1)
        group_features = torch.mean(group_features, dim=1)

        output['ind_features'] = ind_features
        output['group_features'] = group_features

        activities_scores = self.classifier_activity(group_features)
        output['activities_scores'] = activities_scores

        if self.config.train.action_loss:
            actions_scores = self.classifier_action(ind_features)
            output['actions_scores'] = actions_scores

        return output
