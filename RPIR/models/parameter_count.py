import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import copy
import torch.nn.functional as F
from thop import profile
import torchvision.models as models
from calflops import calculate_flops
from RPIR.models.RoPE_ND_ViT import rope_nd_vit, RoPE_Layer_scale_init_Block, RoPEAttention
from RPIR.models.RoPE_ND_ViT import vit, Layer_scale_init_Block, Attention
from timm.models.vision_transformer import Mlp
from functools import partial
from easydict import EasyDict

from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, GINConv, GPSConv, LEConv, TransformerConv
from RPIR.utils.utils import fully_connected_edge_index


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        num_features_joint = 128
        self.ind_coord_axis = ['x', 'topx']

        self.joint_rope_use = 'GraphGPS'
        joint_rope_num_heads = 4
        joint_rope_depth = 6
        joint_rope_GNN_num_layer = 3

        self.ind_rope_use = 'GraphGPS'
        ind_rope_num_heads = 4
        ind_rope_depth = 6
        ind_rope_GNN_num_layer = 3

        self.action_loss = False
        activities_num_classes = 8
        actions_num_classes = 9

        self.joint_embedding = nn.Sequential(nn.Linear(2, num_features_joint),
                                            nn.LeakyReLU())

        if self.joint_rope_use == 'rope_attn':
            self.joint_rope = rope_nd_vit(block_layers=RoPE_Layer_scale_init_Block, coor_ndim=2, rope_theta=10.0,
                                                 embed_dim=num_features_joint, num_heads=joint_rope_num_heads,
                                                 depth=joint_rope_depth, mlp_ratio=4., qkv_bias=True, drop_path_rate=0.,
                                                 qk_scale=None, attn_drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                                 act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
            print("关节点间推理使用 rope + attn 。")
        elif self.joint_rope_use == 'attn':
            self.joint_rope = vit(block_layers=Layer_scale_init_Block, embed_dim=num_features_joint,
                                  num_heads=joint_rope_num_heads, depth=joint_rope_depth,
                                  mlp_ratio=4., qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                  norm_layer=nn.LayerNorm, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4)
            print("关节点间推理使用 attn 。")
        elif self.joint_rope_use == 'gcn':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(GCNConv(num_features_joint, num_features_joint))
            print("关节点间推理使用 gcn 。")
        elif self.joint_rope_use == 'GraphSAGE':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(SAGEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("关节点间推理使用 GraphSAGE 。")
        elif self.joint_rope_use == 'gat':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(GATConv(in_channels=num_features_joint,
                                               out_channels=num_features_joint,
                                               heads=4,
                                               concat=False,
                                               dropout=0.4))
            print("关节点间推理使用 gat 。")
        elif self.joint_rope_use == 'gin':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(GINConv(nn.Sequential(nn.Linear(num_features_joint, num_features_joint),
                                                             nn.ReLU(),
                                                             nn.Linear(num_features_joint, num_features_joint))))
            print("关节点间推理使用 gin 。")
        elif self.joint_rope_use == 'LightGCN':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(LEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("关节点间推理使用 LightGCN 。")
        elif self.joint_rope_use == 'GraphTransformer':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(TransformerConv(in_channels=num_features_joint,
                                                       out_channels=num_features_joint,
                                                       heads=4, concat=False, dropout=0.4))
            print("关节点间推理使用 GraphTransformer 。")
        elif self.joint_rope_use == 'gatv2':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(GATv2Conv(in_channels=num_features_joint,
                                                 out_channels=num_features_joint,
                                                 heads=4,
                                                 concat=False,
                                                 dropout=0.4))
            print("关节点间推理使用 gatv2 。")
        elif self.joint_rope_use == 'GraphGPS':
            self.joint_rope = nn.ModuleList()
            for _ in range(joint_rope_GNN_num_layer):
                self.joint_rope.append(GPSConv(channels=num_features_joint,
                                               conv=GCNConv(num_features_joint, num_features_joint),
                                               heads=4, dropout=0.4))
            print("关节点间推理使用 GraphGPS 。")
        else:
            print("关节点间不进行任何推理。")


        if self.ind_rope_use == 'rope_attn':
            self.ind_rope = rope_nd_vit(block_layers=RoPE_Layer_scale_init_Block, coor_ndim=len(self.ind_coord_axis), rope_theta=10.0,
                                          embed_dim=num_features_joint,
                                          num_heads=ind_rope_num_heads,
                                          depth=ind_rope_depth, mlp_ratio=4.,
                                          qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
            print("个体骨架间推理使用 rope + attn 。")
        elif self.ind_rope_use == 'attn':
            self.ind_rope = vit(block_layers=Layer_scale_init_Block, embed_dim=num_features_joint,
                                  num_heads=ind_rope_num_heads, depth=ind_rope_depth,
                                  mlp_ratio=4., qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0.,
                                  norm_layer=nn.LayerNorm, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4)
            print("个体骨架间推理使用 attn 。")
        elif self.ind_rope_use == 'gcn':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(GCNConv(num_features_joint, num_features_joint))
            print("个体骨架间推理使用 gcn 。")
        elif self.ind_rope_use == 'GraphSAGE':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(SAGEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("个体骨架间推理使用 GraphSAGE 。")
        elif self.ind_rope_use == 'gat':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(GATConv(in_channels=num_features_joint,
                                               out_channels=num_features_joint,
                                               heads=4,
                                               concat=False,
                                               dropout=0.4))
            print("个体骨架间推理使用 gat 。")
        elif self.ind_rope_use == 'gin':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(GINConv(nn.Sequential(nn.Linear(num_features_joint, num_features_joint),
                                                             nn.ReLU(),
                                                             nn.Linear(num_features_joint, num_features_joint))))
            print("个体骨架间推理使用 gin 。")
        elif self.ind_rope_use == 'LightGCN':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(LEConv(in_channels=num_features_joint, out_channels=num_features_joint))
            print("个体骨架间推理使用 LightGCN 。")
        elif self.ind_rope_use == 'GraphTransformer':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(TransformerConv(in_channels=num_features_joint,
                                                       out_channels=num_features_joint,
                                                       heads=4, concat=False, dropout=0.4))
            print("个体骨架间推理使用 GraphTransformer 。")
        elif self.ind_rope_use == 'gatv2':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(GATv2Conv(in_channels=num_features_joint,
                                                 out_channels=num_features_joint,
                                                 heads=4,
                                                 concat=False,
                                                 dropout=0.4))
            print("个体骨架间推理使用 gatv2 。")
        elif self.ind_rope_use == 'GraphGPS':
            self.ind_rope = nn.ModuleList()
            for _ in range(ind_rope_GNN_num_layer):
                self.ind_rope.append(GPSConv(channels=num_features_joint,
                                               conv=GCNConv(num_features_joint, num_features_joint),
                                               heads=4, dropout=0.4))
            print("个体骨架间推理使用 GraphGPS 。")
        else:
            print("个体骨架间不进行任何推理。")

        self.classifier_activity = nn.Linear(num_features_joint, activities_num_classes)

        if self.action_loss:
            self.classifier_action = nn.Linear(num_features_joint, actions_num_classes)
            print("初始化个体动作分类器。")

    # early fusion
    # def forward(self, x):
    #
    #     top_coordinates = torch.randn(1, 12, 2).cuda()
    #     orientations = torch.randn(1, 12, 8).cuda()
    #     poses = torch.randn(1, 84, 17, 2).cuda()
    #     tracks = torch.randn(1, 7, 12, 4).cuda()
    #
    #     output = {}
    #
    #     B, T, N = tracks.shape[0:3]
    #     N_point = poses.shape[2]
    #
    #     # joint_embedding
    #     joint_embedding = self.joint_embedding(poses)
    #     joint_features = torch.reshape(joint_embedding, [B * T * N, N_point, -1])
    #
    #     # joint rope
    #     if self.joint_rope_use == 'rope_attn_hgnn' or self.joint_rope_use == 'rope_attn':
    #         joint_coor = torch.reshape(poses, [B * T * N, N_point, 2])
    #         joint_features_02 = self.joint_rope(joint_features, joint_coor)
    #         # joint_features_02 = self.norm1(joint_features_02)   # 新增
    #         ind_features = joint_features_02[:, 0]
    #         ind_features = torch.reshape(ind_features, [B, T, N, -1])
    #         ind_features = torch.mean(ind_features, dim=1)
    #         joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
    #         joint_features_mean = torch.mean(torch.mean(joint_features, dim=3), dim=1)
    #         # ind_features = torch.cat([ind_features, joint_features_mean], dim=-1)
    #         ind_features = (ind_features + joint_features_mean) / 2.0
    #     elif self.joint_rope_use == 'attn':
    #         joint_features_02 = self.joint_rope(joint_features)
    #         ind_features = joint_features_02[:, 0]
    #         ind_features = torch.reshape(ind_features, [B, T, N, -1])
    #         ind_features = torch.mean(ind_features, dim=1)
    #         joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
    #         joint_features_mean = torch.mean(torch.mean(joint_features, dim=3), dim=1)
    #         # ind_features = torch.cat([ind_features, joint_features_mean], dim=-1)
    #         ind_features = (ind_features + joint_features_mean) / 2.0
    #     else:
    #         ind_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
    #         ind_features = torch.mean(torch.mean(ind_features, dim=3), dim=1)
    #
    #     # 处理个体坐标 以及  ind rope
    #     if self.ind_rope_use == 'rope_attn_hgnn' or self.ind_rope_use == 'rope_attn':
    #         center = (tracks[:, :, :, :2] + tracks[:, :, :, 2:]) / 2.0
    #         ind_coord_x = center[:, int(T / 2), :, 0]
    #         ind_coord_y = center[:, int(T / 2), :, 1]
    #         ind_coord_topx = top_coordinates[:, :, 0]
    #         ind_coord_topy = top_coordinates[:, :, 1]
    #         _, ind_coord_o = torch.max(orientations, dim=2)
    #         ind_coord_o = ind_coord_o / 8.0
    #         ind_coord = []
    #         if 'x' in self.ind_coord_axis:
    #             ind_coord.append(ind_coord_x)
    #         if 'y' in self.ind_coord_axis:
    #             ind_coord.append(ind_coord_y)
    #         if 'topx' in self.ind_coord_axis:
    #             ind_coord.append(ind_coord_topx)
    #         if 'topy' in self.ind_coord_axis:
    #             ind_coord.append(ind_coord_topy)
    #         if 'o' in self.ind_coord_axis:
    #             ind_coord.append(ind_coord_o)
    #         ind_coord = torch.stack(ind_coord, dim=-1).to(device=poses.device)
    #         ind_features_02 = self.ind_rope(ind_features, ind_coord)
    #         # ind_features_02 = self.norm2(ind_features_02)   # 新增
    #         group_features = ind_features_02[:, 0]
    #         ind_features_mean = torch.mean(ind_features, dim=1)
    #         # group_features = torch.cat([group_features, ind_features_mean], dim=1)
    #         group_features = (group_features + ind_features_mean) / 2.0
    #     elif self.ind_rope_use == 'attn':
    #         ind_features_02 = self.ind_rope(ind_features)
    #         group_features = ind_features_02[:, 0]
    #         ind_features_mean = torch.mean(ind_features, dim=1)
    #         # group_features = torch.cat([group_features, ind_features_mean], dim=1)
    #         group_features = (group_features + ind_features_mean) / 2.0
    #     else:
    #         group_features = torch.mean(ind_features, dim=1)
    #
    #     output['ind_features'] = ind_features
    #     output['group_features'] = group_features
    #
    #     activities_scores = self.classifier_activity(group_features)
    #     output['activities_scores'] = activities_scores
    #
    #     if self.action_loss:
    #         actions_scores = self.classifier_action(ind_features)
    #         output['actions_scores'] = actions_scores
    #
    #     return output

    # late fusion
    def forward(self, x):

        top_coordinates = torch.randn(1, 12, 2).cuda()
        orientations = torch.randn(1, 12, 8).cuda()
        poses = torch.randn(1, 84, 17, 2).cuda()
        tracks = torch.randn(1, 7, 12, 4).cuda()

        output = {}

        B, T, N = tracks.shape[0:3]
        N_point = poses.shape[2]

        # joint_embedding
        joint_embedding = self.joint_embedding(poses)
        joint_features = torch.reshape(joint_embedding, [B * T * N, N_point, -1])

        # joint rope
        if self.joint_rope_use == 'rope_attn_hgnn' or self.joint_rope_use == 'rope_attn':
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
        elif self.joint_rope_use == 'attn':
            joint_features_02 = self.joint_rope(joint_features)
            ind_features = joint_features_02[:, 0]
            ind_features = torch.reshape(ind_features, [B, T, N, -1])
            # ind_features = torch.mean(ind_features, dim=1)
            joint_features = torch.reshape(joint_features, [B, T, N, N_point, -1])
            joint_features_mean = torch.mean(joint_features, dim=3)
            # ind_features = torch.cat([ind_features, joint_features_mean], dim=-1)
            ind_features = (ind_features + joint_features_mean) / 2.0
        elif self.joint_rope_use in ['gcn', 'GraphSAGE', 'gat', 'gin', 'LightGCN', 'GraphTransformer', 'gatv2', 'GraphGPS']:
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
        if self.ind_rope_use == 'rope_attn_hgnn' or self.ind_rope_use == 'rope_attn':
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
            group_features = torch.mean(group_features, dim=1)
        elif self.ind_rope_use == 'attn':
            ind_features = torch.reshape(ind_features, [B * T, N, -1])
            ind_features_02 = self.ind_rope(ind_features)
            group_features = ind_features_02[:, 0]
            group_features = torch.reshape(group_features, [B, T, -1])
            ind_features = torch.reshape(ind_features, [B, T, N, -1])
            ind_features_mean = torch.mean(ind_features, dim=2)
            # group_features = torch.cat([group_features, ind_features_mean], dim=1)
            group_features = (group_features + ind_features_mean) / 2.0
            group_features = torch.mean(group_features, dim=1)
        elif self.ind_rope_use in ['gcn', 'GraphSAGE', 'gat', 'gin', 'LightGCN', 'GraphTransformer', 'gatv2', 'GraphGPS']:
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
            group_features = torch.mean(group_features, dim=1)

        output['ind_features'] = ind_features
        output['group_features'] = group_features

        activities_scores = self.classifier_activity(group_features)
        output['activities_scores'] = activities_scores

        if self.action_loss:
            actions_scores = self.classifier_action(ind_features)
            output['actions_scores'] = actions_scores

        return output

if __name__ == '__main__':
    # 计算模型的参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 实例化模型
    model = MyModel().cuda()

    # 计算模型的参数量
    print(f"Total trainable parameters: {count_parameters(model)}")

    # 利用calculate_flops计算
    FLOPs, MACs, params = calculate_flops(model.cuda(), input_shape=(1, 1024), output_as_string=False)
    print(params, FLOPs)