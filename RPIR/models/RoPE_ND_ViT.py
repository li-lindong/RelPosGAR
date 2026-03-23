import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from functools import partial
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

# from models_v2 import vit_models, Layer_scale_init_Block, Attention
from RPIR.models.spectral_cluster.spectralcluster import SpectralClusterer
import RPIR.models.HGNN.hypergraph_utils as hgut
from RPIR.models.HGNN.HGNN import HGNN_conv
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == x.shape:
        return freqs_cis
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        # 对查询和键向量应用旋转位置嵌入（除了第一个cls token）
        rotated_q, rotated_k = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        # 将未经过旋转位置嵌入处理的第一个token与经过处理的剩余tokens拼接起来
        q = torch.cat([q[:, :, :1], rotated_q], dim=2)
        k = torch.cat([k[:, :, :1], rotated_k], dim=2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class RoPEAttention_HGNN(Attention):
    """Multi-head Attention block with rotary position embeddings."""
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.hgnn = HGNN_conv(dim, dim)
        self.attn_hg_drop = nn.Dropout(attn_drop)

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        # 对查询和键向量应用旋转位置嵌入（除了第一个cls token）
        rotated_q, rotated_k = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        # 将未经过旋转位置嵌入处理的第一个token与经过处理的剩余tokens拼接起来
        q = torch.cat([q[:, :, :1], rotated_q], dim=2)
        k = torch.cat([k[:, :, :1], rotated_k], dim=2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x1 = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 超图推理
        q = q.transpose(1, 2).reshape(B, N, C)
        k = k.transpose(1, 2).reshape(B, N, C)
        v = v.transpose(1, 2).reshape(B, N, C)
        attn_hg = (q * (q.shape[-1] ** -0.5)) @ k.transpose(-2, -1)
        attn_hg = attn_hg.softmax(dim=-1)
        attn_hg = self.attn_hg_drop(attn_hg)
        x2 = []
        for b in range(B):
            # 根据注意力矩阵划分群组
            attn_hg_b = attn_hg[b, :, :]
            this_N = attn_hg_b.shape[0]
            this_cluster = SpectralClusterer(
                min_clusters=1,
                max_clusters=int(this_N), )
            this_social_group_predict = this_cluster.predict(attn_hg_b.detach().cpu().numpy())
            # 将群组划分转换为H矩阵
            n_edges = max(this_social_group_predict) + 1
            node_id = [i for i in range(this_N)]
            node_id = np.array(node_id)
            H_matrix = np.zeros((this_N, n_edges))
            H_matrix[node_id, this_social_group_predict] = 1
            # 超图推理
            G_matrix = hgut.generate_G_from_H(H_matrix)
            G_matrix = torch.tensor(G_matrix, device=x.device, dtype=x.dtype)
            x2.append(F.dropout(F.relu(self.hgnn(v[b, :, :], G_matrix))))
        x2 = torch.stack(x2, dim=0)

        x = (x1 + x2) / 2.0
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x

class RoPE_HGNN_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = RoPEAttention_HGNN
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x

def compute_mixed_cis(freqs: torch.Tensor, coor: torch.Tensor, num_heads: int):
    B, N, coor_ndim = coor.shape
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_cis = []
        for i in range(coor_ndim):
            coor_i = coor[:, :, i]
            freqs_cis.append((coor_i.unsqueeze(-1).unsqueeze(1) @ freqs[i].unsqueeze(1).unsqueeze(0)).view(B, depth, N, num_heads, -1).permute(0, 1, 3, 2, 4))
        freqs_cis = torch.polar(torch.ones_like(freqs_cis[0]), torch.stack(freqs_cis, dim=0).sum(dim=0))

    return freqs_cis

def init_random_nd_freqs(coor_ndim, dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """
    初始化各坐标轴的分频分量
    :param coor_ndim: 坐标轴维度
    :param dim: 特征维度
    :param num_heads: 自注意力模块头数
    :param theta:
    :param rotate: 是否采用旋转向量编码
    :return:
    """
    freqs = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        if rotate:
            angles = [torch.rand(1) * 2 * torch.pi + agl * torch.pi / (coor_ndim+1) for agl in range(1, coor_ndim+1)]
        else: angles = [torch.zeros(1)] * coor_ndim
        for j in range(len(angles)):
            freqs.append(torch.cat([mag * torch.cos(angles[j]), mag * torch.sin(torch.pi/2 + angles[j])], dim=-1))

    freqs = torch.stack(freqs, dim=0)
    freqs = torch.reshape(freqs, [coor_ndim, num_heads, -1])
    return freqs

class rope_nd_vit(nn.Module):
    def __init__(self, block_layers, coor_ndim=2, rope_theta=100.0, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.,
                 qkv_bias=False, drop_path_rate=0., qk_scale=None, attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4):
        """
        :param embed_dim: 每个头的特征维度
        :param num_heads: 多头自注意力的头数
        :param block_layers: 自注意力层（带RoPE）
        :param depth: 注意力块层数
        :param rope_theta:
        """
        super(rope_nd_vit, self).__init__()
        self.coor_ndim = coor_ndim
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        freqs = []
        for i, _ in enumerate(self.blocks):
            freqs.append(init_random_nd_freqs(coor_ndim=self.coor_ndim, dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta))
        freqs = torch.stack(freqs, dim=1).view(self.coor_ndim, len(self.blocks), -1)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

        self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)  # 提前为compute_mixed_cis函数设置num_heads参数，在调用时就不用传入该参数了

        self.norm = norm_layer(embed_dim)

    def forward(self, x, coor):
        """
        :param x: 节点特征 [B, N, D]
        :param coor: 每个节点的二维坐标 [B, N, self.coor_ndim]
        :return:
        """
        assert x.shape[0] == coor.shape[0]
        B, N, D = x.shape  # batch size, number of nodes, dimension of features

        # 嵌入分类头
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        freqs = self.compute_cis(freqs=self.freqs, coor=coor)
        for i, blk in enumerate(self.blocks):
            x = blk(x, freqs[:, i])

        x = self.norm(x)

        return x

class vit(nn.Module):
    def __init__(self, block_layers, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.,
                 qkv_bias=False, drop_path_rate=0., qk_scale=None, attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4):
        """
        :param embed_dim: 每个头的特征维度
        :param num_heads: 多头自注意力的头数
        :param block_layers: 自注意力层（带RoPE）
        :param depth: 注意力块层数
        :param rope_theta:
        """
        super(vit, self).__init__()
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """
        :param x: 节点特征 [B, N, D]
        :return:
        """
        B, N, D = x.shape  # batch size, number of nodes, dimension of features

        # 嵌入分类头
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # rope_nd_vit
    model = rope_nd_vit(block_layers=RoPE_Layer_scale_init_Block, coor_ndim=2, rope_theta=10.0, embed_dim=512, num_heads=1, depth=1, mlp_ratio=4.,
                 qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
    model.to(device)
    x = torch.randn(2, 17, 512).to(device)
    coor = torch.randn(2, 17, 2).to(device)
    y = model(x, coor)
    print(y.shape)

    # vit
    # model = vit(block_layers=Layer_scale_init_Block, embed_dim=512, num_heads=16, depth=12, mlp_ratio=4.,
    #              qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0., norm_layer=nn.LayerNorm,
    #              act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4)
    # model.to(device)
    # x = torch.randn(2, 17, 512).to(device)
    # y = model(x)
    # print(y.shape)

    # rope_nd_vit_hgnn
    # model = rope_nd_vit(block_layers=RoPE_HGNN_Layer_scale_init_Block, coor_ndim=2, rope_theta=10.0, embed_dim=512, num_heads=16, depth=24, mlp_ratio=4.,
    #              qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #              act_layer=nn.GELU, Attention_block=RoPEAttention_HGNN, Mlp_block=Mlp, init_scale=1e-4)
    # model.to(device)
    # x = torch.randn(2, 17, 512).to(device)
    # coor = torch.randn(2, 17, 2).to(device)
    # y = model(x, coor)
    # print(y.shape)

    # 利用calculate_flops计算
    # from calflops import calculate_flops
    # FLOPs, MACs, params = calculate_flops(model.cuda(), input_shape=(1, 17, 1024), output_as_string=False)
    # print(params, FLOPs)