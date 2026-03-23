import pdb

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from functools import partial
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
# from models_v2 import vit_models, Layer_scale_init_Block, Attention
import time

# 定义Attention类，实现了多头自注意力机制
class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 查询、键、值线性变换
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力Dropout
        self.proj = nn.Linear(dim, dim)  # 输出线性变换
        self.proj_drop = nn.Dropout(proj_drop)  # 输出Dropout

    def forward(self, x):
        B, N, C = x.shape  # 获取输入张量的形状
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离查询、键、值

        q = q * self.scale  # 应用缩放因子

        attn = (q @ k.transpose(-2, -1))  # 计算注意力分数
        attn = attn.softmax(dim=-1)  # 软化注意力分数
        attn = self.attn_drop(attn)  # 应用Dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 应用注意力分数到值上
        x = self.proj(x)  # 线性变换
        x = self.proj_drop(x)  # 输出Dropout
        return x

class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    reshape_for_broadcast 函数的作用是调整频率张量 freqs_cis 的形状，以便它可以正确地广播到与输入张量 x 相匹配的形状。这种操作对于确保在应用旋转位置嵌入时，每个查询（q）和键（k）向量都能乘以相应的频率值至关重要。
    :param freqs_cis: 复数频率表，形状通常为 (N, D) 或 (L, N, D)，其中 N 是序列长度，D 是特征维度，L 是层数。
    :param x:  输入张量，通常是经过处理的查询或键向量，其形状可能为 [B, num_heads, N, head_dim]，其中 B 是批次大小，num_heads 是注意力头的数量，N 是序列长度，head_dim 是每个头的维度。
    :return:
    """
    ndim = x.ndim   # 返回张量的维度
    assert 0 <= 1 < ndim
    # 检查 freqs_cis 的形状是否匹配输入张量 x 的最后两个维度 (x.shape[-2], x.shape[-1]) 或最后三个维度 (x.shape[-3], x.shape[-2], x.shape[-1])。
    # 如果匹配最后一个二维形状，则创建一个新的形状列表 shape，其中除了最后两个维度外，其余维度都设为1。这样做的目的是为了后续可以进行正确的广播操作。
    # 如果匹配最后三维形状，则类似地创建一个新的形状列表 shape，但这次保留最后三个维度不变，其余维度设为1。
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    :param xq: 查询向量张量，形状为 [B, num_heads, N, head_dim]。
    :param xk: 键向量张量，形状与查询向量相同。
    :param freqs_cis: 复数频率表，用于生成旋转位置嵌入。
    :return:
    """
    # 转换为复数形式：这两行代码将输入的实数张量xq和xk转换为复数张量。首先，调整形状以适应复数表示（每两个相邻元素视为一个复数），然后使用torch.view_as_complex将其转换为复数形式。
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 广播频率表：使用reshape_for_broadcast函数调整freqs_cis的形状，以便它可以正确地广播到xq_的形状。这一步是为了确保每个查询和键向量都能乘以相应的频率值。
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 应用旋转位置嵌入：
    # （1）将调整后的频率表freqs_cis乘以查询和键向量的复数形式。这样做的目的是为了将旋转位置信息编码到这些向量中。
    # （2）使用torch.view_as_real将结果转换回实数形式，并通过flatten操作恢复原始形状。
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # 返回结果：最后，将处理后的查询和键向量转换回原始数据类型，并移动到正确的设备（CPU或GPU），然后返回。
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""

    # 重写父类的forward方法，增加对旋转位置嵌入的支持
    def forward(self, x, freqs_cis):
        B, N, C = x.shape  # 获取输入x的形状：[Batch, Num_Patches, Embedding_Dim]

        # 将输入x通过线性变换得到qkv（查询、键、值），并调整维度以适应多头注意力机制
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离出查询(q), 键(k), 值(v)

        # q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        # 对查询和键向量应用旋转位置嵌入（除了第一个cls token）
        rotated_q, rotated_k = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)
        # 将未经过旋转位置嵌入处理的第一个token与经过处理的剩余tokens拼接起来
        q = torch.cat([q[:, :, :1], rotated_q], dim=2)
        k = torch.cat([k[:, :, :1], rotated_k], dim=2)

        attn = (q * self.scale) @ k.transpose(-2, -1)   # 计算注意力分数，这里乘以缩放因子是为了稳定softmax计算中的梯度
        attn = attn.softmax(dim=-1) # 使用softmax函数将注意力分数转换为概率分布
        attn = self.attn_drop(attn) # 应用Dropout到注意力分数上

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 根据注意力分数加权求和得到输出特征
        x = self.proj(x)    # 线性变换输出特征
        x = self.proj_drop(x)   # 应用Dropout到最终输出上

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

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        # pdb.set_trace()
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """
    init_random_2d_freqs 函数用于生成二维旋转位置嵌入所需的复数频率表。这些频率表被用来在 RoPEAttention 中对查询（q）和键（k）向量应用旋转位置嵌入，从而增强模型对序列中相对位置信息的捕捉能力。
    :param dim: 每个头的维度（head_dim），即特征维度。
    :param num_heads: 注意力机制中的头数。
    :param theta: 控制频率衰减的速度，默认值为10.0。
    :param rotate: 是否随机初始化每个注意力头的角度，默认值为True。
    :return:
    """
    freqs_x = []
    freqs_y = []
    # 计算频率幅度：
    # torch.arange(0, dim, 4) 生成从0到dim（不包括dim），步长为4的索引数组。
    # [: (dim // 4)] 取前 dim // 4 个元素，确保我们有足够的频率成分来覆盖整个维度。
    # 1 / (theta ** ...) 使用指数函数计算频率的幅度，随着频率索引的增加，幅度逐渐减小，这有助于控制高频分量的影响。
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

class rope_2d_vit(nn.Module):
    def __init__(self, block_layers, rope_theta=100.0, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.,
                 qkv_bias=False, drop_path_rate=0., qk_scale=None, attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp, init_scale=1e-4):
        """
        :param embed_dim: 特征维度
        :param num_heads: 多头自注意力的头数
        :param block_layers: 自注意力层（带RoPE）
        :param depth: 注意力块层数
        :param rope_theta:
        """
        super(rope_2d_vit, self).__init__()
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
            freqs.append(init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta))
        freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

        self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)  # 提前为compute_mixed_cis函数设置num_heads参数，在调用时就不用传入该参数了

        self.norm = norm_layer(embed_dim)

    def forward(self, x, coor_xy):
        """
        :param x: 节点特征 [B, N, D]
        :param coor_xy: 每个节点的二维坐标 [B, N, 2]
        :return:
        """
        assert x.shape[0] == coor_xy.shape[0]
        B, N, D = x.shape  # batch size, number of nodes, dimension of features

        # 嵌入分类头
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x_b_list = []
        for b in range(B):
            t_x, t_y = coor_xy[b, :, 0], coor_xy[b, :, 1]
            # time1 = time.time()
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            # time2 = time.time()

            x_b = x[b, :, :].unsqueeze(0)
            for i, blk in enumerate(self.blocks):
                x_b = blk(x_b, freqs_cis=freqs_cis[i])
            # time3 = time.time()
            x_b_list.append(x_b)

            # print('freqs_cis_time:', time2 - time1, 'blk_time:', time3 - time2)

        x_b_list_cat = torch.cat(x_b_list, dim=0)
        x_b_list_cat = self.norm(x_b_list_cat)
        # x_b_list_cat = x_b_list_cat[:, 0]

        return x_b_list_cat

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = rope_2d_vit(block_layers=RoPE_Layer_scale_init_Block, rope_theta=10.0, embed_dim=512, num_heads=16, depth=24, mlp_ratio=4.,
                 qkv_bias=True, drop_path_rate=0., qk_scale=None, attn_drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, Attention_block=RoPEAttention, Mlp_block=Mlp, init_scale=1e-4)
    model.to(device)
    x = torch.randn(2, 17, 512).to(device)
    coor_xy = torch.randn(2, 17, 2).to(device)
    y = model(x, coor_xy)
    print(y.shape)

    # 利用calculate_flops计算
    # from calflops import calculate_flops
    # FLOPs, MACs, params = calculate_flops(model.cuda(), input_shape=(1, 17, 1024), output_as_string=False)
    # print(params, FLOPs)