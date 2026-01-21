from torch import nn
from typing import Tuple, Union
from lsf3d.network_architecture.neural_network import SegmentationNetwork

import torch
import math
import copy
from torch import nn
from einops import rearrange
from functools import partial
import datetime
import torch.nn.functional as F
import time

class LiteSegFormer3D(SegmentationNetwork):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [4, 2, 1, 1],
        embed_dims: list = [32, 64, 160, 256],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [4, 4, 4, 4],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 16,
        decoder_dropout: float = 0.0,
        do_ds: bool = True,
    ):
        """
        in_channels: 输入通道数
        img_volume_dim: 图像体积的空间分辨率（深度、宽度、高度）
        sr_ratios: 用于下采样嵌入补丁序列长度的比率
        embed_dims: PatchEmbedded输入的隐藏大小
        patch_kernel_size: 补丁嵌入模块中卷积的核大小
        patch_stride: 补丁嵌入模块中卷积的步幅
        patch_padding: 补丁嵌入模块中卷积的填充
        mlp_ratios: MLP中隐藏状态投影维度增加的比率
        num_heads: 注意力头的数量
        depths: 注意力层的数量
        decoder_head_embedding_dim: all-mlp-decoder模块中MLP层的投影维度
        num_classes: 网络输出通道的数量
        decoder_dropout: 拼接特征图的丢弃率
        do_ds: 是否进行深度监督

        """
        super().__init__()
        self.do_ds = do_ds
        self.num_classes = num_classes
        self.litesegformer_encoder = LiteKANformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
        )
        # decoder takes in the feature maps in the reversed order
        reversed_embed_dims = embed_dims[::-1]
        self.litesegformer_decoder = DecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, DynamicTanh):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        # embedding the input
        input_shape = x.shape[2:]  # (D, H, W)
        features = self.litesegformer_encoder(x)
        c1, c2, c3, c4 = features

        # Get main output and intermediate features
        x = self.litesegformer_decoder(c1, c2, c3, c4)
        
        if self.do_ds:
            # Get deep supervision outputs
            _c2 = self.litesegformer_decoder.linear_c2(c2).permute(0, 2, 1).reshape(x.shape[0], -1, c2.shape[2], c2.shape[3], c2.shape[4]).contiguous()
            _c3 = self.litesegformer_decoder.linear_c3(c3).permute(0, 2, 1).reshape(x.shape[0], -1, c3.shape[2], c3.shape[3], c3.shape[4]).contiguous()
            
            outputs = self.litesegformer_decoder.get_ds_outputs(x, _c2, _c3)
            return outputs
        else:
            return x
    
# ----------------------------------------------------- encoder -----------------------------------------------------

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 4,
        embed_dim: int = 768,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        """
        in_channels: 输入体积的通道数
        embed_dim: 补丁的嵌入维度
        """
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = DynamicTanh(normalized_shape=embed_dim, alpha_init_value=True)

    def forward(self, x):
        # x: (B, C, D, H, W)
        patches = self.patch_embeddings(x)
        # patches: (B, embed_dim, D', H', W')
        spatial_shape = patches.shape[2:]  # (D', H', W')
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        patches = self.norm(patches)
        return patches, spatial_shape

class FastKANLayer_Att(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = embed_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = DynamicTanh(normalized_shape=input_dim, alpha_init_value=True)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, embed_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, embed_dim)

    def forward(self, x, use_layernorm=False): #TODO
        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 确保输入张量是连续的
        x = x.contiguous()
        
        # 展平为二维张量
        x = x.view(-1, self.input_dim)  # (batch_size * seq_len * height, input_dim)

        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        
        # 计算 spline_linear 的输出
        ret = self.spline_linear(spline_basis.view(spline_basis.size(0), -1))  # (batch_size * seq_len * height, embed_dim)
        
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base

        # 恢复原始形状
        ret = ret.view(original_shape[0], original_shape[1], original_shape[2], -1)  # (batch_size, seq_len, height, embed_dim)
        return ret

class AGBLA(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        embed_dim: PatchEmbedded输入的隐藏大小
        num_heads: 注意力头的数量
        sr_ratio: 用于下采样嵌入补丁序列长度的比率
        qkv_bias: 线性投影是否具有偏置
        attn_dropout: 注意力组件的丢弃率
        proj_dropout: 最终线性投影的丢弃率
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        # embedding dimesion of each attention head
        self.attention_head_dim = embed_dim // num_heads

        # The same input is used to generate the query, key, and value,
        # (batch_size, num_patches, hidden_size) -> (batch_size, num_patches, attention_head_size)
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = DynamicTanh(normalized_shape=embed_dim, alpha_init_value=True)
        self.fkan = FastKANLayer_Att(input_dim=embed_dim, embed_dim=embed_dim)

    def forward(self, x, spatial_shape):
        # (batch_size, num_patches, hidden_size)
        B, N, C = x.shape
        D, H, W = spatial_shape

        # (batch_size, num_head, sequence_length, embed_dim)
        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            # (batch_size, sequence_length, embed_dim) -> (batch_size, embed_dim, D, H, W)
            x_ = x.permute(0, 2, 1).reshape(B, C, D, H, W)
            # (batch_size, embed_dim, D, H, W) -> (batch_size, embed_dim, D/sr, H/sr, W/sr)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).permute(0, 2, 1)
            # normalizing the layer
            x_ = self.sr_norm(x_)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]
        q = self.fkan(q)
        k = self.fkan(k)
        v = self.fkan(v)

        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.num_heads)
        attnention_prob = attention_score.softmax(dim=-1)
        attnention_prob = self.attn_dropout(attnention_prob)
        out = (attnention_prob @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out

class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        # 初始化可学习参数
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 输入形状: (batch, patch_cube, hidden_size)
        x = torch.tanh(self.alpha * x)  # 应用动态 Tanh 激活
        x = x * self.weight + self.bias  # 应用缩放和偏移
        return x

INNER_DIM = 64

class ANS3DOp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(channels)
        self.act2 = nn.GELU()
    def forward(self, x):
        # 输入形状: (batch_size, channels, depth, height, width)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x

class ANSFFN(nn.Module):
    def __init__(self, in_dim, factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, INNER_DIM)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(INNER_DIM, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = ANS3DOp(INNER_DIM)

        self.norm = nn.LayerNorm(in_dim)
        # self.norm = DynamicTanh(normalized_shape=in_dim, alpha_init_value=True)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, spatial_shape=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)


        B, N, C = project1.shape
        assert spatial_shape is not None, "spatial_shape must be provided for ANSFFN!"
        D, H, W = spatial_shape
        project1 = project1.reshape(B, D, H, W, C)
        project1 = project1.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        project1 = project1.reshape(B, N, C)  # (B, N, C)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        # mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        embed_dim: PatchEmbedded输入的隐藏大小
        mlp_ratio: 在_MLP组件中嵌入补丁的投影维度增加的比率
        num_heads: 注意力头的数量
        sr_ratio: 用于下采样嵌入补丁序列长度的比率
        qkv_bias: 线性投影是否具有偏置
        attn_dropout: 注意力组件的丢弃率
        proj_dropout: 最终线性投影的丢弃率
        """
        super().__init__()
        # 使用 DynamicTanh 替换 LayerNorm
        self.norm1 = DynamicTanh(normalized_shape=embed_dim, alpha_init_value=True)
        self.agbla = AGBLA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
        )
        self.norm2 = DynamicTanh(normalized_shape=embed_dim, alpha_init_value=True)
        self.act_fn = nn.GELU()
        self.ansffn = ANSFFN(embed_dim, 8)

    def forward(self, x, spatial_shape):
        # 输入形状: (batch, num_patches, hidden_size)
        # spatial_shape: (D, H, W)
        x = x + self.agbla(self.norm1(x), spatial_shape=spatial_shape)  # 第一个残差连接
        x = x + self.act_fn(self.norm2(x))    # 第二个残差连接
        x = self.ansffn(x, spatial_shape=spatial_shape)
        return x


class LiteKANformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [8, 4, 2, 1],
        embed_dims: list = [64, 128, 320, 512],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
    ):
        """
        in_channels: 输入通道数
        img_volume_dim: 图像体积的空间分辨率（深度、宽度、高度）
        sr_ratios: 用于下采样嵌入补丁序列长度的比率
        embed_dims: PatchEmbedded输入的隐藏大小
        patch_kernel_size: 补丁嵌入模块中卷积的核大小
        patch_stride: 补丁嵌入模块中卷积的步幅
        patch_padding: 补丁嵌入模块中卷积的填充
        mlp_ratio: 在MLP中隐藏状态投影维度增加的比率
        num_heads: 注意力头的数量
        depth: 注意力层的数量
        """
        super().__init__()

        # patch embedding at different Pyramid level
        self.embed_1 = PatchEmbedding(
            in_channel=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0],
            stride=patch_stride[0],
            padding=patch_padding[0],
        )
        self.embed_2 = PatchEmbedding(
            in_channel=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1],
            stride=patch_stride[1],
            padding=patch_padding[1],
        )
        self.embed_3 = PatchEmbedding(
            in_channel=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2],
            stride=patch_stride[2],
            padding=patch_padding[2],
        )
        self.embed_4 = PatchEmbedding(
            in_channel=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3],
            stride=patch_stride[3],
            padding=patch_padding[3],
        )

        # block 1
        self.tf_block1 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[0],
                    num_heads=num_heads[0],
                    # mlp_ratio=mlp_ratios[0],
                    sr_ratio=sr_ratios[0],
                    qkv_bias=True,
                )
                for _ in range(depths[0])
            ]
        )
        self.norm1 = DynamicTanh(normalized_shape=embed_dims[0], alpha_init_value=True)

        # block 2
        self.tf_block2 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[1],
                    num_heads=num_heads[1],
                    # mlp_ratio=mlp_ratios[1],
                    sr_ratio=sr_ratios[1],
                    qkv_bias=True,
                )
                for _ in range(depths[1])
            ]
        )
        self.norm2 = DynamicTanh(normalized_shape=embed_dims[1], alpha_init_value=True)

        # block 3
        self.tf_block3 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[2],
                    num_heads=num_heads[2],
                    # mlp_ratio=mlp_ratios[2],
                    sr_ratio=sr_ratios[2],
                    qkv_bias=True,
                )
                for _ in range(depths[2])
            ]
        )
        self.norm3 = DynamicTanh(normalized_shape=embed_dims[2], alpha_init_value=True)

        # block 4
        self.tf_block4 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[3],
                    num_heads=num_heads[3],
                    # mlp_ratio=mlp_ratios[3],
                    sr_ratio=sr_ratios[3],
                    qkv_bias=True,
                )
                for _ in range(depths[3])
            ]
        )
        self.norm4 = DynamicTanh(normalized_shape=embed_dims[3], alpha_init_value=True)

    def forward(self, x):
        out = []
        # stage 1
        x, shape1 = self.embed_1(x)  # (B, N, C), (D, H, W)
        B, N, C = x.shape
        for i, blk in enumerate(self.tf_block1):
            x = blk(x, spatial_shape=shape1)
        x = self.norm1(x)
        x_1 = x.reshape(B, *shape1, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x_1)

        # stage 2
        x, shape2 = self.embed_2(x_1)
        B, N, C = x.shape
        for i, blk in enumerate(self.tf_block2):
            x = blk(x, spatial_shape=shape2)
        x = self.norm2(x)
        x_2 = x.reshape(B, *shape2, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x_2)

        # stage 3
        x, shape3 = self.embed_3(x_2)
        B, N, C = x.shape
        for i, blk in enumerate(self.tf_block3):
            x = blk(x, spatial_shape=shape3)
        x = self.norm3(x)
        x_3 = x.reshape(B, *shape3, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x_3)

        # stage 4
        x, shape4 = self.embed_4(x_3)
        B, N, C = x.shape
        for i, blk in enumerate(self.tf_block4):
            x = blk(x, spatial_shape=shape4)
        x = self.norm4(x)
        x_4 = x.reshape(B, *shape4, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x_4)

        return out

###################################################################################
    # 移除 cube_root，所有空间 shape 均由真实 shape 传递
    

###################################################################################
# ----------------------------------------------------- decoder -------------------
class MLP_(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = DynamicTanh(normalized_shape=embed_dim, alpha_init_value=True)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        return x

class SplineLinear(nn.Linear):
    """
        截断正太分布避免极端权重值，保持数值稳定，截断过大或过小的权重值
        SplineLinear继承Linear, 通过可学习权重对核空间特征进行线性组合, 实现非线性到线性的平滑过渡
    """
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale # 标准差
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids) # 生成等距点
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1) # 影响半径

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

###################################################################################

class ADCNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3x3 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv1x3x1 = nn.Conv3d(in_channels, out_channels, (1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv3x1x1 = nn.Conv3d(in_channels, out_channels, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        out = self.conv3x3x3(x) + self.conv1x3x1(x) + self.conv3x1x1(x)
        out = self.bn(out)
        out = self.act(out)
        return out

class DecoderHead(nn.Module):

    def __init__(
        self,
        input_feature_dims: list = [512, 320, 128, 64],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.0,
    ):
        """
        input_feature_dims: 由transformer编码器生成的输出特征通道列表
        decoder_head_embedding_dim: 全MLP解码器模块中MLP层的投影维度
        num_classes: 输出通道数
        dropout: 连接的特征图的丢弃率
        """
        super().__init__()
        self.linear_c4 = MLP_(
            input_dim=input_feature_dims[0],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c3 = MLP_(
            input_dim=input_feature_dims[1],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c2 = MLP_(
            input_dim=input_feature_dims[2],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c1 = MLP_(
            input_dim=input_feature_dims[3],
            embed_dim=decoder_head_embedding_dim,
        )
        self.act_c2 = ADCNet3D(
            input_feature_dims[2], 
            input_feature_dims[2])
        self.act_c1 = ADCNet3D(
            input_feature_dims[3], 
            input_feature_dims[3])
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        
        # Main prediction head
        self.linear_pred = nn.Conv3d(
            decoder_head_embedding_dim, num_classes, kernel_size=1
        )
        
        # Deep supervision prediction heads
        self.ds_pred_1 = nn.Conv3d(decoder_head_embedding_dim, num_classes, kernel_size=1)
        self.ds_pred_2 = nn.Conv3d(decoder_head_embedding_dim, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        n, _, _, _, _ = c4.shape
        
        # Process c4
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4]).contiguous()
        _c4 = torch.nn.functional.interpolate(_c4, size=c1.size()[2:], mode="trilinear", align_corners=False)

        # Process c3
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4]).contiguous()
        _c3 = torch.nn.functional.interpolate(_c3, size=c1.size()[2:], mode="trilinear", align_corners=False)

        # Process c2
        _c2 = self.linear_c2(self.act_c2(c2)).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4]).contiguous()
        _c2 = torch.nn.functional.interpolate(_c2, size=c1.size()[2:], mode="trilinear", align_corners=False)

        # Process c1
        _c1 = self.linear_c1(self.act_c1(c1)).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4]).contiguous()

        # Fuse features
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        
        # Main output
        x = self.linear_pred(x)
        
        # 确保输出尺寸与目标尺寸匹配 (16, 160, 160)
        x = torch.nn.functional.interpolate(x, size=(16, 160, 160), mode="trilinear", align_corners=False)
        
        return x

    def get_ds_outputs(self, x, _c2, _c3):
        """Get deep supervision outputs from different decoder stages"""
        # Main output (already interpolated to input size)
        x1 = x
        
        # Second scale output
        x2 = self.ds_pred_1(_c3)
        x2 = torch.nn.functional.interpolate(x2, size=(16, 160, 160), mode="trilinear", align_corners=False)
        
        # Third scale output
        x3 = self.ds_pred_2(_c2)
        x3 = torch.nn.functional.interpolate(x3, size=(16, 160, 160), mode="trilinear", align_corners=False)
        
        return [x1, x2, x3]
