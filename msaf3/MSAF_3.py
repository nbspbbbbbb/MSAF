# -*- coding: utf-8 -*-
"""
三模态版 MSAF 模型定义。

这个版本遵循“最小改动接入三模态”的思路：
1. 保留原始 HSI / LiDAR 的双分支和两级融合结构；
2. 新增一个独立 RGB 分支；
3. 将融合单元从三视角扩展为四视角：
   HSI 空间视角 + HSI 光谱视角 + LiDAR 空间视角 + RGB 空间视角；
4. Transformer 输出后重组为 HSI / LiDAR / RGB 三路特征图；
5. 最终仍保持原始 MSAF 的 sum fusion 风格，再接分类头。
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


def _build_mobilenet_v3_small(use_pretrained=True):
    if use_pretrained:
        try:
            return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except AttributeError:
            return models.mobilenet_v3_small(pretrained=True)

    try:
        return models.mobilenet_v3_small(weights=None)
    except TypeError:
        return models.mobilenet_v3_small(pretrained=False)


class ModalityCNN(nn.Module):
    """单模态轻量卷积分支，对应论文中的 MCB 分支骨干。"""

    def __init__(self, in_channels, use_pretrained=True, keep_original_stem=False):
        super().__init__()
        self._model = _build_mobilenet_v3_small(use_pretrained=use_pretrained)

        if keep_original_stem and in_channels == 3:
            return

        stem_conv = self._model.features[0][0]
        inv_conv = self._model.features[1].block[0][0]
        use_bias = stem_conv.bias is not None

        self._model.features[0][0] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=stem_conv.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        self._model.features[1].block[0][0] = nn.Conv2d(
            in_channels=inv_conv.in_channels,
            out_channels=inv_conv.out_channels,
            kernel_size=inv_conv.kernel_size,
            stride=1,
            padding=inv_conv.padding,
            groups=inv_conv.groups,
            bias=use_bias,
        )

        torch.cuda.empty_cache()
        del stem_conv, inv_conv


class HSI_CNN(ModalityCNN):
    """HSI 分支。"""

    def __init__(self, in_channels, use_pretrained=True):
        super().__init__(in_channels=in_channels, use_pretrained=use_pretrained)


class Lidar_CNN(ModalityCNN):
    """LiDAR 分支。"""

    def __init__(self, in_channels, use_pretrained=False):
        super().__init__(in_channels=in_channels, use_pretrained=use_pretrained)


class RGB_CNN(ModalityCNN):
    """RGB 分支。"""

    def __init__(self, in_channels=3, use_pretrained=True):
        # 为了和 HSI / LiDAR 分支保持相同空间尺度，RGB 分支也沿用
        # 原始 MSAF 中 stride=1 的轻量 stem，而不是 MobileNet 默认下采样 stem。
        super().__init__(in_channels=in_channels, use_pretrained=use_pretrained)


class SelfAttention(nn.Module):
    """掩码自注意力，对应论文中的 MSAM。"""

    def __init__(self, n_embd, n_head, dim_head, attn_pdrop, resid_pdrop, mask_ratio=0.1):
        super().__init__()
        inner_dim = dim_head * n_head
        project_out = not (n_head == 1 and dim_head == n_embd)

        self.n_head = n_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.mask_ratio = mask_ratio

        self.key = nn.Linear(n_embd, inner_dim, bias=False)
        self.query = nn.Linear(n_embd, inner_dim, bias=False)
        self.value = nn.Linear(n_embd, inner_dim, bias=False)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = (
            nn.Sequential(nn.Linear(inner_dim, n_embd), self.resid_drop)
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        k = self.key(x).view(batch_size, seq_len, self.n_head, self.dim_head).transpose(1, 2)
        q = self.query(x).view(batch_size, seq_len, self.n_head, self.dim_head).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_head, self.dim_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale

        # 论文 Sec. II-D：在 softmax 前对注意力强度矩阵施加随机掩码。
        mask_prob = torch.full_like(att, self.mask_ratio)
        att = att + torch.bernoulli(mask_prob) * -1e12

        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.dim_head)
        return self.proj(y)


class Block(nn.Module):
    """标准 Transformer 块：LN -> MSAM -> 残差，再 LN -> MLP -> 残差。"""

    def __init__(self, n_embd, n_head, dim_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, dim_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MVT3(nn.Module):
    """
    三模态版 Transformer 融合单元。

    四视角 token 的拼接顺序固定为：
    1. HSI 空间视角
    2. HSI 光谱视角
    3. LiDAR 空间视角
    4. RGB 空间视角
    """

    def __init__(
        self,
        n_embd,
        n_head,
        dim_head,
        block_exp,
        n_layer,
        block_w,
        block_h,
        embd_pdrop,
        attn_pdrop,
        resid_pdrop,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.block_w = block_w
        self.block_h = block_h
        self.num_spatial_views = 3

        # 将 HSI 光谱视角中的 k_t * k_t 维 patch token 投影回统一通道维。
        self.inner_proj = nn.Linear(block_w * block_h, n_embd)

        total_tokens = self.num_spatial_views * block_w * block_h + n_embd
        self.pos_emb = nn.Parameter(torch.zeros(1, total_tokens, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, dim_head, block_exp, attn_pdrop, resid_pdrop)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hsi_tensor, lidar_tensor, rgb_tensor):
        batch_size = hsi_tensor.shape[0]
        height = self.block_h
        width = self.block_w
        spatial_tokens = height * width

        hsi_tensor = hsi_tensor.view(batch_size, -1, height, width)
        lidar_tensor = lidar_tensor.view(batch_size, -1, height, width)
        rgb_tensor = rgb_tensor.view(batch_size, -1, height, width)

        # HSI 空间视角 token，形状为 (k_t * k_t, C)。
        hsi_flat = hsi_tensor.view(batch_size, -1, spatial_tokens)
        hsi_spatial = hsi_flat.permute(0, 2, 1).contiguous()

        # HSI 光谱视角 token，形状为 (C, C)。
        hsi_spectral = self.inner_proj(hsi_flat)

        # LiDAR 和 RGB 只保留空间视角。
        lidar_spatial = lidar_tensor.view(batch_size, -1, spatial_tokens).permute(0, 2, 1).contiguous()
        rgb_spatial = rgb_tensor.view(batch_size, -1, spatial_tokens).permute(0, 2, 1).contiguous()

        # 四视角拼接：HSI空间 + HSI光谱 + LiDAR空间 + RGB空间。
        token_embeddings = torch.cat(
            [hsi_spatial, hsi_spectral, lidar_spatial, rgb_spatial],
            dim=1,
        )

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x.permute(0, 2, 1).contiguous()

        # 只重建三路空间特征图；HSI 光谱视角只参与融合，不单独恢复。
        hsi_start = 0
        hsi_end = spatial_tokens
        lidar_start = hsi_end + self.n_embd
        lidar_end = lidar_start + spatial_tokens
        rgb_start = lidar_end
        rgb_end = rgb_start + spatial_tokens

        hsi_out = x[:, :, hsi_start:hsi_end].contiguous().view(batch_size, -1, height, width)
        lidar_out = x[:, :, lidar_start:lidar_end].contiguous().view(batch_size, -1, height, width)
        rgb_out = x[:, :, rgb_start:rgb_end].contiguous().view(batch_size, -1, height, width)
        return hsi_out, lidar_out, rgb_out


class MCF3(nn.Module):
    """
    三模态版 MSAF 主体。

    结构上尽量保持原始双模态实现的风格：
    - 三个独立卷积分支；
    - 两级级联 MVSE + MSAM 融合；
    - 最后仍然先堆叠三路特征，再做 sum fusion 和分类。
    """

    def __init__(
        self,
        HSIband,
        lidarband,
        rgbband,
        num_classes,
        use_pretrained=True,
        use_rgb_pretrained=True,
    ):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((5, 5))

        self.image_encoder = HSI_CNN(HSIband, use_pretrained=use_pretrained)
        self.lidar_encoder = Lidar_CNN(lidarband, use_pretrained=False)
        self.rgb_encoder = RGB_CNN(rgbband, use_pretrained=use_rgb_pretrained)

        self.mlp_head = nn.Linear(24, num_classes)
        torch.nn.init.xavier_uniform_(self.mlp_head.weight)
        torch.nn.init.normal_(self.mlp_head.bias, std=1e-6)

        self.transformer1 = MVT3(
            n_embd=16,
            n_head=4,
            dim_head=16,
            block_exp=4,
            n_layer=2,
            block_w=5,
            block_h=5,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )
        self.transformer2 = MVT3(
            n_embd=24,
            n_head=4,
            dim_head=24,
            block_exp=4,
            n_layer=2,
            block_w=5,
            block_h=5,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def forward(self, hsi, lidar, rgb):
        batch_size, _, height, width = hsi.shape

        # 第一阶段 MCB1。
        hsi_stage0 = self.image_encoder._model.features[0](hsi)
        hsi_stage1 = self.image_encoder._model.features[1](hsi_stage0)

        lidar_stage0 = self.lidar_encoder._model.features[0](lidar)
        lidar_stage1 = self.lidar_encoder._model.features[1](lidar_stage0)

        rgb_stage0 = self.rgb_encoder._model.features[0](rgb)
        rgb_stage1 = self.rgb_encoder._model.features[1](rgb_stage0)

        # 第一阶段 MVSE + MSAM 融合。
        hsi_pool1 = self.avgpool(hsi_stage1)
        lidar_pool1 = self.avgpool(lidar_stage1)
        rgb_pool1 = self.avgpool(rgb_stage1)

        hsi_fuse1, lidar_fuse1, rgb_fuse1 = self.transformer1(hsi_pool1, lidar_pool1, rgb_pool1)
        hsi_fuse1 = F.interpolate(hsi_fuse1, size=[height, width], mode="bilinear")
        lidar_fuse1 = F.interpolate(lidar_fuse1, size=[height, width], mode="bilinear")
        rgb_fuse1 = F.interpolate(rgb_fuse1, size=[height, width], mode="bilinear")

        hsi_stage1 = hsi_stage1 + hsi_fuse1
        lidar_stage1 = lidar_stage1 + lidar_fuse1
        rgb_stage1 = rgb_stage1 + rgb_fuse1

        # 第二阶段 MCB2。
        hsi_stage2 = self.image_encoder._model.features[2](hsi_stage1)
        lidar_stage2 = self.lidar_encoder._model.features[2](lidar_stage1)
        rgb_stage2 = self.rgb_encoder._model.features[2](rgb_stage1)

        # 第二阶段 MVSE + MSAM 融合。
        hsi_pool2 = self.avgpool_2(hsi_stage2)
        lidar_pool2 = self.avgpool_2(lidar_stage2)
        rgb_pool2 = self.avgpool_2(rgb_stage2)

        hsi_fuse2, lidar_fuse2, rgb_fuse2 = self.transformer2(hsi_pool2, lidar_pool2, rgb_pool2)
        hsi_stage2 = hsi_pool2 + hsi_fuse2
        lidar_stage2 = lidar_pool2 + lidar_fuse2
        rgb_stage2 = rgb_pool2 + rgb_fuse2

        # 最后保持原始 MSAF 风格：三路特征先 sum fusion，再送分类头。
        hsi_feat = self.image_encoder._model.avgpool(hsi_stage2)
        hsi_feat = torch.flatten(hsi_feat, 1).view(batch_size, 1, -1)

        lidar_feat = self.lidar_encoder._model.avgpool(lidar_stage2)
        lidar_feat = torch.flatten(lidar_feat, 1).view(batch_size, 1, -1)

        rgb_feat = self.rgb_encoder._model.avgpool(rgb_stage2)
        rgb_feat = torch.flatten(rgb_feat, 1).view(batch_size, 1, -1)

        fused_features = torch.cat([hsi_feat, lidar_feat, rgb_feat], dim=1)
        fused_features = torch.sum(fused_features, dim=1)
        fused_features = fused_features.view(batch_size, -1)
        return self.mlp_head(fused_features)


MSAF3 = MCF3
