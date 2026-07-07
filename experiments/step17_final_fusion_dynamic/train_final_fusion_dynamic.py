# -*- coding: utf-8 -*-
"""MSAF3 with q-guided CoRiM/FW weights applied at final feature fusion.

【中文说明：这是当前采用的最终融合动态加权版本】
本文件继承Step15已经实现的单模态头、可靠度q、CoRiM/FW和损失函数，
但重写forward，把权重从Transformer1输入处移到两个Transformer之后、
三条最终分支求和之前。这样既保留原MSAF3两阶段跨模态交互，又让
样本级可靠性权重直接控制最终分支贡献。

This is a separate experiment from step15.  The unimodal auxiliary heads,
correctness predictor q, CoRiM/FW solver and training losses are reused
unchanged.  Both Transformers receive the original unweighted modality
features.  The per-sample weights are applied only when the three final
modality branches are summed.

With w=(1/3, 1/3, 1/3), ``3 * sum_m(w_m * F_m)`` is exactly the original
MSAF3 feature sum ``sum_m(F_m)``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
STEP15_PATH = (
    HERE.parent
    / "step15_warmup_dynamic_300"
    / "train_warmup_then_dynamic.py"
)


def load_module(name: str, path: Path):
    """加载Step15模块，以复用已验证的q、FW、损失和基础模型代码。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INPUT_DYNAMIC = load_module("step15_input_dynamic", STEP15_PATH)
# BASE提供PureAuxMSAF3和数据/设备辅助接口；compute_loss保持两版损失一致，
# 因而“权重位置”是Step15与Step17之间的主要实验变量。
BASE = INPUT_DYNAMIC.BASE
compute_loss = INPUT_DYNAMIC.compute_loss
q_guided_fw_weights = INPUT_DYNAMIC.q_guided_fw_weights


class FinalFusionDynamicMSAF3(INPUT_DYNAMIC.WarmupDynamicMSAF3):
    """最终版模型：在三条最终模态分支求和时应用动态可靠性权重。"""

    def forward(self, hsi, lidar, rgb):
        """完成一次前向传播。

        逻辑分成两条并行路线：
        A. 浅层纯单模态特征 -> 辅助预测 -> q与FW -> 得到样本权重w；
        B. 三模态特征 -> 原始Transformer1/2 -> 得到三条最终分支。
        最后用路线A的w对路线B的三条分支加权，再交给融合分类头。
        """
        batch_size, _, height, width = hsi.shape
        # ===== 1. 三个CNN各自提取Transformer1之前的浅层特征 =====
        hsi_stage0 = self.image_encoder._model.features[0](hsi)
        hsi_stage1_raw = self.image_encoder._model.features[1](hsi_stage0)
        lidar_stage0 = self.lidar_encoder._model.features[0](lidar)
        lidar_stage1_raw = self.lidar_encoder._model.features[1](lidar_stage0)
        rgb_stage0 = self.rgb_encoder._model.features[0](rgb)
        rgb_stage1_raw = self.rgb_encoder._model.features[1](rgb_stage0)

        # ===== 2. 真正的融合前单模态预测 =====
        # 此处三条特征还没有经过任何跨模态Transformer，因此可用于衡量
        # 原始模态冲突和正确性，而不是“融合后分支之间的差异”。
        pure_features = torch.stack(
            [
                self._pool_feature(hsi_stage1_raw),
                self._pool_feature(lidar_stage1_raw),
                self._pool_feature(rgb_stage1_raw),
            ],
            dim=1,
        )
        aux_logits = torch.stack(
            [
                self.hsi_aux_classifier(pure_features[:, 0]),
                self.lidar_aux_classifier(pure_features[:, 1]),
                self.rgb_aux_classifier(pure_features[:, 2]),
            ],
            dim=1,
        )
        probabilities = F.softmax(aux_logits, dim=2)
        # probabilities形状[B,3,C]；q形状[B,3]，表示各模态预测正确概率。
        q = self.reliability(probabilities)
        if self.dynamic_enabled:
            # warm-up结束后，对每个样本独立执行FW，得到和为1的三模态权重。
            weights = q_guided_fw_weights(
                probabilities,
                q,
                self.fw_alpha,
                self.fw_beta,
                self.fw_gamma,
                self.fw_iterations,
                self.q_balance_lambda,
            )
        else:
            # warm-up阶段固定等权；此时最终公式严格退化为原MSAF3求和。
            weights = torch.full(
                (batch_size, 3),
                1.0 / 3.0,
                dtype=probabilities.dtype,
                device=probabilities.device,
            )

        # ===== 3. 原始MSAF3的两阶段Transformer融合主路 =====
        # 与Step15不同，这里Transformer1输入不乘动态权重。
        hsi_pool1 = self.avgpool(hsi_stage1_raw)
        lidar_pool1 = self.avgpool(lidar_stage1_raw)
        rgb_pool1 = self.avgpool(rgb_stage1_raw)
        hsi_fuse1, lidar_fuse1, rgb_fuse1 = self.transformer1(
            hsi_pool1, lidar_pool1, rgb_pool1
        )
        hsi_stage1 = hsi_stage1_raw + F.interpolate(
            hsi_fuse1, size=[height, width], mode="bilinear"
        )
        lidar_stage1 = lidar_stage1_raw + F.interpolate(
            lidar_fuse1, size=[height, width], mode="bilinear"
        )
        rgb_stage1 = rgb_stage1_raw + F.interpolate(
            rgb_fuse1, size=[height, width], mode="bilinear"
        )

        hsi_stage2 = self.image_encoder._model.features[2](hsi_stage1)
        # 第二阶段先继续CNN编码，再进行更深层的跨模态Transformer交互。
        lidar_stage2 = self.lidar_encoder._model.features[2](lidar_stage1)
        rgb_stage2 = self.rgb_encoder._model.features[2](rgb_stage1)
        hsi_pool2 = self.avgpool_2(hsi_stage2)
        lidar_pool2 = self.avgpool_2(lidar_stage2)
        rgb_pool2 = self.avgpool_2(rgb_stage2)
        hsi_fuse2, lidar_fuse2, rgb_fuse2 = self.transformer2(
            hsi_pool2, lidar_pool2, rgb_pool2
        )
        final_features = torch.stack(
            [
                self._pool_feature(hsi_pool2 + hsi_fuse2),
                self._pool_feature(lidar_pool2 + lidar_fuse2),
                self._pool_feature(rgb_pool2 + rgb_fuse2),
            ],
            dim=1,
        )

        # ===== 4. 本版本真正发生动态融合的位置 =====
        # final_features形状[B,3,D]。einsum等价于逐模态乘w后求和。
        # 乘3用于保持原始特征尺度：w=(1/3,1/3,1/3)时，
        # 3*sum(w_m*F_m) = F_H + F_L + F_R，正好等于原MSAF3。
        fused_feature = 3.0 * torch.einsum("bm,bmd->bd", weights, final_features)
        # 返回字典既包含最终预测，也保留辅助预测、q和权重，供损失与诊断使用。
        return {
            "logits_fusion": self.fusion_classifier(fused_feature),
            "logits_hsi_aux": aux_logits[:, 0],
            "logits_lidar_aux": aux_logits[:, 1],
            "logits_rgb_aux": aux_logits[:, 2],
            "aux_probabilities": probabilities,
            "q": q,
            "weights": weights,
        }
