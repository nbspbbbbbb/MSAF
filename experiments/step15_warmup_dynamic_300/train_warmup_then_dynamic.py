# -*- coding: utf-8 -*-
"""Warm up equal MSAF3, then train q-guided dynamic Transformer-1 fusion.

【中文说明：这是保留用于消融实验的旧版本】
本文件把动态权重乘在第一个 Transformer 的输入上，而不是最终求和处。
它同时集中实现了后续最终版本复用的几个核心模块：
1. 三个融合前单模态辅助分类头；
2. 基于预测统计量的正确性可靠度 q；
3. CoRiM 风险梯度与 Frank-Wolfe（FW）权重迭代；
4. 高冲突、高置信错误校准损失（formula16_loss）；
5. warm-up、断点续训和 checkpoint 保存。

Schedule
--------
1. Train through epoch 50 with dynamic fusion disabled.
2. Keep exact resumable checkpoints at epochs 30 and 50.
3. If epochs 31-50 improve validation OA by < 0.1 percentage point over
   epochs 1-30, switch from epoch 30; otherwise switch from epoch 50.
4. Resume that exact model/optimizer/scheduler state, enable q-guided
   gradient-balanced CoRiM/FW only on Transformer-1 inputs, and train until
   the total epoch number reaches 300.

The original residual branches, Transformer 2 and final equal feature sum are
unchanged.  The script is resumable and does not edit msaf3 source files.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
STEP4_PATH = (
    EXPERIMENTS
    / "step4_correctness_calibration"
    / "run_formula16_finetune.py"
)


def load_module(name: str, path: Path):
    """按文件路径动态导入模块。

    这样各实验可以复用 Step4/Step1 的数据集和基础网络，又不必修改
    原始 msaf3 源码。name 只是本次导入使用的内部模块名。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("formula16_base", STEP4_PATH)
EPS = 1e-7
MODALITIES = 3


def parse_args() -> argparse.Namespace:
    """定义训练参数及四类损失/动态融合的超参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\DATA_3\Houston2018\prepared_msaf3"),
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--total-epochs", type=int, default=300)
    parser.add_argument("--warmup-probe-epochs", type=int, default=50)
    parser.add_argument("--warmup-candidate", type=int, default=30)
    parser.add_argument("--warmup-min-improvement", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--formula16-weight", type=float, default=0.2)
    parser.add_argument("--q-loss-weight", type=float, default=0.1)
    parser.add_argument("--fw-alpha", type=float, default=1.0)
    parser.add_argument("--fw-beta", type=float, default=0.25)
    parser.add_argument("--fw-gamma", type=float, default=64.0)
    parser.add_argument("--fw-iterations", type=int, default=10)
    parser.add_argument("--q-balance-lambda", type=float, default=0.05)
    parser.add_argument("--validation-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子，便于重复实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def stratified_limit(index_array: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """按类别等量抽样，避免验证子集被大类别主导。"""
    rng = np.random.RandomState(seed)
    selected = []
    for label in np.unique(index_array[:, 0]):
        candidates = np.flatnonzero(index_array[:, 0] == label)
        count = min(per_class, len(candidates))
        selected.append(index_array[rng.choice(candidates, size=count, replace=False)])
    result = np.concatenate(selected, axis=0)
    return result[rng.permutation(len(result))]


def normalized_js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算两个类别概率分布的归一化 JS 散度。

    JS=0 表示完全一致；除以 log(2) 后理论范围约为 [0, 1]。
    clamp_min 用于避免 log(0) 导致数值异常。
    """
    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)
    middle = 0.5 * (p + q)
    value = 0.5 * torch.sum(p * (torch.log(p) - torch.log(middle)), dim=-1)
    value += 0.5 * torch.sum(q * (torch.log(q) - torch.log(middle)), dim=-1)
    return value / math.log(2.0)


def pairwise_conflict(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """由三组单模态概率计算模态冲突和样本总体冲突。

    输入形状：[batch, 3, classes]。
    modality_conflict[:, m] 是模态 m 与另外两模态 JS 的平均值；
    global_conflict 是 HSI-LiDAR、HSI-RGB、LiDAR-RGB 三对 JS 的平均值。
    """
    js_hl = normalized_js(probabilities[:, 0], probabilities[:, 1])
    js_hr = normalized_js(probabilities[:, 0], probabilities[:, 2])
    js_lr = normalized_js(probabilities[:, 1], probabilities[:, 2])
    modality_conflict = torch.stack(
        [0.5 * (js_hl + js_hr), 0.5 * (js_hl + js_lr), 0.5 * (js_hr + js_lr)],
        dim=1,
    )
    global_conflict = (js_hl + js_hr + js_lr) / 3.0
    return modality_conflict, global_conflict


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """归一化预测熵；越大表示预测越不确定。"""
    return -torch.sum(
        probabilities.clamp_min(EPS) * torch.log(probabilities.clamp_min(EPS)), dim=-1
    ) / math.log(probabilities.shape[-1])


def prediction_margin(probabilities: torch.Tensor) -> torch.Tensor:
    """Top-1 与 Top-2 概率之差；越大通常表示模型越自信。"""
    top2 = torch.topk(probabilities, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def corim_base_gradient(
    probabilities: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """计算 CoRiM 式风险对三个模态权重的梯度。

    probabilities: 三个单模态预测概率 [B,3,C]；weights: 当前权重 [B,3]。
    alpha 控制融合预测熵，beta 控制单模态熵，gamma 控制融合与单模态
    的一致性/JS项。返回 [B,3]，每个值表示继续提高该模态权重的风险方向。
    """
    modality_count = probabilities.shape[1]
    fused = torch.einsum("bm,bmc->bc", weights, probabilities).clamp_min(EPS)
    modality_entropy = -torch.sum(
        probabilities.clamp_min(EPS) * torch.log(probabilities.clamp_min(EPS)), dim=2
    )
    fused_gradient = -torch.einsum(
        "bmc,bc->bm", probabilities, 1.0 + torch.log(fused)
    )
    middle = 0.5 * (probabilities + fused[:, None, :])
    log_ratio_sum = torch.sum(
        torch.log(fused[:, None, :] / middle.clamp_min(EPS)), dim=1
    )
    consistency_gradient = 0.5 / modality_count * torch.einsum(
        "bmc,bc->bm", probabilities, log_ratio_sum
    )
    return alpha * fused_gradient + beta * modality_entropy + gamma * consistency_gradient


@torch.no_grad()
def q_guided_fw_weights(
    probabilities: torch.Tensor,
    q: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    iterations: int,
    balance_lambda: float,
) -> torch.Tensor:
    """从等权出发，用 q 引导的 Frank-Wolfe 迭代求样本级权重。

    步骤：
    1. 每个样本初始化为 (1/3,1/3,1/3)；
    2. 把 -log(q) 作为“预测不正确”的代价，并做中心化和尺度对齐；
    3. 用总体冲突 omega 门控 q：模态一致时少干预，高冲突时多参考 q；
    4. 每轮计算 CoRiM 梯度，选择梯度最小的单纯形顶点；
    5. 将当前权重朝该顶点移动，最终仍满足非负且和为1。

    @torch.no_grad 表示FW只是当前前向过程中的显式权重求解，不通过
    FW迭代反向传播；q 预测头由后面的 q_BCE 单独监督训练。
    """
    probabilities = probabilities.detach()
    q = q.detach()
    batch, modalities, _ = probabilities.shape
    weights = torch.full(
        (batch, modalities),
        1.0 / modalities,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    base_initial = corim_base_gradient(
        probabilities, weights, alpha, beta, gamma
    )
    base_range = torch.max(base_initial, dim=1).values - torch.min(base_initial, dim=1).values
    raw_q_cost = -torch.log(q.clamp(EPS, 1.0))
    centered_q_cost = raw_q_cost - raw_q_cost.mean(dim=1, keepdim=True)
    q_range = torch.max(raw_q_cost, dim=1).values - torch.min(raw_q_cost, dim=1).values
    scale = torch.where(
        q_range >= 1e-3,
        base_range / q_range.clamp_min(1e-3),
        torch.zeros_like(q_range),
    )
    _, omega = pairwise_conflict(probabilities)
    q_direction = omega[:, None] * scale[:, None] * centered_q_cost
    step_size = 1.0 / math.sqrt(iterations)
    for _ in range(iterations):
        gradient = corim_base_gradient(probabilities, weights, alpha, beta, gamma)
        gradient = gradient + balance_lambda * q_direction
        chosen = torch.argmin(gradient, dim=1)
        vertex = F.one_hot(chosen, num_classes=modalities).to(probabilities.dtype)
        weights = (1.0 - step_size) * weights + step_size * vertex
    return weights


class WarmupDynamicMSAF3(BASE.PureAuxMSAF3):
    """旧版模型：在 Transformer1 输入处应用动态权重。"""
    def __init__(self, hsi_band, lidar_band, rgb_band, num_classes, args):
        """建立MSAF3主干、三个单模态头和三个模态专属可靠度头。"""
        super().__init__(
            hsi_band, lidar_band, rgb_band, num_classes, use_pretrained=True
        )
        self.num_classes = num_classes
        self.dynamic_enabled = False
        self.fw_alpha = args.fw_alpha
        self.fw_beta = args.fw_beta
        self.fw_gamma = args.fw_gamma
        self.fw_iterations = args.fw_iterations
        self.q_balance_lambda = args.q_balance_lambda
        reliability_dim = 5 + num_classes
        # 每个模态一个独立的单层逻辑回归头，不是多层MLP。
        # 5项统计量 + 预测类别one-hot(C维) -> 1个logit -> sigmoid得到q。
        self.reliability_heads = nn.ModuleList(
            [nn.Linear(reliability_dim, 1) for _ in range(3)]
        )
        for head in self.reliability_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def reliability(self, probabilities: torch.Tensor) -> torch.Tensor:
        """用无标签预测统计量估计每个模态“预测正确”的概率 q。

        输入特征依次为：margin、Top-1概率、熵、该模态冲突、总体冲突、
        预测类别one-hot。detach 防止 q_BCE 为了容易拟合而反向扭曲单模态
        分类概率；单模态分类头仍由辅助分类损失训练。
        """
        # The probe sees label-free statistics only.  Detaching prevents its
        # BCE loss from changing the auxiliary classifier merely to simplify q.
        detached = probabilities.detach()
        rho = prediction_margin(detached)
        top1, prediction = torch.max(detached, dim=2)
        ent = entropy(detached)
        modality_conflict, global_conflict = pairwise_conflict(detached)
        q_values = []
        for modality in range(3):
            one_hot = F.one_hot(
                prediction[:, modality], num_classes=self.num_classes
            ).to(detached.dtype)
            features = torch.cat(
                [
                    rho[:, modality : modality + 1],
                    top1[:, modality : modality + 1],
                    ent[:, modality : modality + 1],
                    modality_conflict[:, modality : modality + 1],
                    global_conflict[:, None],
                    one_hot,
                ],
                dim=1,
            )
            q_values.append(torch.sigmoid(self.reliability_heads[modality](features)))
        return torch.cat(q_values, dim=1)

    def forward(self, hsi, lidar, rgb):
        """旧版完整前向：先估计可靠性权重，再加权 Transformer1 输入。"""
        batch_size, _, height, width = hsi.shape
        hsi_stage0 = self.image_encoder._model.features[0](hsi)
        hsi_stage1_raw = self.image_encoder._model.features[1](hsi_stage0)
        lidar_stage0 = self.lidar_encoder._model.features[0](lidar)
        lidar_stage1_raw = self.lidar_encoder._model.features[1](lidar_stage0)
        rgb_stage0 = self.rgb_encoder._model.features[0](rgb)
        rgb_stage1_raw = self.rgb_encoder._model.features[1](rgb_stage0)

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
        # q 是每个样本、每个模态的正确性可靠度，形状为 [B,3]。
        q = self.reliability(probabilities)
        if self.dynamic_enabled:
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
            # warm-up阶段关闭动态机制，严格使用三模态等权。
            weights = torch.full(
                (batch_size, 3),
                1.0 / 3.0,
                dtype=probabilities.dtype,
                device=probabilities.device,
            )

        hsi_pool1 = self.avgpool(hsi_stage1_raw)
        lidar_pool1 = self.avgpool(lidar_stage1_raw)
        rgb_pool1 = self.avgpool(rgb_stage1_raw)
        # Dynamic weights gate only what each pure branch sends into T1.
        # Residual branch identities below remain unweighted.
        hsi_fuse1, lidar_fuse1, rgb_fuse1 = self.transformer1(
            3.0 * weights[:, 0, None, None, None] * hsi_pool1,
            3.0 * weights[:, 1, None, None, None] * lidar_pool1,
            3.0 * weights[:, 2, None, None, None] * rgb_pool1,
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
        fused_feature = final_features.sum(dim=1)
        return {
            "logits_fusion": self.fusion_classifier(fused_feature),
            "logits_hsi_aux": aux_logits[:, 0],
            "logits_lidar_aux": aux_logits[:, 1],
            "logits_rgb_aux": aux_logits[:, 2],
            "aux_probabilities": probabilities,
            "q": q,
            "weights": weights,
        }


def formula16_loss(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """冲突门控的高置信错误惩罚。

    仅当某单模态预测错误时 wrong=1；错误预测的margin越大，
    -log(1-margin)越大；样本总体冲突越强，惩罚也越强。
    该项用于缓解“低熵/高置信但预测错误”的模态误导融合。
    """
    _, conflict = pairwise_conflict(probabilities)
    rho = prediction_margin(probabilities)
    prediction = probabilities.argmax(dim=2)
    wrong = prediction.ne(target[:, None]).float().detach()
    penalty = -torch.log((1.0 - rho).clamp_min(EPS))
    return (conflict.detach()[:, None] * wrong * penalty).mean()


def compute_loss(output: dict, target: torch.Tensor, args, criterion):
    """组合总损失：融合CE + 单模态辅助CE + 公式16 + q正确性BCE。"""
    fusion = criterion(output["logits_fusion"], target)
    aux = sum(criterion(output[key], target) for key in BASE.AUX_KEYS)
    calibration = formula16_loss(output["aux_probabilities"], target)
    correct = (
        output["aux_probabilities"].argmax(dim=2) == target[:, None]
    ).float().detach()
    # correct 是训练期监督标签：1表示该模态当前预测正确，0表示错误。
    # 推理时没有真实标签，不计算此BCE，只使用已经训练好的q预测头。
    q_loss = F.binary_cross_entropy(output["q"], correct)
    total = (
        fusion
        + args.aux_weight * aux
        + args.formula16_weight * calibration
        + args.q_loss_weight * q_loss
    )
    return total, fusion, aux, calibration, q_loss


@torch.no_grad()
def evaluate(model, loader) -> dict[str, float]:
    """验证融合OA、q的BCE以及平均最大模态权重（监控权重塌缩）。"""
    model.eval()
    total = correct = 0
    q_loss_sum = weight_max_sum = 0.0
    batches = 0
    for hsi, lidar, rgb, target in loader:
        hsi, lidar, rgb, target = (
            hsi.to(BASE.DEVICE),
            lidar.to(BASE.DEVICE),
            rgb.to(BASE.DEVICE),
            target.to(BASE.DEVICE),
        )
        output = model(hsi, lidar, rgb)
        prediction = output["logits_fusion"].argmax(dim=1)
        total += int(target.numel())
        correct += int(prediction.eq(target).sum().item())
        correctness = (
            output["aux_probabilities"].argmax(dim=2) == target[:, None]
        ).float()
        q_loss_sum += float(F.binary_cross_entropy(output["q"], correctness).item())
        weight_max_sum += float(output["weights"].max(dim=1).values.mean().item())
        batches += 1
    return {
        "OA": correct / max(1, total),
        "q_BCE": q_loss_sum / max(1, batches),
        "mean_max_weight": weight_max_sum / max(1, batches),
    }


def checkpoint_payload(model, optimizer, scheduler, epoch, phase, best_val, best_epoch, args):
    """打包可完整续训的状态，而不只是模型参数。"""
    return {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "phase": phase,
        "best_val_OA": float(best_val),
        "best_epoch": int(best_epoch),
        "config": vars(args),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, phase, best_val, best_epoch, args):
    # 先写同目录临时文件，完整写完后再原子替换正式checkpoint。
    # 因而中途断电/磁盘写入失败时，上一版checkpoint通常仍然完整。
    # Windows杀毒软件短暂占用文件时最多重试10次。
    payload = checkpoint_payload(
        model, optimizer, scheduler, epoch, phase, best_val, best_epoch, args
    )
    temp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    torch.save(payload, temp_path)
    for attempt in range(10):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(1.0)


def append_csv(path: Path, row: dict) -> None:
    """把一轮指标追加到CSV；文件首次创建时写表头。"""
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, optimizer, args, criterion):
    """执行一轮训练并返回各损失、OA和权重统计。"""
    model.train()
    sums = np.zeros(6, dtype=np.float64)
    total = correct = batches = 0
    for hsi, lidar, rgb, target in loader:
        hsi, lidar, rgb, target = (
            hsi.to(BASE.DEVICE),
            lidar.to(BASE.DEVICE),
            rgb.to(BASE.DEVICE),
            target.to(BASE.DEVICE),
        )
        optimizer.zero_grad()
        output = model(hsi, lidar, rgb)
        losses = compute_loss(output, target, args, criterion)
        losses[0].backward()
        optimizer.step()
        prediction = output["logits_fusion"].argmax(dim=1)
        total += int(target.numel())
        correct += int(prediction.eq(target).sum().item())
        for index, loss in enumerate(losses):
            sums[index] += float(loss.item())
        sums[5] += float(output["weights"].max(dim=1).values.mean().item())
        batches += 1
    return {
        "train_total_loss": sums[0] / batches,
        "train_fusion_loss": sums[1] / batches,
        "train_aux_loss": sums[2] / batches,
        "train_formula16_loss": sums[3] / batches,
        "train_q_loss": sums[4] / batches,
        "train_mean_max_weight": sums[5] / batches,
        "train_OA": correct / total,
    }


def load_training_state(path, model, optimizer, scheduler):
    # 加载本地可信的完整训练状态：模型、优化器、学习率调度器。
    # weights_only=False 是因为checkpoint不只有张量权重，还有优化器等对象。
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def main() -> None:
    """旧版独立实验入口：warm-up决策后切换Transformer1输入动态加权。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    dataset = BASE.load_cache(args.data_dir)
    train_index = np.asarray(dataset["train_idx"], dtype=np.int32)
    validation_index = stratified_limit(
        np.asarray(dataset["test_idx"], dtype=np.int32),
        args.validation_per_class,
        args.seed + 10000,
    )
    train_dataset = BASE.MaterializedTriModalDataset(
        dataset["features"], dataset["rgb"], train_index,
        dataset["label_shift"], args.patch_size,
        dataset["hsi_channels"], dataset["lidar_channels"],
    )
    validation_dataset = BASE.MaterializedTriModalDataset(
        dataset["features"], dataset["rgb"], validation_index,
        dataset["label_shift"], args.patch_size,
        dataset["hsi_channels"], dataset["lidar_channels"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = WarmupDynamicMSAF3(
        dataset["hsi_channels"], dataset["lidar_channels"],
        dataset["rgb_channels"], dataset["num_classes"], args,
    ).to(BASE.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.7
    )
    criterion = nn.CrossEntropyLoss()
    log_path = args.output_dir / "epoch_log.csv"
    warmup_latest = args.output_dir / "warmup_latest.pt"
    dynamic_latest = args.output_dir / "dynamic_latest.pt"
    decision_path = args.output_dir / "switch_decision.json"

    if dynamic_latest.exists():
        # 优先从动态阶段最新状态继续；否则再检查warm-up状态。
        state = load_training_state(dynamic_latest, model, optimizer, scheduler)
        start_epoch = int(state["epoch"]) + 1
        best_val = float(state["best_val_OA"])
        best_epoch = int(state["best_epoch"])
        phase = "dynamic"
        model.dynamic_enabled = True
        print(f"Resuming dynamic phase at epoch {start_epoch}", flush=True)
    else:
        phase = "warmup"
        model.dynamic_enabled = False
        if warmup_latest.exists():
            state = load_training_state(warmup_latest, model, optimizer, scheduler)
            start_epoch = int(state["epoch"]) + 1
            best_val = float(state["best_val_OA"])
            best_epoch = int(state["best_epoch"])
            print(f"Resuming warmup at epoch {start_epoch}", flush=True)
        else:
            start_epoch = 1
            best_val = -1.0
            best_epoch = 0

    if phase == "warmup":
        # warm-up期间模型内部仍计算q，但融合权重固定为1/3。
        for epoch in range(start_epoch, args.warmup_probe_epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, args, criterion
            )
            scheduler.step()
            val_metrics = evaluate(model, validation_loader)
            if val_metrics["OA"] >= best_val:
                best_val = val_metrics["OA"]
                best_epoch = epoch
            row = {
                "epoch": epoch,
                "phase": "warmup",
                "dynamic_enabled": 0,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **train_metrics,
                "val_OA": val_metrics["OA"],
                "val_q_BCE": val_metrics["q_BCE"],
                "val_mean_max_weight": val_metrics["mean_max_weight"],
                "seconds": time.perf_counter() - epoch_start,
            }
            append_csv(log_path, row)
            save_checkpoint(
                warmup_latest, model, optimizer, scheduler, epoch,
                "warmup", best_val, best_epoch, args,
            )
            if epoch in (args.warmup_candidate, args.warmup_probe_epochs):
                save_checkpoint(
                    args.output_dir / f"warmup_epoch_{epoch}.pt",
                    model, optimizer, scheduler, epoch,
                    "warmup", best_val, best_epoch, args,
                )
            print(
                f"epoch={epoch:03d} phase=warmup train_OA={train_metrics['train_OA']:.4f} "
                f"val_OA={val_metrics['OA']:.4f} qBCE={val_metrics['q_BCE']:.4f} "
                f"time={row['seconds']:.1f}s",
                flush=True,
            )

        rows = list(csv.DictReader(log_path.open(encoding="utf-8-sig")))
        warmup_rows = [row for row in rows if row["phase"] == "warmup"]
        first = [
            float(row["val_OA"]) for row in warmup_rows
            if int(row["epoch"]) <= args.warmup_candidate
        ]
        late = [
            float(row["val_OA"]) for row in warmup_rows
            if args.warmup_candidate < int(row["epoch"]) <= args.warmup_probe_epochs
        ]
        improvement = max(late) - max(first)
        # 若31--50轮相对前30轮提升不足阈值，就选第30轮切换动态融合；
        # 否则使用第50轮状态。此逻辑只属于早期Step15独立实验。
        switch_epoch = (
            args.warmup_candidate
            if improvement < args.warmup_min_improvement
            else args.warmup_probe_epochs
        )
        decision = {
            "best_val_OA_epochs_1_to_30": max(first),
            "best_val_OA_epochs_31_to_50": max(late),
            "late_improvement": improvement,
            "minimum_required_improvement": args.warmup_min_improvement,
            "selected_switch_epoch": switch_epoch,
        }
        decision_path.write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        selected_path = args.output_dir / f"warmup_epoch_{switch_epoch}.pt"
        state = load_training_state(selected_path, model, optimizer, scheduler)
        start_epoch = switch_epoch + 1
        best_val = -1.0
        best_epoch = switch_epoch
        model.dynamic_enabled = True
        save_checkpoint(
            dynamic_latest, model, optimizer, scheduler, switch_epoch,
            "dynamic", best_val, best_epoch, args,
        )
        print(f"Switching dynamic fusion on after epoch {switch_epoch}", flush=True)

    for epoch in range(start_epoch, args.total_epochs + 1):
        # 动态阶段：每个batch、每个样本都重新根据当前概率求FW权重。
        model.dynamic_enabled = True
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, args, criterion)
        scheduler.step()
        val_metrics = evaluate(model, validation_loader)
        if val_metrics["OA"] >= best_val:
            best_val = val_metrics["OA"]
            best_epoch = epoch
            save_checkpoint(
                args.output_dir / "best_dynamic.pt",
                model, optimizer, scheduler, epoch,
                "dynamic", best_val, best_epoch, args,
            )
        row = {
            "epoch": epoch,
            "phase": "dynamic",
            "dynamic_enabled": 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            "val_OA": val_metrics["OA"],
            "val_q_BCE": val_metrics["q_BCE"],
            "val_mean_max_weight": val_metrics["mean_max_weight"],
            "seconds": time.perf_counter() - epoch_start,
        }
        append_csv(log_path, row)
        save_checkpoint(
            dynamic_latest, model, optimizer, scheduler, epoch,
            "dynamic", best_val, best_epoch, args,
        )
        print(
            f"epoch={epoch:03d} phase=dynamic train_OA={train_metrics['train_OA']:.4f} "
            f"val_OA={val_metrics['OA']:.4f} maxW={val_metrics['mean_max_weight']:.4f} "
            f"best={best_val:.4f}@{best_epoch} time={row['seconds']:.1f}s",
            flush=True,
        )

    save_checkpoint(
        args.output_dir / "final_epoch_300.pt",
        model, optimizer, scheduler, args.total_epochs,
        "dynamic", best_val, best_epoch, args,
    )
    summary = {
        "status": "training_complete",
        "total_epochs": args.total_epochs,
        "best_dynamic_val_OA": best_val,
        "best_dynamic_epoch": best_epoch,
        "switch_decision": json.loads(decision_path.read_text(encoding="utf-8")),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
