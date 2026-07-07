from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import os
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
MSAF3_DIR = ROOT_DIR / "msaf3"
if str(MSAF3_DIR) not in sys.path:
    sys.path.insert(0, str(MSAF3_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _install_optional_import_stub(module_name: str, **attrs) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    module = types.ModuleType(module_name)
    for name, value in attrs.items():
        setattr(module, name, value)
    sys.modules[module_name] = module


_install_optional_import_stub("torchsummary", summary=lambda *args, **kwargs: None)
_install_optional_import_stub("thop", profile=lambda *args, **kwargs: (0.0, 0.0))

import train_msaf3 as msaf3_train  # noqa: E402
from MSAF_3 import MCF3  # noqa: E402


MODALITIES = ("hsi", "lidar", "rgb")
PAIRS = (("hsi", "lidar"), ("hsi", "rgb"), ("lidar", "rgb"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MSAF3 training with gradient-balance/conflict diagnostics."
    )
    parser.add_argument("--dataset", choices=["houston2018", "szutree"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--szutree-split-mode", choices=["count", "percent"], default="percent")
    parser.add_argument("--train-percent-per-class", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--rgb-file", type=Path, default=None)
    parser.add_argument("--label-file", type=Path, default=None)
    parser.add_argument("--probe-stage", choices=["stage1", "stage2", "both"], default="stage2")
    parser.add_argument("--save-raw-batch", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--skip-masked-eval", action="store_true")
    parser.add_argument("--severe-conflict-threshold", type=float, default=-0.2)
    parser.add_argument("--min-train-size", type=int, default=100)
    parser.add_argument("--allow-small-data", action="store_true")
    return parser.parse_args()


def format_seconds(total_seconds: float) -> str:
    total_seconds = float(total_seconds)
    minutes, seconds = divmod(total_seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
    return f"{int(minutes):02d}:{seconds:05.2f}"


def set_seed(seed: int) -> None:
    msaf3_train.set_seed(seed)


def selected_stages(probe_stage: str) -> list[str]:
    if probe_stage == "both":
        return ["stage1", "stage2"]
    return [probe_stage]


class FusionGradientProbe:
    """Capture gradients at the shared pre-fusion pooling calls.

    MSAF3 calls `model.avgpool` three times in fixed order for HSI, LiDAR/CHM,
    and RGB before transformer1. It similarly calls `model.avgpool_2` three
    times before transformer2. The hook keeps those tensors and reads their
    gradients after backward().
    """

    def __init__(self, model: nn.Module, stages: list[str]):
        self.stages = stages
        self.current: dict[str, list[torch.Tensor]] = {stage: [] for stage in stages}
        self.handles = []
        if "stage1" in stages:
            self.handles.append(model.avgpool.register_forward_hook(self._make_hook("stage1")))
        if "stage2" in stages:
            self.handles.append(model.avgpool_2.register_forward_hook(self._make_hook("stage2")))

    def _make_hook(self, stage: str):
        def hook(_module, _inputs, output):
            if not torch.is_grad_enabled():
                return
            if not isinstance(output, torch.Tensor):
                return
            output.retain_grad()
            self.current[stage].append(output)

        return hook

    def clear(self) -> None:
        self.current = {stage: [] for stage in self.stages}

    def gradients(self) -> dict[str, dict[str, torch.Tensor]]:
        result: dict[str, dict[str, torch.Tensor]] = {}
        for stage, tensors in self.current.items():
            if len(tensors) < len(MODALITIES):
                continue
            stage_grads = {}
            for modality, tensor in zip(MODALITIES, tensors[: len(MODALITIES)]):
                if tensor.grad is not None:
                    stage_grads[modality] = tensor.grad.detach()
            if len(stage_grads) == len(MODALITIES):
                result[stage] = stage_grads
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def flatten_per_sample(grad: torch.Tensor) -> torch.Tensor:
    return grad.float().reshape(grad.shape[0], -1)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def balance_from_cge(e_i: float, e_j: float) -> tuple[float, float, float]:
    denom = e_i + e_j
    if denom <= 0:
        return 0.5, 0.5, 1.0
    w_i = e_i / denom
    w_j = e_j / denom
    return w_i, w_j, 8.0 * w_i * w_j - 1.0


def cosine_stats(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    severe_threshold: float,
) -> dict[str, float]:
    a = flatten_per_sample(grad_a)
    b = flatten_per_sample(grad_b)
    eps = 1e-12
    sample_cos = (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + eps)
    batch_a = a.reshape(-1)
    batch_b = b.reshape(-1)
    batch_cos = torch.dot(batch_a, batch_b) / (batch_a.norm() * batch_b.norm() + eps)
    return {
        "cos_batch": float(batch_cos.item()),
        "cos_mean": float(sample_cos.mean().item()),
        "cos_std": float(sample_cos.std(unbiased=False).item()) if sample_cos.numel() > 1 else 0.0,
        "cos_min": float(sample_cos.min().item()),
        "cos_max": float(sample_cos.max().item()),
        "conflict_rate": float((sample_cos < 0).float().mean().item()),
        "severe_conflict_rate": float((sample_cos < severe_threshold).float().mean().item()),
    }


def build_gradient_fieldnames() -> list[str]:
    fields = ["epoch", "batch", "global_step", "stage", "lr", "loss", "batch_acc"]
    for modality in MODALITIES:
        fields.extend(
            [
                f"{modality}_grad_norm",
                f"{modality}_grad_energy",
                f"{modality}_cge",
                f"{modality}_cge_share",
            ]
        )
    for a, b in PAIRS:
        pair = f"{a}_{b}"
        fields.extend(
            [
                f"{pair}_balance",
                f"{pair}_cos_batch",
                f"{pair}_cos_mean",
                f"{pair}_cos_std",
                f"{pair}_cos_min",
                f"{pair}_cos_max",
                f"{pair}_conflict_rate",
                f"{pair}_severe_conflict_rate",
            ]
        )
    return fields


def build_bcp_fieldnames() -> list[str]:
    return [
        "epoch",
        "batch",
        "global_step",
        "stage",
        "pair",
        "modality_i",
        "modality_j",
        "cge_i",
        "cge_j",
        "wi",
        "wj",
        "balance_factor",
        "nonconflict_factor",
        "cos_mean",
        "cos_std",
        "cos_min",
        "cos_max",
        "conflict_rate",
        "severe_conflict_rate",
    ]


def build_epoch_fieldnames() -> list[str]:
    fields = ["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    for modality in MODALITIES:
        fields.extend([f"{modality}_cge", f"{modality}_cge_share"])
    for a, b in PAIRS:
        pair = f"{a}_{b}"
        fields.extend(
            [
                f"{pair}_balance",
                f"{pair}_cos_batch_mean",
                f"{pair}_cos_mean_mean",
                f"{pair}_conflict_rate_mean",
                f"{pair}_severe_conflict_rate_mean",
            ]
        )
    return fields


def build_epoch_overview_fieldnames() -> list[str]:
    fields = ["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    for modality in MODALITIES:
        fields.extend(
            [
                f"{modality}_cge",
                f"{modality}_cge_delta",
                f"{modality}_cge_share",
                f"{modality}_delta_share",
            ]
        )
    fields.extend(
        [
            "dominant_modality",
            "weak_modality",
            "cge_gap",
            "dominance_ratio",
            "effective_modalities",
            "imbalance_level",
            "avg_pair_cos",
            "min_pair_cos",
            "worst_conflict_pair",
            "mean_sample_conflict_rate",
            "any_pair_conflict_rate",
            "severe_conflict_rate",
            "conflict_level",
            "joint_problem_score",
            "joint_problem_level",
        ]
    )
    return fields


def build_pair_conflict_epoch_fieldnames() -> list[str]:
    return [
        "epoch",
        "stage",
        "pair",
        "cos_batch_mean",
        "cos_mean_mean",
        "cos_min_observed",
        "cos_max_observed",
        "conflict_rate_mean",
        "severe_conflict_rate_mean",
        "conflict_level",
    ]


def build_final_decision_fieldnames() -> list[str]:
    return [
        "dataset",
        "train_size",
        "test_size",
        "num_classes",
        "stage",
        "epoch",
        "imbalance_level",
        "conflict_level",
        "joint_problem_level",
        "joint_problem_score",
        "dominant_modality",
        "weak_modality",
        "cge_gap",
        "dominance_ratio",
        "effective_modalities",
        "avg_pair_cos",
        "min_pair_cos",
        "worst_conflict_pair",
        "mean_sample_conflict_rate",
        "any_pair_conflict_rate",
        "severe_conflict_rate",
        "full_acc",
        "max_mask_drop_setting",
        "max_mask_drop",
        "interpretation",
    ]


def stage_metric_rows(
    epoch: int,
    batch_idx: int,
    global_step: int,
    stage: str,
    grads: dict[str, torch.Tensor],
    cge: dict[str, dict[str, float]],
    lr: float,
    loss_value: float,
    batch_acc: float,
    severe_threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, dict[str, float]]]:
    row: dict[str, float] = {
        "epoch": epoch,
        "batch": batch_idx,
        "global_step": global_step,
        "stage": stage,
        "lr": lr,
        "loss": loss_value,
        "batch_acc": batch_acc,
    }

    batch_energy = {}
    for modality in MODALITIES:
        grad = grads[modality].float()
        energy = float(grad.pow(2).sum().item()) * lr
        batch_energy[modality] = energy
        cge[stage][modality] += energy
        row[f"{modality}_grad_norm"] = float(grad.norm().item())
        row[f"{modality}_grad_energy"] = energy
        row[f"{modality}_cge"] = cge[stage][modality]

    total_cge = sum(cge[stage].values())
    for modality in MODALITIES:
        row[f"{modality}_cge_share"] = safe_div(cge[stage][modality], total_cge)

    pair_rows = []
    pair_stats_for_epoch = {}
    for a, b in PAIRS:
        pair = f"{a}_{b}"
        stats = cosine_stats(grads[a], grads[b], severe_threshold)
        w_i, w_j, balance = balance_from_cge(cge[stage][a], cge[stage][b])
        pair_stats_for_epoch[pair] = {**stats, "balance": balance}

        row[f"{pair}_balance"] = balance
        for key, value in stats.items():
            row[f"{pair}_{key}"] = value

        pair_rows.append(
            {
                "epoch": epoch,
                "batch": batch_idx,
                "global_step": global_step,
                "stage": stage,
                "pair": pair,
                "modality_i": a,
                "modality_j": b,
                "cge_i": cge[stage][a],
                "cge_j": cge[stage][b],
                "wi": w_i,
                "wj": w_j,
                "balance_factor": balance,
                "nonconflict_factor": stats["cos_batch"],
                "cos_mean": stats["cos_mean"],
                "cos_std": stats["cos_std"],
                "cos_min": stats["cos_min"],
                "cos_max": stats["cos_max"],
                "conflict_rate": stats["conflict_rate"],
                "severe_conflict_rate": stats["severe_conflict_rate"],
            }
        )

    return row, pair_rows, pair_stats_for_epoch


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def min_or_zero(values: list[float]) -> float:
    return float(np.min(values)) if values else 0.0


def max_or_zero(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def modality_share(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {modality: safe_div(values[modality], total) for modality in MODALITIES}


def effective_modality_count(shares: dict[str, float]) -> float:
    denom = sum(value * value for value in shares.values())
    return safe_div(1.0, denom)


def classify_imbalance(cge_gap: float, dominance_ratio: float) -> str:
    if cge_gap >= 0.25 or dominance_ratio >= 2.0:
        return "strong"
    if cge_gap >= 0.15 or dominance_ratio >= 1.5:
        return "moderate"
    return "weak"


def classify_conflict(min_pair_cos: float, mean_conflict_rate: float, severe_conflict_rate: float) -> str:
    if severe_conflict_rate >= 0.10 or mean_conflict_rate >= 0.30 or min_pair_cos <= -0.20:
        return "strong"
    if mean_conflict_rate >= 0.10 or min_pair_cos < 0.0:
        return "moderate"
    return "weak"


def classify_joint_problem(cge_gap: float, mean_conflict_rate: float, severe_conflict_rate: float) -> tuple[float, str]:
    score = cge_gap * max(mean_conflict_rate, severe_conflict_rate)
    if score >= 0.075:
        return score, "strong"
    if score >= 0.025:
        return score, "moderate"
    return score, "weak"


def pair_level(row: dict[str, float]) -> str:
    return classify_conflict(
        min_pair_cos=float(row["cos_mean_mean"]),
        mean_conflict_rate=float(row["conflict_rate_mean"]),
        severe_conflict_rate=float(row["severe_conflict_rate_mean"]),
    )


def make_pair_conflict_rows(
    epoch: int,
    stage: str,
    epoch_pair_values: dict[tuple[str, str, str], list[float]],
) -> list[dict[str, float]]:
    rows = []
    for a, b in PAIRS:
        pair = f"{a}_{b}"
        row = {
            "epoch": epoch,
            "stage": stage,
            "pair": pair,
            "cos_batch_mean": mean_or_zero(epoch_pair_values[(stage, pair, "cos_batch")]),
            "cos_mean_mean": mean_or_zero(epoch_pair_values[(stage, pair, "cos_mean")]),
            "cos_min_observed": min_or_zero(epoch_pair_values[(stage, pair, "cos_min")]),
            "cos_max_observed": max_or_zero(epoch_pair_values[(stage, pair, "cos_max")]),
            "conflict_rate_mean": mean_or_zero(epoch_pair_values[(stage, pair, "conflict_rate")]),
            "severe_conflict_rate_mean": mean_or_zero(epoch_pair_values[(stage, pair, "severe_conflict_rate")]),
        }
        row["conflict_level"] = pair_level(row)
        rows.append(row)
    return rows


def make_epoch_overview_row(
    epoch: int,
    stage: str,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    lr: float,
    cge: dict[str, dict[str, float]],
    epoch_cge_start: dict[str, dict[str, float]],
    epoch_pair_values: dict[tuple[str, str, str], list[float]],
    epoch_stage_values: dict[tuple[str, str], list[float]],
) -> dict[str, float]:
    current_cge = {modality: cge[stage][modality] for modality in MODALITIES}
    delta_cge = {
        modality: cge[stage][modality] - epoch_cge_start[stage][modality]
        for modality in MODALITIES
    }
    shares = modality_share(current_cge)
    delta_shares = modality_share(delta_cge)
    dominant = max(MODALITIES, key=lambda modality: shares[modality])
    weak = min(MODALITIES, key=lambda modality: shares[modality])
    cge_gap = shares[dominant] - shares[weak]
    dominance_ratio = safe_div(shares[dominant], shares[weak])
    effective_modalities = effective_modality_count(shares)

    pair_rows = make_pair_conflict_rows(epoch, stage, epoch_pair_values)
    avg_pair_cos = mean_or_zero([float(row["cos_mean_mean"]) for row in pair_rows])
    min_pair_cos = min_or_zero([float(row["cos_mean_mean"]) for row in pair_rows])
    worst_pair = min(pair_rows, key=lambda row: float(row["cos_mean_mean"]))["pair"] if pair_rows else ""
    mean_conflict_rate = mean_or_zero([float(row["conflict_rate_mean"]) for row in pair_rows])
    severe_conflict_rate = mean_or_zero([float(row["severe_conflict_rate_mean"]) for row in pair_rows])
    any_pair_conflict_rate = mean_or_zero(epoch_stage_values[(stage, "any_pair_conflict")])

    imbalance = classify_imbalance(cge_gap, dominance_ratio)
    conflict = classify_conflict(min_pair_cos, mean_conflict_rate, severe_conflict_rate)
    joint_score, joint_level = classify_joint_problem(cge_gap, mean_conflict_rate, severe_conflict_rate)

    row: dict[str, float] = {
        "epoch": epoch,
        "stage": stage,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": lr,
        "dominant_modality": dominant,
        "weak_modality": weak,
        "cge_gap": cge_gap,
        "dominance_ratio": dominance_ratio,
        "effective_modalities": effective_modalities,
        "imbalance_level": imbalance,
        "avg_pair_cos": avg_pair_cos,
        "min_pair_cos": min_pair_cos,
        "worst_conflict_pair": worst_pair,
        "mean_sample_conflict_rate": mean_conflict_rate,
        "any_pair_conflict_rate": any_pair_conflict_rate,
        "severe_conflict_rate": severe_conflict_rate,
        "conflict_level": conflict,
        "joint_problem_score": joint_score,
        "joint_problem_level": joint_level,
    }
    for modality in MODALITIES:
        row[f"{modality}_cge"] = current_cge[modality]
        row[f"{modality}_cge_delta"] = delta_cge[modality]
        row[f"{modality}_cge_share"] = shares[modality]
        row[f"{modality}_delta_share"] = delta_shares[modality]
    return row


def make_epoch_row(
    epoch: int,
    stage: str,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    lr: float,
    cge: dict[str, dict[str, float]],
    epoch_pair_values: dict[tuple[str, str, str], list[float]],
) -> dict[str, float]:
    row: dict[str, float] = {
        "epoch": epoch,
        "stage": stage,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": lr,
    }
    total_cge = sum(cge[stage].values())
    for modality in MODALITIES:
        row[f"{modality}_cge"] = cge[stage][modality]
        row[f"{modality}_cge_share"] = safe_div(cge[stage][modality], total_cge)

    for a, b in PAIRS:
        pair = f"{a}_{b}"
        _w_i, _w_j, balance = balance_from_cge(cge[stage][a], cge[stage][b])
        row[f"{pair}_balance"] = balance
        row[f"{pair}_cos_batch_mean"] = mean_or_zero(epoch_pair_values[(stage, pair, "cos_batch")])
        row[f"{pair}_cos_mean_mean"] = mean_or_zero(epoch_pair_values[(stage, pair, "cos_mean")])
        row[f"{pair}_conflict_rate_mean"] = mean_or_zero(epoch_pair_values[(stage, pair, "conflict_rate")])
        row[f"{pair}_severe_conflict_rate_mean"] = mean_or_zero(
            epoch_pair_values[(stage, pair, "severe_conflict_rate")]
        )
    return row


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: torch.device,
    mask: str | None = None,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    batches_seen = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch_idx, (hsi, lidar, rgb, target) in enumerate(data_loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            rgb = rgb.to(device)
            target = target.to(device)

            if mask == "hsi":
                hsi = torch.zeros_like(hsi)
            elif mask == "lidar":
                lidar = torch.zeros_like(lidar)
            elif mask == "rgb":
                rgb = torch.zeros_like(rgb)

            output = model(hsi, lidar, rgb)
            loss = criterion(output, target.long())
            pred = output.argmax(dim=1)
            total_loss += float(loss.item())
            correct += int(pred.eq(target).sum().item())
            total += int(target.numel())
            conf_mat += confusion_matrix(
                target.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                labels=np.arange(num_classes),
            )
            batches_seen += 1

    row_sum = conf_mat.sum(axis=1)
    per_class_acc = np.nan_to_num(np.diag(conf_mat) / np.maximum(row_sum, 1))
    return {
        "loss": total_loss / max(1, batches_seen),
        "acc": safe_div(correct, total),
        "correct": correct,
        "total": total,
        "confusion": conf_mat,
        "per_class_acc": per_class_acc,
    }


def write_masked_eval(
    model: nn.Module,
    criterion: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    output_dir: Path,
    max_batches: int | None,
) -> None:
    masks = [None, "hsi", "lidar", "rgb"]
    labels = {None: "full", "hsi": "without_hsi", "lidar": "without_lidar", "rgb": "without_rgb"}
    results = {}
    for mask in masks:
        results[mask] = evaluate_loader(
            model,
            test_loader,
            criterion,
            num_classes,
            device,
            mask=mask,
            max_batches=max_batches,
        )

    full_acc = float(results[None]["acc"])
    summary_path = output_dir / "masked_eval_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "loss", "acc", "acc_percent", "drop_from_full", "correct", "total"])
        writer.writeheader()
        for mask in masks:
            res = results[mask]
            acc = float(res["acc"])
            writer.writerow(
                {
                    "setting": labels[mask],
                    "loss": float(res["loss"]),
                    "acc": acc,
                    "acc_percent": acc * 100.0,
                    "drop_from_full": full_acc - acc,
                    "correct": int(res["correct"]),
                    "total": int(res["total"]),
                }
            )

    per_class_path = output_dir / "masked_eval_per_class.csv"
    with per_class_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["class_id"] + [labels[mask] for mask in masks] + [
            "drop_without_hsi",
            "drop_without_lidar",
            "drop_without_rgb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        full_per_class = results[None]["per_class_acc"]
        for class_id in range(num_classes):
            row = {"class_id": class_id}
            for mask in masks:
                row[labels[mask]] = float(results[mask]["per_class_acc"][class_id])
            row["drop_without_hsi"] = float(full_per_class[class_id] - results["hsi"]["per_class_acc"][class_id])
            row["drop_without_lidar"] = float(full_per_class[class_id] - results["lidar"]["per_class_acc"][class_id])
            row["drop_without_rgb"] = float(full_per_class[class_id] - results["rgb"]["per_class_acc"][class_id])
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def format_float(value: float) -> str:
    return f"{value:.4f}"


def latest_epoch_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    latest_epoch = max(int(float(row["epoch"])) for row in rows if row.get("epoch"))
    return [row for row in rows if int(float(row.get("epoch", 0))) == latest_epoch]


def masked_summary(masked_rows: list[dict[str, str]]) -> tuple[float, str, float]:
    if not masked_rows:
        return 0.0, "", 0.0
    full_acc = 0.0
    for row in masked_rows:
        if row.get("setting") == "full":
            full_acc = to_float(row, "acc")
            break
    candidates = [
        (to_float(row, "drop_from_full"), row.get("setting", ""))
        for row in masked_rows
        if row.get("setting") != "full"
    ]
    if not candidates:
        return full_acc, "", 0.0
    max_drop, setting = max(candidates)
    return full_acc, setting, max_drop


def make_interpretation(row: dict[str, str], max_mask_drop_setting: str, max_mask_drop: float) -> str:
    imbalance = row.get("imbalance_level", "weak")
    conflict = row.get("conflict_level", "weak")
    joint = row.get("joint_problem_level", "weak")
    dominant = row.get("dominant_modality", "")
    weak = row.get("weak_modality", "")
    worst_pair = row.get("worst_conflict_pair", "")

    if joint in {"strong", "moderate"}:
        base = f"{joint} evidence of the coupled imbalance/conflict problem"
    elif imbalance in {"strong", "moderate"}:
        base = f"{imbalance} modality imbalance, but conflict is {conflict}"
    elif conflict in {"strong", "moderate"}:
        base = f"{conflict} gradient conflict, but CGE imbalance is {imbalance}"
    else:
        base = "weak evidence for the paper's coupled problem in this run"

    details = f"dominant={dominant}, weak={weak}, worst_pair={worst_pair}"
    if max_mask_drop_setting:
        details += f", largest_mask_drop={max_mask_drop_setting}:{format_float(max_mask_drop)}"
    return f"{base}; {details}."


def write_final_decision(output_dir: Path) -> None:
    config_path = output_dir / "run_config.json"
    epoch_rows = latest_epoch_rows(read_csv_rows(output_dir / "epoch_overview.csv"))
    masked_rows = read_csv_rows(output_dir / "masked_eval_summary.csv")
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_info = config.get("dataset", {})
    full_acc, max_drop_setting, max_drop = masked_summary(masked_rows)

    final_path = output_dir / "final_decision.csv"
    with final_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=build_final_decision_fieldnames())
        writer.writeheader()
        for row in epoch_rows:
            final_row = {
                "dataset": dataset_info.get("dataset_name", ""),
                "train_size": dataset_info.get("train_size", ""),
                "test_size": dataset_info.get("test_size", ""),
                "num_classes": dataset_info.get("num_classes", ""),
                "stage": row.get("stage", ""),
                "epoch": row.get("epoch", ""),
                "imbalance_level": row.get("imbalance_level", ""),
                "conflict_level": row.get("conflict_level", ""),
                "joint_problem_level": row.get("joint_problem_level", ""),
                "joint_problem_score": row.get("joint_problem_score", ""),
                "dominant_modality": row.get("dominant_modality", ""),
                "weak_modality": row.get("weak_modality", ""),
                "cge_gap": row.get("cge_gap", ""),
                "dominance_ratio": row.get("dominance_ratio", ""),
                "effective_modalities": row.get("effective_modalities", ""),
                "avg_pair_cos": row.get("avg_pair_cos", ""),
                "min_pair_cos": row.get("min_pair_cos", ""),
                "worst_conflict_pair": row.get("worst_conflict_pair", ""),
                "mean_sample_conflict_rate": row.get("mean_sample_conflict_rate", ""),
                "any_pair_conflict_rate": row.get("any_pair_conflict_rate", ""),
                "severe_conflict_rate": row.get("severe_conflict_rate", ""),
                "full_acc": full_acc,
                "max_mask_drop_setting": max_drop_setting,
                "max_mask_drop": max_drop,
            }
            final_row["interpretation"] = make_interpretation(row, max_drop_setting, max_drop)
            writer.writerow(final_row)


def write_excel_summary(output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[warn] pandas is not installed; skipped diagnostic_summary.xlsx")
        return

    workbook_path = output_dir / "diagnostic_summary.xlsx"
    sheets = [
        ("final_decision", output_dir / "final_decision.csv"),
        ("epoch_overview", output_dir / "epoch_overview.csv"),
        ("pair_conflict_epoch", output_dir / "pair_conflict_epoch.csv"),
        ("masked_eval", output_dir / "masked_eval_summary.csv"),
        ("masked_per_class", output_dir / "masked_eval_per_class.csv"),
    ]
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, path in sheets:
            if path.exists():
                pd.read_csv(path).to_excel(writer, sheet_name=sheet_name, index=False)


def write_diagnostic_summary(output_dir: Path) -> None:
    config_path = output_dir / "run_config.json"
    final_rows = read_csv_rows(output_dir / "final_decision.csv")
    epoch_rows = latest_epoch_rows(read_csv_rows(output_dir / "epoch_overview.csv"))
    masked_rows = read_csv_rows(output_dir / "masked_eval_summary.csv")
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    lines = ["# MSAF3 Diagnostic Summary", ""]
    dataset_info = config.get("dataset", {})
    if dataset_info:
        lines.extend(
            [
                "## Dataset",
                "",
                f"- dataset: {dataset_info.get('dataset_name')}",
                f"- train_size: {dataset_info.get('train_size')}",
                f"- test_size: {dataset_info.get('test_size')}",
                f"- num_classes: {dataset_info.get('num_classes')}",
                f"- hsi/lidar/rgb channels: {dataset_info.get('hsi_channels')}/{dataset_info.get('lidar_channels')}/{dataset_info.get('rgb_channels')}",
                "",
            ]
        )

    if final_rows:
        lines.extend(["## Final Decision", ""])
        for row in final_rows:
            lines.extend(
                [
                    f"### {row.get('stage', '')}",
                    "",
                    f"- joint_problem_level: {row.get('joint_problem_level')}",
                    f"- imbalance_level: {row.get('imbalance_level')}",
                    f"- conflict_level: {row.get('conflict_level')}",
                    f"- dominant / weak modality: {row.get('dominant_modality')} / {row.get('weak_modality')}",
                    f"- worst_conflict_pair: {row.get('worst_conflict_pair')}",
                    f"- interpretation: {row.get('interpretation')}",
                    "",
                ]
            )

    if epoch_rows:
        lines.extend(["## Last-Epoch Modality CGE", ""])
        for row in epoch_rows:
            stage = row.get("stage", "")
            shares = {m: to_float(row, f"{m}_cge_share") for m in MODALITIES}
            lines.append(f"### {stage}")
            lines.append("")
            lines.append(
                "- CGE share: "
                + ", ".join([f"{m}={format_float(shares[m])}" for m in MODALITIES])
                + f" ; gap={format_float(to_float(row, 'cge_gap'))}"
            )
            lines.append(f"- dominance_ratio: {format_float(to_float(row, 'dominance_ratio'))}")
            lines.append(f"- effective_modalities: {format_float(to_float(row, 'effective_modalities'))}")
            lines.append(f"- avg_pair_cos: {format_float(to_float(row, 'avg_pair_cos'))}")
            lines.append(f"- min_pair_cos: {format_float(to_float(row, 'min_pair_cos'))}")
            lines.append(f"- mean_sample_conflict_rate: {format_float(to_float(row, 'mean_sample_conflict_rate'))}")
            lines.append("")

    if masked_rows:
        lines.extend(["## Masked Evaluation", ""])
        full_acc = 0.0
        for row in masked_rows:
            if row.get("setting") == "full":
                full_acc = to_float(row, "acc")
                break
        for row in masked_rows:
            setting = row.get("setting", "")
            acc = to_float(row, "acc")
            drop = to_float(row, "drop_from_full")
            lines.append(f"- {setting}: acc={format_float(acc)}, drop_from_full={format_float(drop)}")
        lines.append("")
        largest_drop = max((to_float(row, "drop_from_full"), row.get("setting", "")) for row in masked_rows)
        if largest_drop[0] > 0.10:
            lines.append(
                f"Masked-eval signal: `{largest_drop[1]}` has the largest accuracy drop "
                f"({format_float(largest_drop[0])}), suggesting the removed modality is heavily used."
            )
        elif full_acc > 0:
            lines.append("Masked-eval signal: no single modality removal caused a very large drop.")
        lines.append("")

    lines.extend(
        [
            "## How To Judge",
            "",
            "- Strong imbalance: one modality CGE share stays much larger than the others, especially gap > 0.25.",
            "- Strong conflict: min_pair_cos is negative/low or mean_sample_conflict_rate > 0.30.",
            "- Coupled problem: CGE_gap and conflict_rate are both high, reflected by joint_problem_level.",
            "- Modality dependence: masking one modality causes a much larger accuracy drop than masking others.",
            "",
        ]
    )
    (output_dir / "diagnostic_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_datasets(args: argparse.Namespace, dataset: dict, seed: int):
    if dataset["split_mode"] == "fixed":
        train_idx = dataset["train_idx"]
        test_idx = dataset["test_idx"]
        label_shift = dataset["label_shift"]
    else:
        if args.dataset == "szutree" and args.szutree_split_mode == "percent":
            train_idx, test_idx = msaf3_train.split_per_class_percent(
                dataset["labels"], seed, args.train_percent_per_class
            )
        else:
            train_idx, test_idx = msaf3_train.split_per_class_count(
                dataset["labels"], seed, args.samples_per_class
            )
        label_shift = int(min(train_idx[:, 0].min(), test_idx[:, 0].min()))

    train_sample = msaf3_train.infer_train_sample(train_idx, dataset["num_classes"])
    train_dataset = msaf3_train.PreparedTriModalDataset(
        dataset["features"],
        dataset["rgb"],
        train_idx,
        label_shift,
        args.patch_size,
        dataset["hsi_channels"],
        dataset["lidar_channels"],
    )
    val_dataset = msaf3_train.PreparedTriModalDataset(
        dataset["features"],
        dataset["rgb"],
        train_idx,
        label_shift,
        args.patch_size,
        dataset["hsi_channels"],
        dataset["lidar_channels"],
    )
    test_dataset = msaf3_train.PreparedTriModalDataset(
        dataset["features"],
        dataset["rgb"],
        test_idx,
        label_shift,
        args.patch_size,
        dataset["hsi_channels"],
        dataset["lidar_channels"],
    )
    return train_dataset, val_dataset, test_dataset, train_sample


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def save_best_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def ensure_dataset_size(args: argparse.Namespace, train_dataset, test_dataset, train_sample) -> None:
    if len(train_dataset) >= args.min_train_size:
        return
    message = (
        f"Refusing to run diagnostics on a tiny training split: train_size={len(train_dataset)}, "
        f"test_size={len(test_dataset)}, train_sample={train_sample}. This usually means --data-dir points "
        "to a toy/prepared cache rather than the full dataset. Use the full raw data directory, or pass "
        "--allow-small-data only for a smoke test."
    )
    if args.allow_small_data:
        print(f"[warn] {message}")
        return
    raise ValueError(message)


def train_with_diagnostics(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> nn.Module:
    stages = selected_stages(args.probe_stage)
    cge = {stage: {modality: 0.0 for modality in MODALITIES} for stage in stages}
    probe = FusionGradientProbe(model, stages)
    best_state = save_best_state(model)
    best_val_acc = 0.0
    best_epoch = 0
    early_counter = 0
    global_step = 0

    raw_batch_path = output_dir / "raw_batch_metrics.csv"
    raw_bcp_path = output_dir / "raw_bcp_pairs.csv"
    epoch_overview_path = output_dir / "epoch_overview.csv"
    pair_conflict_path = output_dir / "pair_conflict_epoch.csv"

    gradient_fields = build_gradient_fieldnames()
    bcp_fields = build_bcp_fieldnames()
    epoch_overview_fields = build_epoch_overview_fieldnames()
    pair_conflict_fields = build_pair_conflict_epoch_fieldnames()

    try:
        with epoch_overview_path.open("w", newline="", encoding="utf-8") as epoch_file, pair_conflict_path.open(
            "w", newline="", encoding="utf-8"
        ) as pair_file:
            epoch_writer = csv.DictWriter(epoch_file, fieldnames=epoch_overview_fields)
            pair_writer = csv.DictWriter(pair_file, fieldnames=pair_conflict_fields)
            epoch_writer.writeheader()
            pair_writer.writeheader()

            raw_batch_file = None
            raw_bcp_file = None
            gradient_writer = None
            bcp_writer = None
            if args.save_raw_batch:
                raw_batch_file = raw_batch_path.open("w", newline="", encoding="utf-8")
                raw_bcp_file = raw_bcp_path.open("w", newline="", encoding="utf-8")
                gradient_writer = csv.DictWriter(raw_batch_file, fieldnames=gradient_fields)
                bcp_writer = csv.DictWriter(raw_bcp_file, fieldnames=bcp_fields)
                gradient_writer.writeheader()
                bcp_writer.writeheader()

            for epoch in range(1, args.epochs + 1):
                epoch_start = time.perf_counter()
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                batches_seen = 0
                epoch_pair_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
                epoch_stage_values: dict[tuple[str, str], list[float]] = defaultdict(list)
                epoch_cge_start = {
                    stage: {modality: cge[stage][modality] for modality in MODALITIES}
                    for stage in stages
                }

                for batch_idx, (hsi, lidar, rgb, target) in enumerate(train_loader, start=1):
                    if args.max_train_batches is not None and batch_idx > args.max_train_batches:
                        break

                    hsi = hsi.to(device)
                    lidar = lidar.to(device)
                    rgb = rgb.to(device)
                    target = target.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    probe.clear()
                    output = model(hsi, lidar, rgb)
                    loss = criterion(output, target.long())
                    loss.backward()

                    lr = current_lr(optimizer)
                    pred = output.argmax(dim=1)
                    batch_correct = int(pred.eq(target).sum().item())
                    batch_total = int(target.numel())
                    batch_acc = safe_div(batch_correct, batch_total)
                    loss_value = float(loss.item())
                    global_step += 1

                    captured = probe.gradients()
                    for stage in stages:
                        if stage not in captured:
                            continue
                        row, pair_rows, pair_stats = stage_metric_rows(
                            epoch=epoch,
                            batch_idx=batch_idx,
                            global_step=global_step,
                            stage=stage,
                            grads=captured[stage],
                            cge=cge,
                            lr=lr,
                            loss_value=loss_value,
                            batch_acc=batch_acc,
                            severe_threshold=args.severe_conflict_threshold,
                        )
                        for pair, stats in pair_stats.items():
                            for key, value in stats.items():
                                epoch_pair_values[(stage, pair, key)].append(value)

                        epoch_stage_values[(stage, "any_pair_conflict")].append(
                            1.0 if any(stats["cos_batch"] < 0.0 for stats in pair_stats.values()) else 0.0
                        )

                        if args.save_raw_batch and (batch_idx == 1 or global_step % max(1, args.log_every) == 0):
                            gradient_writer.writerow(row)
                            for pair_row in pair_rows:
                                bcp_writer.writerow(pair_row)

                    optimizer.step()

                    train_loss += loss_value
                    correct += batch_correct
                    total += batch_total
                    batches_seen += 1

                train_loss = train_loss / max(1, batches_seen)
                train_acc = safe_div(correct, total)
                val_result = evaluate_loader(
                    model,
                    val_loader,
                    criterion,
                    num_classes=model.mlp_head.out_features,
                    device=device,
                    max_batches=args.max_eval_batches,
                )
                val_acc = float(val_result["acc"])
                val_loss = float(val_result["loss"])
                lr_before_step = current_lr(optimizer)
                scheduler.step()

                for stage in stages:
                    epoch_writer.writerow(
                        make_epoch_overview_row(
                            epoch=epoch,
                            stage=stage,
                            train_loss=train_loss,
                            train_acc=train_acc,
                            val_loss=val_loss,
                            val_acc=val_acc,
                            lr=lr_before_step,
                            cge=cge,
                            epoch_cge_start=epoch_cge_start,
                            epoch_pair_values=epoch_pair_values,
                            epoch_stage_values=epoch_stage_values,
                        )
                    )
                    for pair_row in make_pair_conflict_rows(epoch, stage, epoch_pair_values):
                        pair_writer.writerow(pair_row)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    best_state = save_best_state(model)
                    early_counter = 0
                else:
                    threshold_epoch = 100 if args.epochs > 100 else 50
                    if epoch > threshold_epoch:
                        early_counter += 1
                        if early_counter > 20:
                            print(
                                "Early stopping with best_val_acc: ",
                                best_val_acc,
                                "at epoch %d: ..." % best_epoch,
                            )
                            break

                print(
                    "epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, best epoch %d, time %s"
                    % (
                        epoch,
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc,
                        best_epoch,
                        format_seconds(time.perf_counter() - epoch_start),
                    )
                )

            model.load_state_dict(best_state)
            if raw_batch_file is not None:
                raw_batch_file.close()
            if raw_bcp_file is not None:
                raw_bcp_file.close()
    finally:
        probe.close()

    return model


def main() -> None:
    args = parse_args()
    os.chdir(ROOT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir or (ROOT_DIR / "msaf3_diagnostics" / "results")
    output_dir = output_base / f"{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    dataset_start = time.perf_counter()
    dataset, cache_dir = msaf3_train.resolve_dataset(args)
    print(f"[time] dataset load/prepare: {format_seconds(time.perf_counter() - dataset_start)}")
    print(f"[info] using cache/data source: {cache_dir}")

    train_dataset, val_dataset, test_dataset, train_sample = build_datasets(args, dataset, args.seed)
    ensure_dataset_size(args, train_dataset, test_dataset, train_sample)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        **msaf3_train.get_loader_kwargs(args, True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        **msaf3_train.get_loader_kwargs(args, True),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        **msaf3_train.get_loader_kwargs(args, False),
    )

    config = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "train_sample": train_sample,
        "dataset": {
            "dataset_name": dataset["dataset_name"],
            "split_mode": dataset["split_mode"],
            "hsi_channels": int(dataset["hsi_channels"]),
            "lidar_channels": int(dataset["lidar_channels"]),
            "rgb_channels": int(dataset["rgb_channels"]),
            "num_classes": int(dataset["num_classes"]),
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
        },
        "diagnostic_outputs": [
            "final_decision.csv",
            "epoch_overview.csv",
            "pair_conflict_epoch.csv",
            "masked_eval_summary.csv",
            "masked_eval_per_class.csv",
            "diagnostic_summary.md",
            "diagnostic_summary.xlsx",
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    model = MCF3(
        HSIband=dataset["hsi_channels"],
        lidarband=dataset["lidar_channels"],
        rgbband=dataset["rgb_channels"],
        num_classes=dataset["num_classes"],
        use_pretrained=not args.no_pretrained,
        use_rgb_pretrained=not args.no_pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    train_start = time.perf_counter()
    model = train_with_diagnostics(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        output_dir=output_dir,
        device=device,
    )
    print(f"[time] diagnostic train stage: {format_seconds(time.perf_counter() - train_start)}")

    if not args.skip_masked_eval:
        eval_start = time.perf_counter()
        write_masked_eval(
            model=model,
            criterion=criterion,
            test_loader=test_loader,
            num_classes=dataset["num_classes"],
            device=device,
            output_dir=output_dir,
            max_batches=args.max_eval_batches,
        )
        print(f"[time] masked evaluation: {format_seconds(time.perf_counter() - eval_start)}")

    write_final_decision(output_dir)
    write_diagnostic_summary(output_dir)
    write_excel_summary(output_dir)
    print(f"[info] diagnostics saved to {output_dir}")


if __name__ == "__main__":
    main()
