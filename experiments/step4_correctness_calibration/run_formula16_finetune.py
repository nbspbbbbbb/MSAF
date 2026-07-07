# -*- coding: utf-8 -*-
"""Ablate the conflict-weighted confident-error penalty described by Eq. #16."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
STEP1_DIR = ROOT / "experiments" / "step1_pure_aux"
MSAF3_DIR = ROOT / "msaf3"
for path in (STEP1_DIR, MSAF3_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_step1_pure_aux import MaterializedTriModalDataset, PureAuxMSAF3  # noqa: E402
from train_msaf3 import DEVICE, load_cache  # noqa: E402


AUX_KEYS = ("logits_hsi_aux", "logits_lidar_aux", "logits_rgb_aux")
MODALITIES = ("hsi", "lidar", "rgb")
EPS = 1e-7


def parse_args():
    base = Path(__file__).resolve().parent
    default_warmup = (
        ROOT
        / "experiments"
        / "step3_temperature_calibration"
        / "results"
        / "houston2018_split90_10_seed4_aux01"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-dir", type=Path, default=default_warmup)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\DATA_3\Houston2018\prepared_msaf3"))
    parser.add_argument("--output-dir", type=Path, default=base / "results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--cal-weights", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_loaders(args):
    dataset = load_cache(args.data_dir)
    train_idx = np.load(args.warmup_dir / "train_index.npy")
    calibration_idx = np.load(args.warmup_dir / "calibration_index.npy")
    development_test_idx = np.load(args.warmup_dir / "development_test_index.npy")
    label_shift = int(np.load(args.data_dir / "label_shift.npy"))
    datasets = [
        MaterializedTriModalDataset(
            dataset["features"], dataset["rgb"], indices, label_shift, 11,
            dataset["hsi_channels"], dataset["lidar_channels"],
        )
        for indices in (train_idx, calibration_idx, development_test_idx)
    ]
    train_dataset, calibration_dataset, test_dataset = datasets
    loaders = {
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True),
        "calibration": DataLoader(
            calibration_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        ),
        "development_test": DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        ),
    }
    return dataset, loaders


def build_model(dataset, checkpoint_path):
    model = PureAuxMSAF3(
        dataset["hsi_channels"], dataset["lidar_channels"], dataset["rgb_channels"],
        dataset["num_classes"], use_pretrained=False,
    ).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def normalized_js_torch(p, q):
    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)
    middle = 0.5 * (p + q)
    js = 0.5 * torch.sum(p * (torch.log(p) - torch.log(middle)), dim=1)
    js += 0.5 * torch.sum(q * (torch.log(q) - torch.log(middle)), dim=1)
    return js / math.log(2.0)


def formula16_loss(output, target):
    probabilities = [F.softmax(output[key], dim=1) for key in AUX_KEYS]
    conflict = (
        normalized_js_torch(probabilities[0], probabilities[1])
        + normalized_js_torch(probabilities[0], probabilities[2])
        + normalized_js_torch(probabilities[1], probabilities[2])
    ) / 3.0
    # Conflict is a sample weight, not an optimization target.  Detaching it
    # prevents the network from evading the penalty by simply shrinking JS.
    conflict_weight = conflict.detach()
    modality_losses = []
    for probability in probabilities:
        top2 = torch.topk(probability, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        prediction = probability.argmax(dim=1)
        wrong = prediction.ne(target).float().detach()
        confident_error_penalty = -torch.log((1.0 - margin).clamp_min(EPS))
        modality_losses.append(conflict_weight * wrong * confident_error_penalty)
    stacked = torch.stack(modality_losses, dim=1)
    return stacked.mean(), conflict.detach().mean()


def temporarily_disable_attention_mask(model):
    saved = []
    for module in model.modules():
        if hasattr(module, "mask_ratio"):
            saved.append((module, module.mask_ratio))
            module.mask_ratio = 0.0
    return saved


def restore_attention_mask(saved):
    for module, value in saved:
        module.mask_ratio = value


def softmax_numpy(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def evaluate(model, loader, num_classes):
    saved_mask = temporarily_disable_attention_mask(model)
    model.eval()
    collected = {"label": [], "logits_fusion": []}
    for key in AUX_KEYS:
        collected[key] = []
    with torch.no_grad():
        for hsi, lidar, rgb, target in loader:
            output = model(hsi.to(DEVICE), lidar.to(DEVICE), rgb.to(DEVICE))
            collected["label"].append(target.numpy())
            collected["logits_fusion"].append(output["logits_fusion"].cpu().numpy())
            for key in AUX_KEYS:
                collected[key].append(output[key].cpu().numpy())
    restore_attention_mask(saved_mask)
    labels = np.concatenate(collected["label"]).astype(np.int64)
    logits = {key: np.concatenate(value) for key, value in collected.items() if key != "label"}
    diagnostics = {}
    for name, key in zip(("fusion",) + MODALITIES, ("logits_fusion",) + AUX_KEYS):
        probability = softmax_numpy(logits[key])
        prediction = probability.argmax(axis=1)
        correct = prediction == labels
        conf = confusion_matrix(labels, prediction, labels=np.arange(num_classes))
        row_sum = conf.sum(axis=1)
        per_class = np.divide(np.diag(conf), row_sum, out=np.zeros(num_classes), where=row_sum != 0)
        entry = {
            "accuracy": float(correct.mean()),
            "AA": float(per_class.mean()),
            "prediction": prediction,
            "probability": probability,
            "logits": logits[key],
        }
        if name != "fusion":
            top2 = np.partition(probability, -2, axis=1)[:, -2:]
            top2.sort(axis=1)
            margin = top2[:, 1] - top2[:, 0]
            entry.update(
                {
                    "margin": margin,
                    "correct": correct,
                    "margin_correct_mean": float(margin[correct].mean()),
                    "margin_wrong_mean": float(margin[~correct].mean()),
                    "margin_correctness_auc": float(roc_auc_score(correct.astype(np.int64), margin)),
                    "wrong_margin_ge_03_count": int(((~correct) & (margin >= 0.3)).sum()),
                    "wrong_margin_ge_05_count": int(((~correct) & (margin >= 0.5)).sum()),
                }
            )
        diagnostics[name] = entry

    hsi_p = diagnostics["hsi"]["probability"]
    lidar_p = diagnostics["lidar"]["probability"]
    rgb_p = diagnostics["rgb"]["probability"]

    def js_np(p, q):
        p = np.clip(p, EPS, 1.0)
        q = np.clip(q, EPS, 1.0)
        middle = 0.5 * (p + q)
        return (
            0.5 * np.sum(p * np.log(p / middle), axis=1)
            + 0.5 * np.sum(q * np.log(q / middle), axis=1)
        ) / np.log(2.0)

    conflict = (js_np(hsi_p, lidar_p) + js_np(hsi_p, rgb_p) + js_np(lidar_p, rgb_p)) / 3.0
    high_conflict = conflict >= np.quantile(conflict, 0.8)
    for modality in MODALITIES:
        entry = diagnostics[modality]
        wrong = ~entry["correct"]
        margin = entry["margin"]
        entry["high_conflict_wrong_margin_ge_03_count"] = int(
            (high_conflict & wrong & (margin >= 0.3)).sum()
        )
        entry["high_conflict_wrong_margin_ge_05_count"] = int(
            (high_conflict & wrong & (margin >= 0.5)).sum()
        )
    return labels, diagnostics


def strip_arrays(entry):
    return {key: value for key, value in entry.items() if not isinstance(value, np.ndarray)}


def save_evaluation(output_dir, split_name, labels, diagnostics):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{split_name}_label.npy", labels)
    summary = {}
    for branch, entry in diagnostics.items():
        branch_dir = output_dir / branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        np.save(branch_dir / f"{split_name}_logits.npy", entry["logits"].astype(np.float32))
        np.save(branch_dir / f"{split_name}_probability.npy", entry["probability"].astype(np.float32))
        np.save(branch_dir / f"{split_name}_prediction.npy", entry["prediction"].astype(np.int64))
        summary[branch] = strip_arrays(entry)
    (output_dir / f"{split_name}_diagnostics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def train_variant(args, dataset, loaders, cal_weight, initial_state):
    set_seed(args.seed)
    model = PureAuxMSAF3(
        dataset["hsi_channels"], dataset["lidar_channels"], dataset["rgb_channels"],
        dataset["num_classes"], use_pretrained=False,
    ).to(DEVICE)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(initial_state)
    best_val_oa = -1.0
    epoch_rows = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss_sum = fusion_sum = aux_sum = cal_sum = conflict_sum = 0.0
        batches = 0
        for hsi, lidar, rgb, target in loaders["train"]:
            hsi, lidar, rgb, target = (
                hsi.to(DEVICE), lidar.to(DEVICE), rgb.to(DEVICE), target.to(DEVICE)
            )
            optimizer.zero_grad()
            output = model(hsi, lidar, rgb)
            fusion_loss = criterion(output["logits_fusion"], target)
            aux_loss = sum(criterion(output[key], target) for key in AUX_KEYS)
            cal_loss, conflict_mean = formula16_loss(output, target)
            total_loss = fusion_loss + args.aux_weight * aux_loss + cal_weight * cal_loss
            total_loss.backward()
            optimizer.step()
            batches += 1
            total_loss_sum += float(total_loss.item())
            fusion_sum += float(fusion_loss.item())
            aux_sum += float(aux_loss.item())
            cal_sum += float(cal_loss.item())
            conflict_sum += float(conflict_mean.item())

        _, val_diagnostics = evaluate(model, loaders["calibration"], dataset["num_classes"])
        val_oa = val_diagnostics["fusion"]["accuracy"]
        if val_oa >= best_val_oa:
            best_val_oa = val_oa
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        row = {
            "epoch": epoch,
            "total_loss": total_loss_sum / batches,
            "fusion_loss": fusion_sum / batches,
            "aux_loss": aux_sum / batches,
            "calibration_loss": cal_sum / batches,
            "conflict_mean": conflict_sum / batches,
            "val_fusion_OA": val_oa,
        }
        epoch_rows.append(row)
        print(
            f"lambda={cal_weight:g} epoch={epoch:02d} total={row['total_loss']:.6f} "
            f"cal={row['calibration_loss']:.6f} val_OA={val_oa:.4f}"
        )

    model.load_state_dict(best_state, strict=True)
    variant_dir = args.output_dir / f"lambda_cal_{cal_weight:g}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "lambda_cal": float(cal_weight),
            "best_val_OA": float(best_val_oa),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
        },
        variant_dir / "best_model.pt",
    )
    with (variant_dir / "epoch_log.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)
    results = {"lambda_cal": cal_weight, "best_val_OA": best_val_oa, "seconds": time.perf_counter() - start_time}
    for split_name in ("calibration", "development_test"):
        labels, diagnostics = evaluate(model, loaders[split_name], dataset["num_classes"])
        save_evaluation(variant_dir, split_name, labels, diagnostics)
        results[split_name] = {branch: strip_arrays(entry) for branch, entry in diagnostics.items()}
    (variant_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, loaders = make_loaders(args)
    warmup_model = build_model(dataset, args.warmup_dir / "best_model.pt")
    initial_state = {key: value.detach().cpu().clone() for key, value in warmup_model.state_dict().items()}
    del warmup_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_model = PureAuxMSAF3(
        dataset["hsi_channels"], dataset["lidar_channels"], dataset["rgb_channels"],
        dataset["num_classes"], use_pretrained=False,
    ).to(DEVICE)
    baseline_model.load_state_dict(initial_state, strict=True)
    baseline_results = {}
    for split_name in ("calibration", "development_test"):
        labels, diagnostics = evaluate(baseline_model, loaders[split_name], dataset["num_classes"])
        save_evaluation(args.output_dir / "warmup_baseline", split_name, labels, diagnostics)
        baseline_results[split_name] = {branch: strip_arrays(entry) for branch, entry in diagnostics.items()}
    (args.output_dir / "warmup_baseline" / "summary.json").write_text(
        json.dumps(baseline_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_results = {"warmup_baseline": baseline_results}
    for cal_weight in args.cal_weights:
        all_results[f"lambda_cal_{cal_weight:g}"] = train_variant(
            args, dataset, loaders, cal_weight, initial_state
        )
    (args.output_dir / "all_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[saved] {args.output_dir}")


if __name__ == "__main__":
    main()
