# -*- coding: utf-8 -*-
"""Resumable full official Houston2018 test inference.

The official fixed split contains 18,750 training samples and 2,000,160 test
pixels.  Test patches are extracted in vectorized chunks from sliding-window
views, so the full patch tensor is never materialized.  Predictions, confidence
and shallow unimodal conflict are written to resumable NumPy memmaps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
STEP4_PATH = (
    EXPERIMENTS
    / "step4_correctness_calibration"
    / "run_formula16_finetune.py"
)
STEP15_PATH = (
    EXPERIMENTS
    / "step15_warmup_dynamic_300"
    / "train_warmup_then_dynamic.py"
)
STEP17_PATH = (
    EXPERIMENTS
    / "step17_final_fusion_dynamic"
    / "train_final_fusion_dynamic.py"
)
MSAF3_PATH = EXPERIMENTS.parent / "msaf3" / "train_msaf3.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module("formula16_base", STEP4_PATH)
STEP15 = load_module("warmup_dynamic_model", STEP15_PATH)
STEP17 = load_module("final_fusion_dynamic_model", STEP17_PATH)
MSAF3 = load_module("original_msaf3_model", MSAF3_PATH)
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\DATA_3\Houston2018\prepared_msaf3"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            EXPERIMENTS
            / "step1_pure_aux"
            / "results"
            / "houston2018_fulltrain_test200_seed4_aux01"
            / "best_model.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_aux01")
    parser.add_argument(
        "--index-file",
        type=Path,
        default=None,
        help="Optional [label,row,col] test index, used for resampled datasets such as SZUTree.",
    )
    parser.add_argument(
        "--model-kind",
        choices=(
            "pure_aux",
            "warmup_dynamic",
            "final_fusion_dynamic",
            "baseline_mcf3",
        ),
        default="pure_aux",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Stop after this total sample index for a resumable smoke test; 0 means all.",
    )
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def normalized_js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)
    middle = 0.5 * (p + q)
    value = 0.5 * torch.sum(p * (torch.log(p) - torch.log(middle)), dim=1)
    value += 0.5 * torch.sum(q * (torch.log(q) - torch.log(middle)), dim=1)
    return value / math.log(2.0)


def batch_conflict(output: dict[str, torch.Tensor]) -> torch.Tensor:
    probabilities = [F.softmax(output[key], dim=1) for key in BASE.AUX_KEYS]
    return (
        normalized_js(probabilities[0], probabilities[1])
        + normalized_js(probabilities[0], probabilities[2])
        + normalized_js(probabilities[1], probabilities[2])
    ) / 3.0


def create_or_open_memmap(path: Path, dtype, shape, resume: bool):
    if resume:
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise RuntimeError(f"Memmap mismatch for {path}: {array.shape}, {array.dtype}")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def save_progress(
    path: Path,
    next_index: int,
    total: int,
    cumulative_seconds: float,
    batch_size: int,
) -> None:
    path.write_text(
        json.dumps(
            {
                "next_index": int(next_index),
                "total_samples": int(total),
                "cumulative_seconds": float(cumulative_seconds),
                "batch_size": int(batch_size),
                "complete": bool(next_index >= total),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    conflict: np.ndarray,
    confidence: np.ndarray,
    num_classes: int,
) -> dict:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    row_sum = matrix.sum(axis=1)
    per_class = np.divide(
        np.diag(matrix),
        row_sum,
        out=np.zeros(num_classes, dtype=np.float64),
        where=row_sum != 0,
    )
    total = int(matrix.sum())
    oa = float(np.trace(matrix) / total)
    expected = float(matrix.sum(axis=0).dot(matrix.sum(axis=1))) / float(total**2)
    kappa = (oa - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    high_count = max(1, int(math.ceil(0.2 * total)))
    high_indices = np.argpartition(conflict, -high_count)[-high_count:]
    return {
        "samples": total,
        "OA": oa,
        "AA": float(per_class.mean()),
        "kappa": float(kappa),
        "per_class_accuracy": per_class.tolist(),
        "per_class_count": row_sum.tolist(),
        "confusion_matrix": matrix.tolist(),
        "mean_confidence": float(np.mean(confidence)),
        "mean_pairwise_conflict": float(np.mean(conflict)),
        "high_conflict_samples": high_count,
        "high_conflict_OA": float(
            np.mean(predictions[high_indices] == labels[high_indices])
        ),
        "high_conflict_threshold": float(np.min(conflict[high_indices])),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    dataset = BASE.load_cache(args.data_dir)
    if args.index_file is not None:
        test_index = np.asarray(np.load(args.index_file), dtype=np.int32)
        label_shift = int(test_index[:, 0].min())
    else:
        test_index = np.asarray(dataset["test_idx"], dtype=np.int32)
        label_shift = int(dataset["label_shift"])
    labels = test_index[:, 0].astype(np.int64) - label_shift
    total = len(test_index)
    if args.index_file is None and total != 2_000_160:
        raise RuntimeError(f"Unexpected official test size: {total}")

    if "train_idx" in dataset:
        train_counts = np.bincount(
            np.asarray(dataset["train_idx"])[:, 0].astype(np.int64) - label_shift,
            minlength=dataset["num_classes"],
        )
    else:
        train_counts = np.zeros(dataset["num_classes"], dtype=np.int64)
    test_counts = np.bincount(labels, minlength=dataset["num_classes"])
    expected_train = np.asarray(
        [1000, 1000, 1000, 1000, 1000, 1000, 500, 1000, 1000, 1000,
         1000, 1000, 1000, 1000, 1000, 1000, 250, 1000, 1000, 1000],
        dtype=np.int64,
    )
    if args.index_file is None and not np.array_equal(train_counts, expected_train):
        raise RuntimeError(f"Official train split mismatch: {train_counts.tolist()}")

    resume = progress_path.exists()
    if resume:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start_index = int(progress["next_index"])
        cumulative_seconds = float(progress.get("cumulative_seconds", 0.0))
    else:
        start_index = 0
        cumulative_seconds = 0.0

    predictions = create_or_open_memmap(
        args.output_dir / "prediction.npy", np.uint8, (total,), resume
    )
    confidence = create_or_open_memmap(
        args.output_dir / "confidence.npy", np.float32, (total,), resume
    )
    conflict = create_or_open_memmap(
        args.output_dir / "pairwise_conflict.npy", np.float32, (total,), resume
    )

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if args.model_kind == "warmup_dynamic":
        model_args = argparse.Namespace(**checkpoint["config"])
        model = STEP15.WarmupDynamicMSAF3(
            dataset["hsi_channels"],
            dataset["lidar_channels"],
            dataset["rgb_channels"],
            dataset["num_classes"],
            model_args,
        )
        model.dynamic_enabled = int(checkpoint["epoch"]) > int(
            getattr(model_args, "warmup_epochs", 0)
        )
    elif args.model_kind == "final_fusion_dynamic":
        model_args = argparse.Namespace(**checkpoint["config"])
        model = STEP17.FinalFusionDynamicMSAF3(
            dataset["hsi_channels"],
            dataset["lidar_channels"],
            dataset["rgb_channels"],
            dataset["num_classes"],
            model_args,
        )
        model.dynamic_enabled = int(checkpoint["epoch"]) > int(
            getattr(model_args, "warmup_epochs", 0)
        )
    elif args.model_kind == "baseline_mcf3":
        model = MSAF3.MCF3(
            HSIband=dataset["hsi_channels"],
            lidarband=dataset["lidar_channels"],
            rgbband=dataset["rgb_channels"],
            num_classes=dataset["num_classes"],
            use_pretrained=False,
            use_rgb_pretrained=False,
        )
    else:
        model = BASE.PureAuxMSAF3(
            dataset["hsi_channels"],
            dataset["lidar_channels"],
            dataset["rgb_channels"],
            dataset["num_classes"],
            use_pretrained=False,
        )
    model = model.to(BASE.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    saved_mask = BASE.temporarily_disable_attention_mask(model)
    model.eval()

    pad = args.patch_size // 2
    # Padding creates roughly 620 MB of float16 storage, versus ~25 GB if all
    # two-million patches were materialized.
    padded_features = np.pad(
        np.asarray(dataset["features"]),
        ((pad, pad), (pad, pad), (0, 0)),
        mode="constant",
    )
    padded_rgb = np.pad(
        np.asarray(dataset["rgb"]),
        ((pad, pad), (pad, pad), (0, 0)),
        mode="constant",
    )
    feature_windows = np.lib.stride_tricks.sliding_window_view(
        padded_features,
        (args.patch_size, args.patch_size),
        axis=(0, 1),
    )
    rgb_windows = np.lib.stride_tricks.sliding_window_view(
        padded_rgb,
        (args.patch_size, args.patch_size),
        axis=(0, 1),
    )

    target_index = total if args.max_samples <= 0 else min(total, args.max_samples)
    if start_index >= target_index:
        print(f"Already processed {start_index}/{total}; requested target {target_index}.")
    session_start = time.perf_counter()
    chunks = 0
    with torch.inference_mode():
        for begin in range(start_index, target_index, args.batch_size):
            end = min(target_index, begin + args.batch_size)
            coordinates = test_index[begin:end]
            rows = coordinates[:, 1]
            cols = coordinates[:, 2]
            patch = np.ascontiguousarray(
                feature_windows[rows, cols], dtype=np.float32
            )
            rgb_patch = np.ascontiguousarray(
                rgb_windows[rows, cols], dtype=np.float32
            )
            hsi = torch.from_numpy(patch[:, : dataset["hsi_channels"]]).to(
                BASE.DEVICE, non_blocking=True
            )
            lidar = torch.from_numpy(
                patch[
                    :,
                    dataset["hsi_channels"] : dataset["hsi_channels"]
                    + dataset["lidar_channels"],
                ]
            ).to(BASE.DEVICE, non_blocking=True)
            rgb = torch.from_numpy(rgb_patch).to(BASE.DEVICE, non_blocking=True)
            output = model(hsi, lidar, rgb)
            if isinstance(output, dict):
                fusion_logits = output["logits_fusion"]
                batch_conflict_value = batch_conflict(output)
            else:
                fusion_logits = output
                batch_conflict_value = torch.zeros(
                    output.shape[0], dtype=output.dtype, device=output.device
                )
            fusion_probability = F.softmax(fusion_logits, dim=1)
            batch_prediction = fusion_probability.argmax(dim=1)
            predictions[begin:end] = batch_prediction.cpu().numpy().astype(np.uint8)
            confidence[begin:end] = fusion_probability.max(dim=1).values.cpu().numpy()
            conflict[begin:end] = batch_conflict_value.cpu().numpy()
            chunks += 1

            if chunks % args.progress_interval == 0 or end == target_index:
                predictions.flush()
                confidence.flush()
                conflict.flush()
                session_seconds = time.perf_counter() - session_start
                cumulative = cumulative_seconds + session_seconds
                rate = (end - start_index) / max(session_seconds, 1e-9)
                remaining = (total - end) / max(rate, 1e-9)
                save_progress(
                    progress_path, end, total, cumulative, args.batch_size
                )
                print(
                    f"progress={end}/{total} ({100.0 * end / total:.2f}%) "
                    f"rate={rate:.1f} samples/s remaining={remaining / 60.0:.1f} min",
                    flush=True,
                )

    BASE.restore_attention_mask(saved_mask)
    cumulative_seconds += time.perf_counter() - session_start
    save_progress(
        progress_path, target_index, total, cumulative_seconds, args.batch_size
    )

    run_config = {
        "checkpoint": str(args.checkpoint),
        "model_kind": args.model_kind,
        "index_file": str(args.index_file) if args.index_file is not None else None,
        "checkpoint_metadata": {
            "epoch": int(checkpoint.get("epoch", -1)),
            "phase": str(checkpoint.get("phase", "unknown")),
            "best_val_OA": float(checkpoint.get("best_val_OA", float("nan"))),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "config": {
                key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
                for key, value in checkpoint.get("config", {}).items()
            },
        },
        "official_train_counts": train_counts.tolist(),
        "official_test_counts": test_counts.tolist(),
        "official_test_samples": total,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size,
        "processed_samples": target_index,
        "cumulative_seconds": cumulative_seconds,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if target_index < total:
        print(
            f"Smoke/partial run stopped at {target_index}. Re-run without "
            "--max-samples to resume the official full test.",
            flush=True,
        )
        return

    metrics = compute_metrics(
        labels,
        np.asarray(predictions, dtype=np.int64),
        np.asarray(conflict, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
        dataset["num_classes"],
    )
    metrics["elapsed_seconds"] = cumulative_seconds
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Official full Houston2018 test",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Official train counts: {train_counts.tolist()}",
        f"Official test samples: {total}",
        "",
        f"- OA: {metrics['OA']:.6f}",
        f"- AA: {metrics['AA']:.6f}",
        f"- Kappa: {metrics['kappa']:.6f}",
        f"- Fixed top-20% shallow-conflict OA: {metrics['high_conflict_OA']:.6f}",
        f"- Mean confidence: {metrics['mean_confidence']:.6f}",
        f"- Runtime: {cumulative_seconds / 60.0:.2f} minutes",
        "",
        "| class | test count | accuracy |",
        "|---:|---:|---:|",
    ]
    for index, (count, accuracy) in enumerate(
        zip(metrics["per_class_count"], metrics["per_class_accuracy"]), start=1
    ):
        report.append(f"| {index} | {count} | {accuracy:.6f} |")
    report.append("")
    if args.index_file is None:
        report.extend(
            [
                "The checkpoint was trained on the Houston2018 official training split.",
                "See checkpoint/run metadata for the exact validation protocol.",
            ]
        )
    else:
        report.append(f"Evaluation index: `{args.index_file}`")
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
