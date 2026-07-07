import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat

try:
    import h5py
except ImportError:
    h5py = None

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_prep.szutree_utils import (
    build_sorted_index,
    detect_szutree_layout,
    load_szutree_raw,
    resize_rgb_to_match,
    split_per_class_count,
    split_per_class_percent,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["houston2018", "szutree"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--szutree-split-mode", choices=["count", "percent"], default="percent")
    parser.add_argument("--train-percent-per-class", type=float, default=1.0)
    parser.add_argument("--rgb-file", type=Path, default=None)
    parser.add_argument("--label-file", type=Path, default=None)
    return parser.parse_args()


def normalize_features(feats):
    feats = feats.astype(np.float32, copy=False)
    feats_min = feats.min(axis=(0, 1), keepdims=True)
    shifted = feats - feats_min
    feats_max = shifted.max(axis=(0, 1), keepdims=True)
    return np.divide(
        shifted,
        feats_max,
        out=np.zeros_like(shifted, dtype=np.float32),
        where=feats_max != 0,
    )


def ensure_label_array(labels, name):
    if sparse.issparse(labels):
        labels = labels.toarray()
    else:
        labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected {name} labels to be 2D, but got shape {labels.shape}")
    return labels


def ensure_channel_last(arr, channels):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if arr.shape[-1] == channels:
        return arr
    if arr.shape[0] == channels:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Could not align shape {arr.shape} to channel-last with {channels} channels")


def prepare_houston2018(data_dir, output_dir):
    mat_path = data_dir / "houston2018.mat"
    data = loadmat(mat_path)
    hsi = ensure_channel_last(data["hsi"], 50)
    lidar = ensure_channel_last(data["lidar"], 1)
    rgb = ensure_channel_last(data["rgb"], 3)
    train = ensure_label_array(data["train"], "train")
    test = ensure_label_array(data["test"], "test")

    target_h, target_w = hsi.shape[:2]
    rgb = resize_rgb_to_match(rgb, target_h, target_w)
    feats = np.concatenate([hsi, lidar], axis=2)
    feats_norm = normalize_features(feats)
    rgb_norm = normalize_features(rgb)

    train_idx = build_sorted_index(train)
    test_idx = build_sorted_index(test)
    label_shift = int(min(train_idx[:, 0].min(), test_idx[:, 0].min()))

    return {
        "dataset_name": "houston2018",
        "features": feats_norm.astype(np.float16),
        "rgb": rgb_norm.astype(np.float16),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "label_shift": np.array(label_shift, dtype=np.int64),
        "meta": {
            "dataset_name": "houston2018",
            "hsi_channels": 50,
            "lidar_channels": 1,
            "height": int(feats_norm.shape[0]),
            "width": int(feats_norm.shape[1]),
            "source": str(mat_path),
        },
    }


def prepare_szutree(data_dir, output_dir, seed, samples_per_class, split_mode, train_percent_per_class, rgb_file=None, label_file=None):
    raw = load_szutree_raw(data_dir, rgb_override=rgb_file, label_override=label_file)
    hsi = raw["hsi"]
    lidar = raw["lidar"]
    rgb = raw["rgb"]
    labels = raw["labels"]

    feats = np.concatenate([hsi, lidar], axis=2)
    feats_norm = normalize_features(feats)
    rgb_norm = normalize_features(rgb)

    if split_mode == "percent":
        train_idx, test_idx = split_per_class_percent(labels, seed, train_percent_per_class)
    else:
        train_idx, test_idx = split_per_class_count(labels, seed, samples_per_class)
    label_shift = int(min(train_idx[:, 0].min(), test_idx[:, 0].min()))

    return {
        "dataset_name": "szutree",
        "features": feats_norm.astype(np.float16),
        "rgb": rgb_norm.astype(np.float16),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "label_shift": np.array(label_shift, dtype=np.int64),
        "meta": {
            "dataset_name": "szutree",
            "hsi_channels": 98,
            "lidar_channels": 1,
            "height": int(feats_norm.shape[0]),
            "width": int(feats_norm.shape[1]),
            "source": str(data_dir),
            "szutree_layout": detect_szutree_layout(data_dir),
            "split_mode": split_mode,
            "samples_per_class": int(samples_per_class),
            "train_percent_per_class": float(train_percent_per_class),
            "split_seed": int(seed),
            "label_source": raw["label_source"],
            "rgb_source": raw["rgb_source"],
        },
    }


def save_prepared(prepared, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_patches.npy", prepared["features"])
    if prepared["rgb"] is not None:
        np.save(output_dir / "train_rgb_patches.npy", prepared["rgb"])
    np.save(output_dir / "train_labels.npy", prepared["train_idx"].astype(np.int32))
    np.save(output_dir / "test_labels.npy", prepared["test_idx"].astype(np.int32))
    np.save(output_dir / "label_shift.npy", prepared["label_shift"])
    (output_dir / "prepared.meta.json").write_text(
        json.dumps(prepared["meta"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = args.output_dir or (args.data_dir / "prepared")
    if args.dataset == "houston2018":
        prepared = prepare_houston2018(args.data_dir, output_dir)
    else:
        prepared = prepare_szutree(
            args.data_dir,
            output_dir,
            args.seed,
            args.samples_per_class,
            args.szutree_split_mode,
            args.train_percent_per_class,
            rgb_file=args.rgb_file,
            label_file=args.label_file,
        )
    save_prepared(prepared, output_dir)
    print(f"Prepared data saved to {output_dir}")


if __name__ == "__main__":
    main()
