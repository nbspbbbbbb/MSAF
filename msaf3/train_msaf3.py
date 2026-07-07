# -*- coding: utf-8 -*-
"""
三模态 MSAF 训练入口。

支持：
1. Houston2018 三模态：HSI + LiDAR + RGB
2. SZUTree 三模态：HSI + CHM + RGB

设计目标：
1. 所有三模态相关逻辑集中在 msaf3 目录内；
2. 原始双模态文件完全不动；
3. 训练脚本可以直接从原始数据目录启动；
4. 首次运行会自动在数据目录下生成 prepared_msaf3 缓存，后续复用。
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from thop import profile
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except ImportError:
    h5py = None


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MSAF_3 import MCF3
from data_prep.szutree_utils import (
    build_sorted_index,
    choose_rgb_file,
    ensure_channel_last,
    load_mat_dict,
    load_rgb_array,
    load_szutree_raw,
    resize_label_map_nearest,
    resize_rgb_to_match,
    split_per_class_count,
    split_per_class_percent,
    split_index_train_validation,
)


cudnn.deterministic = True
cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这个脚本负责三模态训练全流程：
# 1. 读取或生成 prepared 缓存
# 2. 按固定划分或按类重采样生成 train/test
# 3. 组装 HSI / LiDAR / RGB patch 数据
# 4. 训练 MSAF3 并输出逐轮统计报告


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["houston2018", "szutree"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--itm", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--szutree-split-mode", choices=["count", "percent"], default="percent")
    parser.add_argument("--train-percent-per-class", type=float, default=1.0)
    parser.add_argument(
        "--validation-ratio-within-train",
        type=float,
        default=0.10,
        help="For SZUTree, hold out this fraction of the sampled 1%% development set for validation.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--rgb-file", type=Path, default=None)
    parser.add_argument("--label-file", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def format_seconds(total_seconds):
    total_seconds = float(total_seconds)
    minutes, seconds = divmod(total_seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
    return f"{int(minutes):02d}:{seconds:05.2f}"


def normalize_features(feats):
    # 按通道做 min-max 归一化，和预处理脚本保持同一口径。
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


def resize_label_map_nearest(labels, target_hw):
    target_h, target_w = target_hw
    if labels.shape == (target_h, target_w):
        return labels

    step_h = labels.shape[0] // target_h
    step_w = labels.shape[1] // target_w
    if step_h * target_h == labels.shape[0] and step_w * target_w == labels.shape[1]:
        return labels[::step_h, ::step_w]

    label_tensor = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(label_tensor, size=(target_h, target_w), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy().astype(labels.dtype)


def infer_train_sample(train_idx, num_classes):
    counts = np.bincount(train_idx[:, 0].astype(np.int64), minlength=num_classes + 1)[1:]
    return tuple(int(x) for x in counts.tolist())


def save_cache(cache_dir, meta, arrays):
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "prepared.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for name, value in arrays.items():
        np.save(cache_dir / f"{name}.npy", value)


def load_cache(cache_dir):
    meta_path = cache_dir / "prepared.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cache metadata under {cache_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # 兼容 data_prep/prepare_data.py 生成的标准 prepared 目录：
    # train_patches.npy / train_rgb_patches.npy / train_labels.npy / test_labels.npy
    # 兼容 data_prep/prepare_data.py 生成的标准 prepared 目录。
    if (cache_dir / "train_patches.npy").exists():
        features = np.load(cache_dir / "train_patches.npy", mmap_mode="r")
        rgb = np.load(cache_dir / "train_rgb_patches.npy", mmap_mode="r")
        train_idx = np.load(cache_dir / "train_labels.npy")
        test_idx = np.load(cache_dir / "test_labels.npy")
        label_shift = int(np.load(cache_dir / "label_shift.npy"))
        if rgb.shape[:2] != features.shape[:2]:
            rgb = resize_rgb_to_match(np.asarray(rgb), features.shape[0], features.shape[1]).astype(np.float16)

        hsi_channels = int(meta.get("hsi_channels", features.shape[2] - 1))
        lidar_channels = int(meta.get("lidar_channels", 1))
        num_classes = int(max(train_idx[:, 0].max(), test_idx[:, 0].max()) - label_shift + 1)
        return {
            "dataset_name": meta["dataset_name"],
            "split_mode": "fixed",
            "hsi_channels": hsi_channels,
            "lidar_channels": lidar_channels,
            "rgb_channels": 3,
            "num_classes": num_classes,
            "features": features,
            "rgb": rgb,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "label_shift": label_shift,
            "meta": meta,
        }

    dataset = {
        "dataset_name": meta["dataset_name"],
        "split_mode": meta["split_mode"],
        "hsi_channels": int(meta["hsi_channels"]),
        "lidar_channels": int(meta["lidar_channels"]),
        "rgb_channels": int(meta["rgb_channels"]),
        "num_classes": int(meta["num_classes"]),
        "features": np.load(cache_dir / "features.npy", mmap_mode="r"),
        "rgb": np.load(cache_dir / "rgb.npy", mmap_mode="r"),
        "meta": meta,
    }
    if meta["split_mode"] == "fixed":
        dataset["train_idx"] = np.load(cache_dir / "train_labels.npy")
        dataset["test_idx"] = np.load(cache_dir / "test_labels.npy")
        dataset["label_shift"] = int(np.load(cache_dir / "label_shift.npy"))
    else:
        dataset["labels"] = np.load(cache_dir / "labels.npy", mmap_mode="r")
    return dataset


def maybe_load_existing_prepared_dir(data_dir, dataset_name):
    meta_path = data_dir / "prepared.meta.json"
    feats_path = data_dir / "train_patches.npy"
    rgb_path = data_dir / "train_rgb_patches.npy"
    train_path = data_dir / "train_labels.npy"
    test_path = data_dir / "test_labels.npy"
    shift_path = data_dir / "label_shift.npy"

    if not all(path.exists() for path in (feats_path, rgb_path, train_path, test_path, shift_path)):
        raise FileNotFoundError(f"{data_dir} is not a compatible prepared directory")

    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    features = np.load(feats_path, mmap_mode="r")
    rgb = np.load(rgb_path, mmap_mode="r")
    if rgb.shape[:2] != features.shape[:2]:
        rgb = resize_rgb_to_match(np.asarray(rgb), features.shape[0], features.shape[1]).astype(np.float16)

    train_idx = np.load(train_path)
    test_idx = np.load(test_path)
    label_shift = int(np.load(shift_path))
    num_classes = int(max(train_idx[:, 0].max(), test_idx[:, 0].max()) - label_shift + 1)
    hsi_channels = int(meta.get("hsi_channels", features.shape[2] - 1))
    lidar_channels = int(meta.get("lidar_channels", 1))

    return {
        "dataset_name": dataset_name,
        "split_mode": "fixed",
        "features": features,
        "rgb": rgb,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "label_shift": label_shift,
        "hsi_channels": hsi_channels,
        "lidar_channels": lidar_channels,
        "rgb_channels": 3,
        "num_classes": num_classes,
        "meta": meta,
    }


def prepare_houston2018(raw_dir, cache_dir):
    # Houston2018 自带官方 train/test mask，因此缓存时直接保存固定划分。
    mat_path = raw_dir / "houston2018.mat"
    data = loadmat(mat_path)

    hsi = ensure_channel_last(data["hsi"], data["hsi"].shape[0] if data["hsi"].shape[0] < data["hsi"].shape[-1] else data["hsi"].shape[-1])
    lidar = ensure_channel_last(data["lidar"], 1)
    rgb = ensure_channel_last(data["rgb"], 3)
    train = ensure_label_array(data["train"], "train")
    test = ensure_label_array(data["test"], "test")

    target_h, target_w = hsi.shape[:2]
    rgb = resize_rgb_to_match(rgb, target_h, target_w)

    feats = np.concatenate([hsi, lidar], axis=2)
    feats_norm = normalize_features(feats).astype(np.float16)
    rgb_norm = normalize_features(rgb).astype(np.float16)
    train_idx = build_sorted_index(train)
    test_idx = build_sorted_index(test)
    label_shift = int(min(train_idx[:, 0].min(), test_idx[:, 0].min()))
    num_classes = int(max(train_idx[:, 0].max(), test_idx[:, 0].max()) - label_shift + 1)

    meta = {
        "dataset_name": "houston2018",
        "split_mode": "fixed",
        "hsi_channels": int(hsi.shape[2]),
        "lidar_channels": int(lidar.shape[2]),
        "rgb_channels": 3,
        "height": int(target_h),
        "width": int(target_w),
        "num_classes": int(num_classes),
        "source": str(mat_path),
    }
    save_cache(
        cache_dir,
        meta,
        {
            "features": feats_norm,
            "rgb": rgb_norm,
            "train_labels": train_idx.astype(np.int32),
            "test_labels": test_idx.astype(np.int32),
            "label_shift": np.array(label_shift, dtype=np.int64),
        },
    )
    return load_cache(cache_dir)


def prepare_szutree(raw_dir, cache_dir, rgb_override=None, label_override=None):
    # SZUTree 先缓存整图，真正的按类采样在每个 iteration 内按 seed 重新完成。
    raw = load_szutree_raw(raw_dir, rgb_override=rgb_override, label_override=label_override)
    hsi = raw["hsi"]
    lidar = raw["lidar"]
    rgb = raw["rgb"]
    labels = raw["labels"]

    feats = np.concatenate([hsi, lidar], axis=2)
    feats_norm = normalize_features(feats).astype(np.float16)
    rgb_norm = normalize_features(rgb).astype(np.float16)
    labels = labels.astype(np.int32, copy=False)
    num_classes = int(labels.max())

    meta = {
        "dataset_name": "szutree",
        "split_mode": "resample",
        "hsi_channels": int(hsi.shape[2]),
        "lidar_channels": int(lidar.shape[2]),
        "rgb_channels": 3,
        "height": int(hsi.shape[0]),
        "width": int(hsi.shape[1]),
        "num_classes": int(num_classes),
        "source": str(raw_dir),
        "szutree_layout": raw["layout"],
        "rgb_source": raw["rgb_source"],
        "label_source": raw["label_source"],
    }
    save_cache(
        cache_dir,
        meta,
        {
            "features": feats_norm,
            "rgb": rgb_norm,
            "labels": labels.astype(np.int32),
        },
    )
    return load_cache(cache_dir)


def resolve_dataset(args):
    # 优先复用现有缓存；如果 data-dir 本身就是 prepared 目录，也直接按缓存读取。
    data_dir = args.data_dir
    if data_dir.is_file():
        raise ValueError(f"--data-dir should point to a directory, but got file: {data_dir}")

    cache_dir = args.cache_dir
    if cache_dir is None:
        if (data_dir / "prepared.meta.json").exists():
            cache_dir = data_dir
        else:
            cache_dir = data_dir / "prepared_msaf3"

    if (cache_dir / "prepared.meta.json").exists():
        dataset = load_cache(cache_dir)
        if dataset["dataset_name"] != args.dataset:
            raise ValueError(f"Cache under {cache_dir} is for {dataset['dataset_name']}, not {args.dataset}")
        return dataset, cache_dir

    if args.dataset == "houston2018":
        try:
            dataset = maybe_load_existing_prepared_dir(data_dir, "houston2018")
            return dataset, cache_dir
        except FileNotFoundError:
            pass
        dataset = prepare_houston2018(data_dir, cache_dir)
        return dataset, cache_dir

    dataset = prepare_szutree(data_dir, cache_dir, rgb_override=args.rgb_file, label_override=args.label_file)
    return dataset, cache_dir


class PreparedTriModalDataset(Dataset):
    # 每个样本由同一中心像素处的 HSI / LiDAR / RGB patch 组成。
    def __init__(self, features, rgb, index_array, label_shift, patch_size, hsi_channels, lidar_channels):
        super().__init__()
        self.features = features
        self.rgb = rgb
        self.index_array = np.asarray(index_array, dtype=np.int32)
        self.label_shift = int(label_shift)
        self.patch_size = int(patch_size)
        self.hsi_channels = int(hsi_channels)
        self.lidar_channels = int(lidar_channels)
        self.pad = self.patch_size // 2
        self.labels = self.index_array[:, 0].astype(np.int64) - self.label_shift

    def __len__(self):
        return len(self.index_array)

    def _extract_patch(self, array, row, col):
        # 边界位置采用零填充，保证所有 patch 尺寸一致。
        height, width, channels = array.shape
        patch = np.zeros((self.patch_size, self.patch_size, channels), dtype=np.float32)

        src_r0 = max(0, row - self.pad)
        src_r1 = min(height, row + self.pad + 1)
        src_c0 = max(0, col - self.pad)
        src_c1 = min(width, col + self.pad + 1)

        dst_r0 = self.pad - (row - src_r0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = self.pad - (col - src_c0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        patch[dst_r0:dst_r1, dst_c0:dst_c1] = array[src_r0:src_r1, src_c0:src_c1]
        return patch

    def __getitem__(self, idx):
        label, row, col = self.index_array[idx]
        row = int(row)
        col = int(col)

        patch = self._extract_patch(self.features, row, col)
        rgb_patch = self._extract_patch(self.rgb, row, col)

        hsi = torch.from_numpy(np.transpose(patch[:, :, : self.hsi_channels], (2, 0, 1)).copy())
        lidar = torch.from_numpy(
            np.transpose(
                patch[:, :, self.hsi_channels : self.hsi_channels + self.lidar_channels],
                (2, 0, 1),
            ).copy()
        )
        rgb = torch.from_numpy(np.transpose(rgb_patch, (2, 0, 1)).copy())
        label_tensor = torch.tensor(int(label) - self.label_shift, dtype=torch.int64)
        return hsi, lidar, rgb, label_tensor


def get_loader_kwargs(args, shuffle):
    # Windows 下 num_workers=0 时不能传 persistent_workers。
    kwargs = {
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
    return kwargs


def evaluate_loader(model, data_loader, criterion):
    # 验证集与测试集都复用这套前向评估逻辑。
    model.eval()
    total_correct = 0
    avg_loss = 0.0

    with torch.no_grad():
        for hsi, lidar, rgb, labels in data_loader:
            hsi = hsi.to(DEVICE)
            lidar = lidar.to(DEVICE)
            rgb = rgb.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(hsi, lidar, rgb)
            loss = criterion(outputs, labels.long())
            avg_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += preds.eq(labels).sum().item()

    avg_loss /= max(1, len(data_loader))
    acc = total_correct / max(1, len(data_loader.dataset))
    return acc, avg_loss


def train_model(model, criterion, train_loader, optimizer, scheduler, epochs, val_loader):
    best_model = copy.deepcopy(model)
    best_val_acc = 0.0
    best_epoch = 1
    early_counter = 0
    epoch_times = []
    train_start = datetime.datetime.now()

    for epoch in range(1, epochs + 1):
        epoch_start = datetime.datetime.now()
        model.train()
        total_correct = 0
        train_avg_loss = 0.0

        for hsi, lidar, rgb, target in train_loader:
            hsi = hsi.to(DEVICE)
            lidar = lidar.to(DEVICE)
            rgb = rgb.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()
            output = model(hsi, lidar, rgb)
            loss = criterion(output, target.long())
            train_avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            total_correct += output.argmax(dim=1).eq(target).sum().item()

        train_acc = total_correct / max(1, len(train_loader.dataset))
        train_avg_loss /= max(1, len(train_loader))
        val_acc, val_loss = evaluate_loader(model, val_loader, criterion)
        scheduler.step()

        epoch_seconds = (datetime.datetime.now() - epoch_start).total_seconds()
        epoch_times.append(epoch_seconds)
        print(
            "epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, epoch time %s"
            % (epoch, train_avg_loss, train_acc, val_loss, val_acc, format_seconds(epoch_seconds))
        )

        # 用验证集准确率选择 best checkpoint，并驱动 early stopping。
        if best_val_acc <= val_acc:
            print(
                "Best_Val_Value changed: from %f to %f;" % (best_val_acc, val_acc),
                end="\t",
            )
            best_epoch = epoch
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            print(
                "Best Classification Accuracy %f, Best Classification loss %f, Best Epoch %d"
                % (best_val_acc, val_loss, best_epoch)
            )
            early_counter = 0
        else:
            threshold_epoch = 100 if epochs > 100 else 50
            if epoch > threshold_epoch:
                early_counter += 1
                print(f"Counter {early_counter} of 20")
                if early_counter > 20:
                    print(
                        "Early stopping with best_val_acc: ",
                        best_val_acc,
                        "at epoch %d: ..." % best_epoch,
                    )
                    break

    train_end = datetime.datetime.now()
    if epoch_times:
        print(
            "||======= Average Epoch Time %s (%d epochs) ======||"
            % (format_seconds(sum(epoch_times) / len(epoch_times)), len(epoch_times))
        )
    print("||======= Train Time for %s ======||" % (train_end - train_start))
    return best_model, (train_end - train_start).total_seconds()


def test_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    y_pred = []
    target_all = []
    test_start = datetime.datetime.now()

    with torch.no_grad():
        for hsi, lidar, rgb, target in test_loader:
            hsi = hsi.to(DEVICE)
            lidar = lidar.to(DEVICE)
            rgb = rgb.to(DEVICE)
            target = target.to(DEVICE)

            output = model(hsi, lidar, rgb)
            loss = criterion(output, target.long())
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            y_pred.extend(pred.cpu().numpy().tolist())
            target_all.extend(target.cpu().numpy().tolist())

    test_loss /= max(1, len(test_loader))
    test_end = datetime.datetime.now()
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / max(1, len(test_loader.dataset)),
        )
    )
    print("||======= Test Time for %s ======||" % (test_end - test_start))
    return (
        float(100.0 * correct / max(1, len(test_loader.dataset))),
        float(test_loss),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(target_all, dtype=np.int64),
        (test_end - test_start).total_seconds(),
    )


def AA_andEachClassAccuracy(conf_mat):
    diag = np.diag(conf_mat)
    row_sum = np.sum(conf_mat, axis=1)
    each_acc = np.nan_to_num(diag / row_sum)
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, target, num_classes):
    classification = classification_report(target, y_pred, labels=np.arange(num_classes), zero_division=0)
    conf_mat = confusion_matrix(target, y_pred, labels=np.arange(num_classes))
    oa = np.trace(conf_mat) / np.sum(conf_mat)
    each_acc, aa = AA_andEachClassAccuracy(conf_mat)
    pe = (conf_mat.sum(axis=0) @ conf_mat.sum(axis=1)) / np.square(np.sum(conf_mat))
    kappa = (oa - pe) / (1 - pe) if pe != 1 else 0.0
    return classification, conf_mat, oa, aa, kappa, each_acc


def write_report(
    path,
    train_sample,
    oa_list,
    aa_list,
    kappa_list,
    element_acc,
    training_time,
    testing_time,
    params,
    macs,
):
    # 报告格式尽量与双模态版本保持一致，便于后续统一比较和制表。
    element_mean = np.mean(element_acc, axis=0)
    element_std = np.std(element_acc, axis=0)

    with open(path, "w", encoding="utf-8") as f:
        f.write("training samples are:" + str(train_sample) + "\n")
        f.write("OAs for each iteration are:" + str(oa_list) + "\n")
        f.write("AAs for each iteration are:" + str(aa_list) + "\n")
        f.write("KAPPAs for each iteration are:" + str(kappa_list) + "\n\n")
        f.write("mean_OA ± std_OA is: " + str(np.mean(oa_list)) + " ± " + str(np.std(oa_list)) + "\n")
        f.write("mean_AA ± std_AA is: " + str(np.mean(aa_list)) + " ± " + str(np.std(aa_list)) + "\n")
        f.write("mean_KAPPA ± std_KAPPA is: " + str(np.mean(kappa_list)) + " ± " + str(np.std(kappa_list)) + "\n\n")
        f.write("Mean of all elements in confusion matrix: " + str(element_mean) + "\n")
        f.write("Standard deviation of all elements in confusion matrix: " + str(element_std) + "\n\n")

        per_class = "Per-class accuracy (%): " + " ".join([f"C{i}:{acc * 100:.2f}" for i, acc in enumerate(element_mean)])
        per_class_std = "Per-class accuracy with std (%): " + " ".join(
            [f"C{i}:{acc * 100:.2f}±{std * 100:.2f}" for i, (acc, std) in enumerate(zip(element_mean, element_std))]
        )
        f.write(per_class + "\n")
        f.write(per_class_std + "\n\n")
        f.write("train time for each iteration are:" + str(training_time) + "\n\n")
        f.write("test time for each iteration are:" + str(testing_time) + "\n\n")
        f.write("all iters acc:" + str(element_acc) + "\n\n")
        f.write("The models parameters and Macs are:" + str(params) + " " + str(macs) + "\n")


def main():
    args = parse_args()
    os.chdir(ROOT_DIR)

    load_start = time.perf_counter()
    dataset, cache_dir = resolve_dataset(args)
    print(f"[time] dataset load/prepare: {format_seconds(time.perf_counter() - load_start)}")
    print(f"[info] using cache/data source: {cache_dir}")

    dataset_name = dataset["dataset_name"]
    hsi_band = dataset["hsi_channels"]
    lidar_band = dataset["lidar_channels"]
    rgb_band = dataset["rgb_channels"]
    num_classes = dataset["num_classes"]
    file_name = {"houston2018": "Houston2018_3mod", "szutree": "SZUTree_3mod"}[dataset_name]

    output_dir = args.output_dir or (BASE_DIR / "results" / file_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_accuracies = []
    average_accuracies = []
    kappa_scores = []
    element_acc = np.zeros((args.itm, num_classes), dtype=np.float64)
    training_time = []
    testing_time = []

    for iteration in range(args.itm):
        iteration_start = time.perf_counter()
        current_seed = args.seed + iteration
        set_seed(current_seed)
        print(f"iteration {iteration + 1}/{args.itm}, seed={current_seed}")

        split_start = time.perf_counter()
        # Houston2018 走固定官方划分；SZUTree 则按当前 seed 重新采样。
        if dataset["split_mode"] == "fixed":
            train_idx = dataset["train_idx"]
            val_idx = train_idx
            test_idx = dataset["test_idx"]
            label_shift = dataset["label_shift"]
        else:
            if args.dataset == "szutree" and args.szutree_split_mode == "percent":
                development_idx, test_idx = split_per_class_percent(
                    dataset["labels"], current_seed, args.train_percent_per_class
                )
            else:
                development_idx, test_idx = split_per_class_count(
                    dataset["labels"], current_seed, args.samples_per_class
                )
            if args.dataset == "szutree":
                train_idx, val_idx = split_index_train_validation(
                    development_idx,
                    current_seed + 10000,
                    args.validation_ratio_within_train,
                )
            else:
                train_idx = development_idx
                val_idx = train_idx
            label_shift = int(
                min(train_idx[:, 0].min(), val_idx[:, 0].min(), test_idx[:, 0].min())
            )
        train_sample = infer_train_sample(train_idx, num_classes)
        print(
            f"[split] train={len(train_idx)}, validation={len(val_idx)}, "
            f"test={len(test_idx)}"
        )
        print(f"[time] split build: {format_seconds(time.perf_counter() - split_start)}")

        dataset_build_start = time.perf_counter()
        # 训练/验证/测试都从同一份整图缓存中按索引实时裁 patch。
        train_dataset = PreparedTriModalDataset(
            dataset["features"],
            dataset["rgb"],
            train_idx,
            label_shift,
            args.patch_size,
            hsi_band,
            lidar_band,
        )
        val_dataset = PreparedTriModalDataset(
            dataset["features"],
            dataset["rgb"],
            val_idx,
            label_shift,
            args.patch_size,
            hsi_band,
            lidar_band,
        )
        test_dataset = PreparedTriModalDataset(
            dataset["features"],
            dataset["rgb"],
            test_idx,
            label_shift,
            args.patch_size,
            hsi_band,
            lidar_band,
        )
        print(f"[time] prepared dataset build: {format_seconds(time.perf_counter() - dataset_build_start)}")

        loader_start = time.perf_counter()
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **get_loader_kwargs(args, True))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, **get_loader_kwargs(args, False))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, **get_loader_kwargs(args, False))
        print(f"[time] dataloader build: {format_seconds(time.perf_counter() - loader_start)}")

        model_build_start = time.perf_counter()
        # 三模态主体：三个 CNN 分支 + 两级 Transformer 融合。
        model = MCF3(
            HSIband=hsi_band,
            lidarband=lidar_band,
            rgbband=rgb_band,
            num_classes=num_classes,
            use_pretrained=not args.no_pretrained,
            use_rgb_pretrained=not args.no_pretrained,
        ).to(DEVICE)

        try:
            summary(model, [(hsi_band, args.patch_size, args.patch_size), (lidar_band, args.patch_size, args.patch_size), (rgb_band, args.patch_size, args.patch_size)])
        except Exception as exc:
            print(f"[warn] torchsummary failed: {exc}")
        print(f"[time] model build+summary: {format_seconds(time.perf_counter() - model_build_start)}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
        criterion = nn.CrossEntropyLoss()

        train_stage_start = time.perf_counter()
        model, train_time = train_model(model, criterion, train_loader, optimizer, scheduler, args.epochs, val_loader)
        print(f"[time] train stage: {format_seconds(time.perf_counter() - train_stage_start)}")

        test_stage_start = time.perf_counter()
        test_acc, test_loss, y_pred, target, test_time = test_model(model, criterion, test_loader)
        print(f"[time] test stage: {format_seconds(time.perf_counter() - test_stage_start)}")

        _, _, oa, aa, kappa, each_acc = reports(y_pred, target, num_classes)

        profile_start = time.perf_counter()
        macs = params = 0.0
        try:
            input_hsi = torch.randn(1, hsi_band, args.patch_size, args.patch_size, device=DEVICE)
            input_lidar = torch.randn(1, lidar_band, args.patch_size, args.patch_size, device=DEVICE)
            input_rgb = torch.randn(1, rgb_band, args.patch_size, args.patch_size, device=DEVICE)
            macs, params = profile(model, inputs=(input_hsi, input_lidar, input_rgb), verbose=False)
            print("params, macs", params, macs)
        except Exception as exc:
            print(f"[warn] thop profile failed: {exc}")
        print(f"[time] thop profile: {format_seconds(time.perf_counter() - profile_start)}")

        overall_accuracies.append(float(oa))
        average_accuracies.append(float(aa))
        kappa_scores.append(float(kappa))
        element_acc[iteration, :] = each_acc
        training_time.append(float(train_time))
        testing_time.append(float(test_time))

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 每一轮都输出当前累计结果，方便中途停止时保留已完成统计。
        report_path = output_dir / f"{dataset_name}_3mod_{len(train_dataset)}_{params}_{current_time}_Report.txt"
        report_start = time.perf_counter()
        write_report(
            report_path,
            train_sample,
            overall_accuracies,
            average_accuracies,
            kappa_scores,
            element_acc[: iteration + 1, :],
            training_time,
            testing_time,
            params,
            macs,
        )
        print(f"[time] report write: {format_seconds(time.perf_counter() - report_start)}")
        print(f"[info] report saved to {report_path}")
        print("test metrics:", test_acc, test_loss)
        print(f"[time] iteration total: {format_seconds(time.perf_counter() - iteration_start)}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
