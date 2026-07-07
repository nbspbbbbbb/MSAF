from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

try:
    import h5py
except ImportError:
    h5py = None


def ensure_channel_last(arr, channels):
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if arr.shape[-1] == channels:
        return arr
    if arr.shape[0] == channels:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Could not align shape {arr.shape} to channel-last with {channels} channels")


def resize_label_map_nearest(labels, target_hw):
    target_h, target_w = target_hw
    if labels.shape == (target_h, target_w):
        return labels
    label_tensor = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(label_tensor, size=(target_h, target_w), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy().astype(labels.dtype)


def resize_rgb_to_match(rgb_hwc, target_h, target_w):
    h, w, c = rgb_hwc.shape
    if c != 3:
        raise ValueError(f"Unexpected RGB channel count: {c}")
    if h == target_h and w == target_w:
        return rgb_hwc

    ratio_h = h // target_h
    ratio_w = w // target_w
    if ratio_h * target_h == h and ratio_w * target_w == w:
        h_crop = target_h * ratio_h
        w_crop = target_w * ratio_w
        rgb_cropped = rgb_hwc[:h_crop, :w_crop, :]
        return rgb_cropped.reshape(target_h, ratio_h, target_w, ratio_w, c).mean(axis=(1, 3))

    rgb_tensor = torch.from_numpy(np.transpose(rgb_hwc, (2, 0, 1)).astype(np.float32)).unsqueeze(0)
    resized = F.interpolate(rgb_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return np.transpose(resized.squeeze(0).numpy(), (1, 2, 0))


def downsample_label_hr_to_lr_mode(label_hr):
    h_hr, w_hr = label_hr.shape
    if h_hr % 2 != 0 or w_hr % 2 != 0:
        raise ValueError(f"Expected even-sized HR labels, got {label_hr.shape}")
    h_lr, w_lr = h_hr // 2, w_hr // 2
    blocks = label_hr.reshape(h_lr, 2, w_lr, 2).transpose(0, 2, 1, 3)
    flat = blocks.reshape(h_lr, w_lr, 4)
    out = np.zeros((h_lr, w_lr), dtype=label_hr.dtype)
    for i in range(h_lr):
        for j in range(w_lr):
            vals = flat[i, j]
            counts = np.bincount(vals.astype(np.int64))
            out[i, j] = counts.argmax()
    return out


def align_lr_label_to_hsi_spatial(labels_lr, hsi_hw):
    h, w = int(hsi_hw[0]), int(hsi_hw[1])
    if labels_lr.shape == (h, w):
        return labels_lr
    if labels_lr.shape == (w, h):
        return np.ascontiguousarray(labels_lr.T)
    return resize_label_map_nearest(labels_lr, hsi_hw)


def build_sorted_index(y):
    coords = np.argwhere(y > 0)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    labels = y[coords[:, 0], coords[:, 1]].astype(np.int32, copy=False)
    height, width = y.shape
    order_key = (
        labels.astype(np.int64) * (height * width)
        + coords[:, 0].astype(np.int64) * width
        + coords[:, 1].astype(np.int64)
    )
    order = np.argsort(order_key)
    return np.column_stack([labels[order], coords[order]]).astype(np.int32, copy=False)


def split_per_class_count(labels, seed, samples_per_class):
    rng = np.random.default_rng(seed)
    train = np.zeros_like(labels, dtype=np.int32)
    test = np.zeros_like(labels, dtype=np.int32)
    num_classes = int(labels.max())
    for cls in range(1, num_classes + 1):
        coords = np.argwhere(labels == cls)
        if coords.size == 0:
            continue
        rng.shuffle(coords)
        n_train = min(samples_per_class, len(coords))
        train_coords = coords[:n_train]
        test_coords = coords[n_train:]
        train[train_coords[:, 0], train_coords[:, 1]] = cls
        test[test_coords[:, 0], test_coords[:, 1]] = cls
    return build_sorted_index(train), build_sorted_index(test)


def split_per_class_percent(labels, seed, train_percent_per_class):
    rng = np.random.default_rng(seed)
    train = np.zeros_like(labels, dtype=np.int32)
    test = np.zeros_like(labels, dtype=np.int32)
    num_classes = int(labels.max())
    for cls in range(1, num_classes + 1):
        coords = np.argwhere(labels == cls)
        if coords.size == 0:
            continue
        rng.shuffle(coords)
        n_train = int(round(len(coords) * train_percent_per_class / 100.0))
        n_train = max(1, min(len(coords), n_train))
        train_coords = coords[:n_train]
        test_coords = coords[n_train:]
        train[train_coords[:, 0], train_coords[:, 1]] = cls
        test[test_coords[:, 0], test_coords[:, 1]] = cls
    return build_sorted_index(train), build_sorted_index(test)


def split_index_train_validation(index_array, seed, validation_ratio=0.1):
    """Split an already sampled development index into disjoint train/val sets.

    For SZUTree the outer split first samples (by default) 1% of every class.
    This function then holds out 10% of that sampled 1% for validation.  It
    operates on ``[label, row, col]`` indices so no pixel from the 99% test set
    can enter checkpoint selection.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    index_array = np.asarray(index_array, dtype=np.int32)
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for label in np.unique(index_array[:, 0]):
        class_rows = index_array[index_array[:, 0] == label]
        order = rng.permutation(len(class_rows))
        if len(class_rows) == 1:
            validation_count = 0
        else:
            validation_count = int(round(len(class_rows) * validation_ratio))
            validation_count = max(1, min(len(class_rows) - 1, validation_count))
        validation_parts.append(class_rows[order[:validation_count]])
        train_parts.append(class_rows[order[validation_count:]])
    train_index = np.concatenate(train_parts, axis=0)
    nonempty_validation = [part for part in validation_parts if len(part) > 0]
    if not nonempty_validation:
        raise ValueError("No validation samples could be created")
    validation_index = np.concatenate(nonempty_validation, axis=0)
    train_index = train_index[rng.permutation(len(train_index))]
    validation_index = validation_index[rng.permutation(len(validation_index))]
    return train_index, validation_index


def choose_rgb_file(data_dir, override=None):
    if override is not None:
        rgb_path = override if override.is_absolute() else data_dir / override
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        return rgb_path
    candidates = sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mat" and "rgb" in p.stem.lower()])
    if not candidates:
        raise FileNotFoundError(
            f"Could not find an RGB .mat file under {data_dir}. Use --rgb-file to specify it explicitly."
        )
    return candidates[0]


def load_mat_dict(path):
    try:
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:
        if h5py is None:
            raise
        with h5py.File(path, "r") as f:
            arrays = {}
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    arrays[k] = np.array(f[k])
            return arrays


def load_rgb_array(path):
    arrays = load_mat_dict(path)
    preferred = ["rgb", "RGB", "image", "Image", "ortho", "Ortho", "data", "Data"]
    for key in preferred:
        if key in arrays:
            arr = np.asarray(arrays[key])
            if arr.ndim == 3 and 3 in arr.shape:
                return ensure_channel_last(arr, 3)
    for value in arrays.values():
        arr = np.asarray(value)
        if arr.ndim == 3 and 3 in arr.shape:
            return ensure_channel_last(arr, 3)
    raise ValueError(f"Could not locate a 3-channel RGB array inside {path}")


def detect_szutree_layout(data_dir):
    r1_files_legacy = [data_dir / "HSI.mat", data_dir / "LiDAR.mat", data_dir / "RGB.mat", data_dir / "label.mat"]
    r1_files_actual = [
        data_dir / "data_band98.mat",
        data_dir / "SZUTreeCHM_R1.mat",
        data_dir / "SZUTreeRGB_R1.mat",
    ]
    if all(path.exists() for path in r1_files_legacy) or all(path.exists() for path in r1_files_actual):
        return "r1"
    r2_files = [data_dir / "data_band98.mat", data_dir / "SZUTreeCHM_R2.mat"]
    if all(path.exists() for path in r2_files):
        return "r2"
    raise FileNotFoundError(f"Could not recognize SZUTree layout under {data_dir}")


def load_szutree_raw(data_dir, rgb_override=None, label_override=None):
    layout = detect_szutree_layout(data_dir)
    if layout == "r1":
        if h5py is None:
            raise ImportError("h5py is required to read SZUTree R1 HSI/LiDAR files")

        hsi_path = data_dir / "HSI.mat"
        if not hsi_path.exists():
            hsi_path = data_dir / "data_band98.mat"
        lidar_path = data_dir / "LiDAR.mat"
        if not lidar_path.exists():
            lidar_path = data_dir / "SZUTreeCHM_R1.mat"
        rgb_path = data_dir / "RGB.mat"
        if not rgb_path.exists():
            rgb_path = data_dir / "SZUTreeRGB_R1.mat"
        label_path = label_override if label_override is not None else data_dir / "label.mat"
        if not label_path.exists():
            candidates = [
                data_dir / "SZUTreeData_R1_typeid_with_labels_5cm.mat",
                data_dir / "Annotations_SZUTreeData_R1" / "SZUTreeData_R1_typeid_with_labels_5cm.mat",
            ]
            for candidate in candidates:
                if candidate.exists():
                    label_path = candidate
                    break

        with h5py.File(hsi_path, "r") as f:
            hsi = np.array(f["hyperspectral_data_98bands"]).transpose(1, 2, 0)
        with h5py.File(lidar_path, "r") as f:
            lidar = np.array(f["chm"])
        rgb = load_rgb_array(rgb_path)
        labels_hr = loadmat(label_path)["data"]
        if lidar.ndim == 2:
            lidar = lidar[:, :, np.newaxis]
        labels_lr = downsample_label_hr_to_lr_mode(labels_hr)
        labels = align_lr_label_to_hsi_spatial(labels_lr, hsi.shape[:2])
        rgb = resize_rgb_to_match(ensure_channel_last(rgb, 3), hsi.shape[0], hsi.shape[1])
        return {
            "layout": layout,
            "hsi": hsi,
            "lidar": lidar,
            "rgb": rgb,
            "labels": labels.astype(np.int32, copy=False),
            "label_source": str(label_path),
            "rgb_source": str(rgb_path),
        }

    if h5py is None:
        raise ImportError("h5py is required to read SZUTree R2 HSI/LiDAR files")
    with h5py.File(data_dir / "data_band98.mat", "r") as f:
        hsi = np.array(f["hyperspectral_data_98bands"]).transpose(1, 2, 0)
    with h5py.File(data_dir / "SZUTreeCHM_R2.mat", "r") as f:
        lidar = np.array(f["chm"])
    label_path = label_override if label_override is not None else data_dir / "SZUTreeData_R2_typeid_with_labels.mat"
    rgb_path = choose_rgb_file(data_dir, rgb_override)
    labels = loadmat(label_path)["data"].T
    if lidar.ndim == 2:
        lidar = lidar[:, :, np.newaxis]
    labels = resize_label_map_nearest(labels, hsi.shape[:2])
    rgb = resize_rgb_to_match(load_rgb_array(rgb_path), hsi.shape[0], hsi.shape[1])
    return {
        "layout": layout,
        "hsi": hsi,
        "lidar": lidar,
        "rgb": rgb,
        "labels": labels.astype(np.int32, copy=False),
        "label_source": str(label_path),
        "rgb_source": str(rgb_path),
    }
