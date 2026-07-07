# -*- coding: utf-8 -*-
"""Stage-1 experiment: add uncontaminated unimodal auxiliary heads to MSAF3.

This runner deliberately lives outside the original msaf3/ and diagnosis/
sources.  It reuses their data/training utilities, but replaces the diagnostic
model at runtime with a variant whose auxiliary heads branch off before the
first cross-modal Transformer fusion.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSIS_DIR = ROOT / "diagnosis"
MSAF3_DIR = ROOT / "msaf3"
for path in (DIAGNOSIS_DIR, MSAF3_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_trimodal_with_aux_heads as diagnostic_runner  # noqa: E402
import diagnosis_utils  # noqa: E402
from MSAF_3 import HSI_CNN, Lidar_CNN, MVT3, RGB_CNN  # noqa: E402


_PADDED_ARRAY_CACHE = {}
_MATERIALIZED_PATCH_CACHE = {}


def _cached_zero_pad(array, pad):
    filename = str(getattr(array, "filename", ""))
    key = (filename or id(array), int(pad))
    if key not in _PADDED_ARRAY_CACHE:
        _PADDED_ARRAY_CACHE[key] = np.pad(
            np.asarray(array),
            ((pad, pad), (pad, pad), (0, 0)),
            mode="constant",
        )
    return _PADDED_ARRAY_CACHE[key]


def _materialize_patches(features, rgb, index_array, label_shift, patch_size, hsi_channels, lidar_channels):
    feature_name = str(getattr(features, "filename", "")) or id(features)
    rgb_name = str(getattr(rgb, "filename", "")) or id(rgb)
    index_ptr = int(np.asarray(index_array).__array_interface__["data"][0])
    key = (
        feature_name,
        rgb_name,
        index_ptr,
        len(index_array),
        int(label_shift),
        int(patch_size),
        int(hsi_channels),
        int(lidar_channels),
    )
    if key in _MATERIALIZED_PATCH_CACHE:
        return _MATERIALIZED_PATCH_CACHE[key]

    index_array = np.asarray(index_array, dtype=np.int32)
    patch_size = int(patch_size)
    pad = patch_size // 2
    padded_features = _cached_zero_pad(features, pad)
    padded_rgb = _cached_zero_pad(rgb, pad)
    sample_count = len(index_array)

    hsi_data = np.empty((sample_count, hsi_channels, patch_size, patch_size), dtype=np.float32)
    lidar_data = np.empty((sample_count, lidar_channels, patch_size, patch_size), dtype=np.float32)
    rgb_data = np.empty((sample_count, 3, patch_size, patch_size), dtype=np.float32)
    offsets = np.arange(patch_size, dtype=np.int64)

    for start in range(0, sample_count, 512):
        end = min(sample_count, start + 512)
        chunk = index_array[start:end]
        rows = chunk[:, 1].astype(np.int64)[:, None, None] + offsets[None, :, None]
        cols = chunk[:, 2].astype(np.int64)[:, None, None] + offsets[None, None, :]
        feature_patch = padded_features[rows, cols]
        rgb_patch = padded_rgb[rows, cols]
        hsi_data[start:end] = np.transpose(feature_patch[:, :, :, :hsi_channels], (0, 3, 1, 2))
        lidar_data[start:end] = np.transpose(
            feature_patch[:, :, :, hsi_channels : hsi_channels + lidar_channels],
            (0, 3, 1, 2),
        )
        rgb_data[start:end] = np.transpose(rgb_patch, (0, 3, 1, 2))

    labels = index_array[:, 0].astype(np.int64) - int(label_shift)
    tensors = (
        torch.from_numpy(hsi_data),
        torch.from_numpy(lidar_data),
        torch.from_numpy(rgb_data),
        torch.from_numpy(labels.copy()),
    )
    _MATERIALIZED_PATCH_CACHE[key] = tensors
    return tensors


class MaterializedTriModalDataset(Dataset):
    """Equivalent dataset that extracts each patch once and reuses it every epoch."""

    def __init__(self, features, rgb, index_array, label_shift, patch_size, hsi_channels, lidar_channels):
        super().__init__()
        self.tensors = _materialize_patches(
            features,
            rgb,
            index_array,
            label_shift,
            patch_size,
            hsi_channels,
            lidar_channels,
        )
        self.labels = self.tensors[3].numpy()

    def __len__(self):
        return len(self.tensors[3])

    def __getitem__(self, idx):
        return tuple(tensor[idx] for tensor in self.tensors)


_ORIGINAL_BUILD_SPLIT = diagnosis_utils.build_split


def _stratified_limit(index_array, samples_per_class, seed):
    if samples_per_class <= 0:
        return index_array
    rng = np.random.RandomState(seed)
    selected = []
    for label in np.unique(index_array[:, 0]):
        candidates = np.flatnonzero(index_array[:, 0] == label)
        count = min(int(samples_per_class), len(candidates))
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.append(index_array[chosen])
    limited = np.concatenate(selected, axis=0)
    return limited[rng.permutation(len(limited))]


def pilot_build_split(dataset, args):
    train_idx, test_idx, label_shift = _ORIGINAL_BUILD_SPLIT(dataset, args)
    train_per_class = int(os.environ.get("STEP1_TRAIN_PER_CLASS", "0"))
    test_per_class = int(os.environ.get("STEP1_TEST_PER_CLASS", "0"))
    train_idx = _stratified_limit(train_idx, train_per_class, args.seed)
    test_idx = _stratified_limit(test_idx, test_per_class, args.seed + 10000)
    return train_idx, test_idx, label_shift


class PureAuxMSAF3(nn.Module):
    """MSAF3 baseline plus three pre-fusion unimodal classification heads."""

    def __init__(self, hsi_band, lidar_band, rgb_band, num_classes, use_pretrained=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((5, 5))

        self.image_encoder = HSI_CNN(hsi_band, use_pretrained=use_pretrained)
        self.lidar_encoder = Lidar_CNN(lidar_band, use_pretrained=False)
        self.rgb_encoder = RGB_CNN(rgb_band, use_pretrained=use_pretrained)

        self.transformer1 = MVT3(16, 4, 16, 4, 2, 5, 5, 0.1, 0.1, 0.1)
        self.transformer2 = MVT3(24, 4, 24, 4, 2, 5, 5, 0.1, 0.1, 0.1)

        self.fusion_classifier = nn.Linear(24, num_classes)
        # MobileNetV3-Small features[1] outputs 16 channels.  These heads are
        # intentionally attached before transformer1, so each prediction only
        # sees its own modality.
        self.hsi_aux_classifier = nn.Linear(16, num_classes)
        self.lidar_aux_classifier = nn.Linear(16, num_classes)
        self.rgb_aux_classifier = nn.Linear(16, num_classes)

        for head in (
            self.fusion_classifier,
            self.hsi_aux_classifier,
            self.lidar_aux_classifier,
            self.rgb_aux_classifier,
        ):
            nn.init.xavier_uniform_(head.weight)
            nn.init.normal_(head.bias, std=1e-6)

    @staticmethod
    def _pool_feature(x):
        return torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)

    def forward(self, hsi, lidar, rgb):
        batch_size, _, height, width = hsi.shape

        hsi_stage0 = self.image_encoder._model.features[0](hsi)
        hsi_stage1_raw = self.image_encoder._model.features[1](hsi_stage0)
        lidar_stage0 = self.lidar_encoder._model.features[0](lidar)
        lidar_stage1_raw = self.lidar_encoder._model.features[1](lidar_stage0)
        rgb_stage0 = self.rgb_encoder._model.features[0](rgb)
        rgb_stage1_raw = self.rgb_encoder._model.features[1](rgb_stage0)

        # Pure, pre-fusion predictions used by the future conflict module.
        feat_hsi = self._pool_feature(hsi_stage1_raw)
        feat_lidar = self._pool_feature(lidar_stage1_raw)
        feat_rgb = self._pool_feature(rgb_stage1_raw)

        # Original two-stage MSAF3 fusion path remains unchanged.
        hsi_pool1 = self.avgpool(hsi_stage1_raw)
        lidar_pool1 = self.avgpool(lidar_stage1_raw)
        rgb_pool1 = self.avgpool(rgb_stage1_raw)
        hsi_fuse1, lidar_fuse1, rgb_fuse1 = self.transformer1(hsi_pool1, lidar_pool1, rgb_pool1)

        hsi_stage1 = hsi_stage1_raw + F.interpolate(hsi_fuse1, size=[height, width], mode="bilinear")
        lidar_stage1 = lidar_stage1_raw + F.interpolate(lidar_fuse1, size=[height, width], mode="bilinear")
        rgb_stage1 = rgb_stage1_raw + F.interpolate(rgb_fuse1, size=[height, width], mode="bilinear")

        hsi_stage2 = self.image_encoder._model.features[2](hsi_stage1)
        lidar_stage2 = self.lidar_encoder._model.features[2](lidar_stage1)
        rgb_stage2 = self.rgb_encoder._model.features[2](rgb_stage1)

        hsi_pool2 = self.avgpool_2(hsi_stage2)
        lidar_pool2 = self.avgpool_2(lidar_stage2)
        rgb_pool2 = self.avgpool_2(rgb_stage2)
        hsi_fuse2, lidar_fuse2, rgb_fuse2 = self.transformer2(hsi_pool2, lidar_pool2, rgb_pool2)

        hsi_final = self._pool_feature(hsi_pool2 + hsi_fuse2).view(batch_size, 1, -1)
        lidar_final = self._pool_feature(lidar_pool2 + lidar_fuse2).view(batch_size, 1, -1)
        rgb_final = self._pool_feature(rgb_pool2 + rgb_fuse2).view(batch_size, 1, -1)
        feat_fusion = torch.sum(torch.cat([hsi_final, lidar_final, rgb_final], dim=1), dim=1)

        return {
            "logits_fusion": self.fusion_classifier(feat_fusion),
            "logits_hsi_aux": self.hsi_aux_classifier(feat_hsi),
            "logits_lidar_aux": self.lidar_aux_classifier(feat_lidar),
            "logits_rgb_aux": self.rgb_aux_classifier(feat_rgb),
            "feat_hsi": feat_hsi,
            "feat_lidar": feat_lidar,
            "feat_rgb": feat_rgb,
            "feat_fusion": feat_fusion,
        }

    def diagnostic_params(self, modality):
        if modality == "HSI":
            return list(self.image_encoder._model.features[1].parameters())
        if modality == "LiDAR":
            return list(self.lidar_encoder._model.features[1].parameters())
        if modality == "RGB":
            return list(self.rgb_encoder._model.features[1].parameters())
        raise ValueError(modality)


_ORIGINAL_TRAIN = diagnostic_runner.train


def train_and_save_checkpoint(model, train_loader, val_loader, criterion, args, output_dir):
    train_seconds, best_val = _ORIGINAL_TRAIN(
        model, train_loader, val_loader, criterion, args, output_dir
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_OA": float(best_val),
            "epochs": int(args.epochs),
            "aux_loss_weight": float(args.aux_loss_weight),
            "seed": int(args.seed),
        },
        output_dir / "best_model.pt",
    )
    return train_seconds, best_val


if __name__ == "__main__":
    diagnosis_utils.PreparedTriModalDataset = MaterializedTriModalDataset
    diagnosis_utils.build_split = pilot_build_split
    diagnostic_runner.DiagnosticMSAF3 = PureAuxMSAF3
    diagnostic_runner.train = train_and_save_checkpoint
    diagnostic_runner.main()
