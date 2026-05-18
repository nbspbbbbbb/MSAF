import datetime
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn import preprocessing
from thop import profile
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset

try:
    import h5py
except ImportError:
    h5py = None

import record
from util_torch import Multidata, createPatches, infer_allmap, random_sample, reports, test, train

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "mcf"))
from MCF_demo_1 import MCF


cudnn.deterministic = True
cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchTensorDataset(TensorDataset):
    def __init__(self, hsi, lidar, labels):
        super().__init__(hsi, lidar, labels)
        self.labels = labels.cpu().numpy()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "datasets")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--itm", type=int, default=1)
    parser.add_argument("--dataset", choices=["auto", "houston2013", "muufl", "trento", "szutree"], default="auto")
    return parser.parse_args()


def _load_houston_dataset(data_dir):
    label_path = data_dir / "DFC2013_gt.mat"
    hsi_path = data_dir / "houston_data.mat"
    hsi_alt_path = data_dir / "Houston2013_HSI.mat"
    dsm_alt_path = data_dir / "Houston2013_DSM.mat"
    tr_alt_path = data_dir / "Houston2013_TR.mat"
    te_alt_path = data_dir / "Houston2013_TE.mat"

    default_layout_exists = label_path.exists() and hsi_path.exists()
    alt_layout_exists = all(path.exists() for path in (hsi_alt_path, dsm_alt_path, tr_alt_path, te_alt_path))

    if default_layout_exists:
        labels = loadmat(label_path)["DFC2013_gt"]
        data_hsi = np.array(loadmat(hsi_path)["hsi"])
        data_lidar = np.array(loadmat(hsi_path)["lidar"])
    elif alt_layout_exists:
        train_labels = loadmat(tr_alt_path)["TR_map"]
        test_labels = loadmat(te_alt_path)["TE_map"]
        labels = train_labels + test_labels
        data_hsi = np.array(loadmat(hsi_alt_path)["HSI"])
        data_lidar = np.array(loadmat(dsm_alt_path)["DSM"])
    else:
        expected_files = [str(path) for path in (label_path, hsi_path)]
        alt_files = [str(path) for path in (hsi_alt_path, dsm_alt_path, tr_alt_path, te_alt_path)]
        raise FileNotFoundError(
            "Missing dataset files. Expected either: "
            + ", ".join(expected_files)
            + " or "
            + ", ".join(alt_files)
            + ". Pass --data-dir to the folder that contains your Houston2013 .mat files."
        )

    data_lidar = np.reshape(data_lidar, (data_lidar.shape[0], data_lidar.shape[1], 1))
    return {
        "layout": "full",
        "dataset_name": "houston2013",
        "labels": labels,
        "data_hsi": data_hsi,
        "data_lidar": data_lidar,
    }


def _load_muufl_dataset(data_dir):
    split_files = [
        data_dir / "HSI_Tr.mat",
        data_dir / "HSI_Te.mat",
        data_dir / "LIDAR_Tr.mat",
        data_dir / "LIDAR_Te.mat",
        data_dir / "TrLabel.mat",
        data_dir / "TeLabel.mat",
    ]
    full_file = data_dir / "muufl_gulfport_campus_1_hsi_220_label.mat"

    if all(path.exists() for path in split_files):
        train_hsi = loadmat(split_files[0])["Data"]
        test_hsi = loadmat(split_files[1])["Data"]
        train_lidar = loadmat(split_files[2])["Data"]
        test_lidar = loadmat(split_files[3])["Data"]
        train_labels = loadmat(split_files[4])["Data"].reshape(-1)
        test_labels = loadmat(split_files[5])["Data"].reshape(-1)
        return {
            "layout": "patch",
            "dataset_name": "muufl",
            "train_hsi": train_hsi,
            "test_hsi": test_hsi,
            "train_lidar": train_lidar,
            "test_lidar": test_lidar,
            "train_labels": train_labels,
            "test_labels": test_labels,
        }

    if full_file.exists():
        obj = loadmat(full_file)["hsi"][0, 0]
        data_hsi = np.array(obj["Data"])
        data_lidar = np.array(obj["Lidar"][0, 0][0, 0]["z"])
        labels = np.array(obj["sceneLabels"][0, 0]["labels"])
        return {
            "layout": "full",
            "dataset_name": "muufl",
            "labels": labels,
            "data_hsi": data_hsi,
            "data_lidar": data_lidar,
        }

    raise FileNotFoundError(f"Could not recognize MUUFL dataset layout under {data_dir}")


def _load_trento_dataset(data_dir):
    split_files = [
        data_dir / "HSI_Tr.mat",
        data_dir / "HSI_Te.mat",
        data_dir / "LIDAR_Tr.mat",
        data_dir / "LIDAR_Te.mat",
        data_dir / "TrLabel.mat",
        data_dir / "TeLabel.mat",
    ]
    full_hsi = data_dir / "Italy_hsi.mat"
    full_lidar = data_dir / "Italy_lidar.mat"
    full_label = data_dir / "allgrd.mat"

    if all(path.exists() for path in split_files):
        train_hsi = loadmat(split_files[0])["Data"]
        test_hsi = loadmat(split_files[1])["Data"]
        train_lidar = loadmat(split_files[2])["Data"]
        test_lidar = loadmat(split_files[3])["Data"]
        train_labels = loadmat(split_files[4])["Data"].reshape(-1)
        test_labels = loadmat(split_files[5])["Data"].reshape(-1)
        return {
            "layout": "patch",
            "dataset_name": "trento",
            "train_hsi": train_hsi,
            "test_hsi": test_hsi,
            "train_lidar": train_lidar,
            "test_lidar": test_lidar,
            "train_labels": train_labels,
            "test_labels": test_labels,
        }

    if full_hsi.exists() and full_lidar.exists() and full_label.exists():
        data_hsi = np.array(loadmat(full_hsi)["data"])
        data_lidar = np.array(loadmat(full_lidar)["data"])
        labels = np.array(loadmat(full_label)["mask_test"])
        return {
            "layout": "full",
            "dataset_name": "trento",
            "labels": labels,
            "data_hsi": data_hsi,
            "data_lidar": data_lidar,
        }

    raise FileNotFoundError(f"Could not recognize Trento dataset layout under {data_dir}")


def _resize_label_map(labels, target_hw):
    target_h, target_w = target_hw
    if labels.shape == (target_h, target_w):
        return labels
    label_tensor = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(label_tensor, size=(target_h, target_w), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy().astype(labels.dtype)


def _load_szutree_dataset(data_dir):
    if h5py is None:
        raise ImportError("h5py is required to read SZUTreeData2.0 v7.3 .mat files")

    hsi_path = data_dir / "data_band98.mat"
    chm_path = data_dir / "SZUTreeCHM_R2.mat"
    typeid_path = data_dir / "SZUTreeData_R2_typeid_with_labels.mat"

    if not (hsi_path.exists() and chm_path.exists() and typeid_path.exists()):
        raise FileNotFoundError(f"Could not recognize SZUTree dataset layout under {data_dir}")

    with h5py.File(hsi_path, "r") as hsi_file:
        data_hsi = np.array(hsi_file["hyperspectral_data_98bands"]).transpose(1, 2, 0)
    with h5py.File(chm_path, "r") as chm_file:
        data_lidar = np.array(chm_file["chm"])
    labels = loadmat(typeid_path)["data"]

    if data_lidar.ndim == 2:
        data_lidar = data_lidar[:, :, np.newaxis]

    labels = labels.T
    labels = _resize_label_map(labels, data_hsi.shape[:2])

    return {
        "layout": "full",
        "dataset_name": "szutree",
        "labels": labels,
        "data_hsi": data_hsi,
        "data_lidar": data_lidar,
    }


def load_dataset(data_dir, dataset_name):
    if dataset_name == "houston2013":
        return _load_houston_dataset(data_dir)
    if dataset_name == "muufl":
        return _load_muufl_dataset(data_dir)
    if dataset_name == "trento":
        return _load_trento_dataset(data_dir)
    if dataset_name == "szutree":
        return _load_szutree_dataset(data_dir)

    loaders = (_load_houston_dataset, _load_muufl_dataset, _load_trento_dataset, _load_szutree_dataset)
    for loader in loaders:
        try:
            return loader(data_dir)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Could not infer dataset type from {data_dir}")


def build_splits(labels, data_lidar, train_sample, validate_sample, window_size):
    _, patches_labels = createPatches(data_lidar, labels, windowSize=window_size)
    patches_labels = patches_labels.astype(np.int32)
    train_index, val_index, test_index = random_sample(train_sample, validate_sample, patches_labels)

    train_gt = np.zeros_like(labels).reshape(np.prod(labels.shape[:2]), order="F")
    label_list = labels.reshape(np.prod(labels.shape[:2]), order="F")
    train_gt[train_index] = label_list[train_index]
    train_gt = train_gt.reshape((labels.shape[0], labels.shape[1]), order="F")

    val_gt = np.zeros_like(labels).reshape(np.prod(labels.shape[:2]), order="F")
    val_gt[val_index] = label_list[val_index]
    val_gt = val_gt.reshape((labels.shape[0], labels.shape[1]), order="F")

    test_gt = np.zeros_like(labels).reshape(np.prod(labels.shape[:2]), order="F")
    test_gt[test_index] = label_list[test_index]
    test_gt = test_gt.reshape((labels.shape[0], labels.shape[1]), order="F")

    return train_gt, val_gt, test_gt, len(train_index)


def normalize_inputs(data_hsi_raw, data_lidar_raw):
    data_hsi = data_hsi_raw.reshape(np.prod(data_hsi_raw.shape[:2]), np.prod(data_hsi_raw.shape[2:]))
    data_hsi = preprocessing.scale(data_hsi)
    data_hsi = data_hsi.reshape(data_hsi_raw.shape[0], data_hsi_raw.shape[1], data_hsi_raw.shape[2])

    data_lidar = data_lidar_raw.reshape(np.prod(data_lidar_raw.shape[:2]), np.prod(data_lidar_raw.shape[2:]))
    data_lidar = preprocessing.scale(data_lidar)
    data_lidar = data_lidar.reshape(data_lidar_raw.shape[0], data_lidar_raw.shape[1], data_lidar_raw.shape[2])
    return data_hsi, data_lidar


def create_patch_tensor_datasets(train_hsi, train_lidar, train_labels, test_hsi, test_lidar, test_labels):
    train_hsi = torch.from_numpy(np.asarray(train_hsi.transpose(0, 3, 1, 2), dtype=np.float32))
    train_lidar = torch.from_numpy(np.asarray(train_lidar.transpose(0, 3, 1, 2), dtype=np.float32))
    test_hsi = torch.from_numpy(np.asarray(test_hsi.transpose(0, 3, 1, 2), dtype=np.float32))
    test_lidar = torch.from_numpy(np.asarray(test_lidar.transpose(0, 3, 1, 2), dtype=np.float32))

    train_labels = torch.from_numpy(np.asarray(train_labels - 1, dtype=np.int64))
    test_labels = torch.from_numpy(np.asarray(test_labels - 1, dtype=np.int64))

    train_dataset = PatchTensorDataset(train_hsi, train_lidar, train_labels)
    val_dataset = PatchTensorDataset(train_hsi, train_lidar, train_labels)
    test_dataset = PatchTensorDataset(test_hsi, test_lidar, test_labels)
    return train_dataset, val_dataset, test_dataset


def create_loaders(data_hsi, data_lidar, train_gt, val_gt, test_gt, window_size, batch_size, include_all_map=True):
    train_dataset = Multidata(data_hsi, data_lidar, train_gt, window_size)
    val_dataset = Multidata(data_hsi, data_lidar, val_gt, window_size)
    test_dataset = Multidata(data_hsi, data_lidar, test_gt, window_size)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    all_map_loader = None
    if include_all_map:
        y_temp = np.ones_like(train_gt)
        all_dataset = Multidata(data_hsi, data_lidar, y_temp, window_size)
        all_map_loader = torch.utils.data.DataLoader(all_dataset, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, all_map_loader


def main():
    args = parse_args()
    os.chdir(BASE_DIR)

    dataset = load_dataset(args.data_dir, args.dataset)
    dataset_name = dataset["dataset_name"]
    layout = dataset["layout"]

    itm = args.itm

    window_size = 11
    batch_size = 16
    epochs = 300
    file_name = {"houston2013": "mcf", "muufl": "MUUFL", "trento": "TRENTO", "szutree": "SZUTree"}[dataset_name]

    if dataset_name == "houston2013":
        train_sample = (38, 38, 21, 37, 37, 10, 38, 37, 38, 37, 37, 37, 14, 13, 18)
        validate_sample = (38, 38, 21, 37, 37, 10, 38, 37, 38, 37, 37, 37, 14, 13, 18)
        window_size = 11
        batch_size = 16
    elif dataset_name == "muufl":
        train_sample = (1162, 214, 344, 91, 334, 24, 112, 312, 70, 30, 30)
        validate_sample = (1162, 214, 344, 91, 334, 24, 112, 312, 70, 30, 30)
        window_size = 11
        batch_size = 16
    elif dataset_name == "szutree":
        train_sample = tuple([50] * 21)
        validate_sample = tuple([50] * 21)
        window_size = 11
        batch_size = 16
    else:
        train_sample = (50, 50, 50, 50, 50, 50)
        validate_sample = (50, 50, 50, 50, 50, 50)
        window_size = 11
        batch_size = 16

    if layout == "full":
        labels = dataset["labels"]
        data_hsi_raw = dataset["data_hsi"]
        data_lidar_raw = dataset["data_lidar"]
        num_classes = int(np.max(labels))
        hsi_band = data_hsi_raw.shape[2]
        lidar_band = data_lidar_raw.shape[2]
        data_hsi, data_lidar = normalize_inputs(data_hsi_raw, data_lidar_raw)
    else:
        num_classes = int(max(np.max(dataset["train_labels"]), np.max(dataset["test_labels"])))
        hsi_band = dataset["train_hsi"].shape[3]
        lidar_band = dataset["train_lidar"].shape[3]
        labels = None
        data_hsi = None
        data_lidar = None

    os.makedirs(BASE_DIR / file_name, exist_ok=True)

    kappa_scores = []
    overall_accuracies = []
    average_accuracies = []
    element_acc = np.zeros((itm, num_classes))
    training_time = []
    testing_time = []
    infer_time = []

    for iteration in range(itm):
        current_seed = args.seed + iteration
        set_seed(current_seed)
        print(f"iteration {iteration + 1}/{itm}, seed={current_seed}")

        if layout == "full":
            train_gt, val_gt, test_gt, train_size = build_splits(
                labels,
                data_lidar_raw,
                train_sample,
                validate_sample,
                window_size,
            )
            train_loader, val_loader, test_loader, all_map_loader = create_loaders(
                data_hsi,
                data_lidar,
                train_gt,
                val_gt,
                test_gt,
                window_size,
                batch_size,
                include_all_map=(dataset_name != "szutree"),
            )
        else:
            train_dataset, val_dataset, test_dataset = create_patch_tensor_datasets(
                dataset["train_hsi"],
                dataset["train_lidar"],
                dataset["train_labels"],
                dataset["test_hsi"],
                dataset["test_lidar"],
                dataset["test_labels"],
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            all_map_loader = None
            train_size = len(train_dataset)

        model = MCF(
            hsi_band,
            lidar_band,
            num_classes,
            use_pretrained=not args.no_pretrained,
        ).cuda()
        summary(model, [(hsi_band, window_size, window_size), (lidar_band, window_size, window_size)])

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
        loss_func = nn.CrossEntropyLoss()

        model, train_time = train(
            model,
            loss_func,
            DEVICE,
            train_loader,
            optimizer,
            scheduler,
            epochs,
            val_loader,
            iteration,
        )
        test_acc_temp, test_loss_temp, y_pred, target, test_time = test(model, DEVICE, test_loader)
        oa, aa, kappa, each_acc, accuracy_matrix = reports(y_pred, target)
        if all_map_loader is not None:
            time_infer = infer_allmap(model, DEVICE, dataset_name, file_name, labels, all_map_loader, iteration)
        else:
            time_infer = 0.0

        input_hsi = torch.randn(1, hsi_band, window_size, window_size).cuda()
        input_lidar = torch.randn(1, lidar_band, window_size, window_size).cuda()
        macs, params = profile(model, inputs=(input_hsi, input_lidar))
        print("params, macs", params, macs)

        training_time.append(train_time)
        testing_time.append(test_time)
        infer_time.append(time_infer)
        kappa_scores.append(kappa)
        overall_accuracies.append(oa)
        average_accuracies.append(aa)
        element_acc[iteration, :] = each_acc

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = (
            str(BASE_DIR / file_name)
            + "/"
            + dataset_name
            + "_"
            + str(train_size)
            + "_"
            + str(params)
            + str(current_time)
            + "_Report.txt"
        )
        record.record_output(
            overall_accuracies,
            average_accuracies,
            kappa_scores,
            element_acc,
            training_time,
            testing_time,
            infer_time,
            macs,
            params,
            train_sample,
            report_path,
        )

        print("final test results :", accuracy_matrix)
        print("test metrics:", test_acc_temp, test_loss_temp)

        del model, input_hsi, input_lidar


if __name__ == "__main__":
    main()
