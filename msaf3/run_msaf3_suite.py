# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = BASE_DIR / "train_msaf3.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--itm", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--szutree-split-mode", choices=["count", "percent"], default="percent")
    parser.add_argument("--train-percent-per-class", type=float, default=1.0)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--jobs",
        nargs="+",
        choices=["szutree_r1", "szutree_r2", "houston2018"],
        default=["szutree_r1", "szutree_r2", "houston2018"],
    )
    parser.add_argument("--auto-shutdown", action="store_true")
    parser.add_argument("--shutdown-delay", type=int, default=60)
    return parser.parse_args()


def build_jobs(args):
    results_root = BASE_DIR / "results" / "suite_runs"
    all_jobs = [
        {
            "name": "szutree_r1",
            "dataset": "szutree",
            "data_dir": Path(r"D:\DATA_3\SZUTreeData2.0\SZUTreeData_R1_2.0"),
            "output_dir": results_root / "szutree_r1",
        },
        {
            "name": "szutree_r2",
            "dataset": "szutree",
            "data_dir": Path(r"D:\DATA_3\SZUTreeData2.0\SZUTreeData_R2_2.0"),
            "output_dir": results_root / "szutree_r2",
        },
        {
            "name": "houston2018",
            "dataset": "houston2018",
            "data_dir": Path(r"D:\DATA_3\Houston2018"),
            "output_dir": results_root / "houston2018",
        },
    ]
    selected = set(args.jobs)
    return [job for job in all_jobs if job["name"] in selected]


def build_command(job, args):
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        job["dataset"],
        "--data-dir",
        str(job["data_dir"]),
        "--output-dir",
        str(job["output_dir"]),
        "--seed",
        str(args.seed),
        "--itm",
        str(args.itm),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--patch-size",
        str(args.patch_size),
        "--num-workers",
        str(args.num_workers),
        "--szutree-split-mode",
        args.szutree_split_mode,
        "--train-percent-per-class",
        str(args.train_percent_per_class),
        "--pin-memory",
    ]
    if args.num_workers > 0:
        cmd.append("--persistent-workers")
    if args.szutree_split_mode == "count":
        cmd.extend(
            [
                "--samples-per-class",
                str(args.samples_per_class),
            ]
        )
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    return cmd


def run_job(job, args):
    cmd = build_command(job, args)
    print(f"\n=== RUN {job['name']} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def shutdown_windows(delay_seconds):
    subprocess.run(["shutdown", "/s", "/t", str(delay_seconds)], check=True)


def main():
    args = parse_args()
    jobs = build_jobs(args)

    for job in jobs:
        run_job(job, args)

    print("\nAll jobs completed.")
    if args.auto_shutdown:
        print(f"Scheduling shutdown in {args.shutdown_delay} seconds.")
        shutdown_windows(args.shutdown_delay)


if __name__ == "__main__":
    main()
