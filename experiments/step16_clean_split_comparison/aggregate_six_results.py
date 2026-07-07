# -*- coding: utf-8 -*-
"""Aggregate the six clean-split MSAF3 workbooks into one comparison workbook."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTPUT = HERE / "MSAF3_six_experiments_comparison.xlsx"

DATASETS = [
    ("Houston2018", "houston2018"),
    ("SZUTree R1", "szutree_r1"),
    ("SZUTree R2", "szutree_r2"),
]
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SUBHEADER_FILL = PatternFill("solid", fgColor="D9EAF7")
WHITE_BOLD = Font(color="FFFFFF", bold=True)


def read_run(dataset_key: str, method: str) -> dict:
    directory = RESULTS / f"{dataset_key}_{method}_seed4"
    summary_path = directory / "final_summary.json"
    log_path = directory / "epoch_log.csv"
    excel_paths = list(directory.glob("*.xlsx"))
    if not summary_path.exists() or not log_path.exists() or len(excel_paths) != 1:
        raise FileNotFoundError(
            f"Incomplete run under {directory}: summary={summary_path.exists()}, "
            f"log={log_path.exists()}, Excel files={len(excel_paths)}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with log_path.open(encoding="utf-8-sig", newline="") as handle:
        raw_epoch_rows = list(csv.DictReader(handle))
    # A process interruption can leave an already logged epoch to be rerun.
    # Keep the last complete record for each epoch so curves and timing use
    # exactly epochs 1..300 rather than double-counting the resumed epoch.
    rows_by_epoch = {}
    for row in raw_epoch_rows:
        rows_by_epoch[int(row["epoch"])] = row
    epoch_rows = [rows_by_epoch[index] for index in sorted(rows_by_epoch)]
    if len(epoch_rows) != 300:
        raise RuntimeError(
            f"Expected 300 unique epoch rows for {directory}, got {len(epoch_rows)}"
        )
    actual_training_seconds = sum(float(row["seconds"]) for row in epoch_rows)

    source_workbook = load_workbook(excel_paths[0], read_only=True, data_only=True)
    per_class_sheet = source_workbook["PerClass"]
    per_class_rows = []
    for row in per_class_sheet.iter_rows(min_row=2, values_only=True):
        per_class_rows.append(
            {
                "class": int(row[0]),
                "train": int(row[1]),
                "validation": int(row[2]),
                "test": int(row[3]),
                "accuracy": float(row[4]),
            }
        )
    source_workbook.close()
    return {
        "directory": directory,
        "source_excel": excel_paths[0],
        "summary": summary,
        "epochs": epoch_rows,
        "per_class": per_class_rows,
        "actual_training_seconds": actual_training_seconds,
    }


def style_header(sheet, row=1) -> None:
    for cell in sheet[row]:
        cell.fill = HEADER_FILL
        cell.font = WHITE_BOLD
        cell.alignment = Alignment(horizontal="center", vertical="center")
    sheet.freeze_panes = f"A{row + 1}"
    sheet.auto_filter.ref = sheet.dimensions


def autosize(sheet, maximum=42) -> None:
    for column in range(1, sheet.max_column + 1):
        width = 0
        for cell in sheet[get_column_letter(column)]:
            if cell.value is not None:
                width = max(width, len(str(cell.value)))
        sheet.column_dimensions[get_column_letter(column)].width = min(maximum, width + 2)


def percent_columns(sheet, columns) -> None:
    for column in columns:
        for row in range(2, sheet.max_row + 1):
            sheet.cell(row, column).number_format = "0.0000%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proposed-method",
        choices=("dynamic", "dynamic_final"),
        default="dynamic",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = ("baseline", args.proposed_method)
    runs = {}
    for display, key in DATASETS:
        runs[key] = {method: read_run(key, method) for method in methods}
        baseline_hash = runs[key]["baseline"]["summary"]["split_sha256"]
        proposed_hash = runs[key][args.proposed_method]["summary"]["split_sha256"]
        if baseline_hash != proposed_hash:
            raise RuntimeError(
                f"Split mismatch for {key}: {baseline_hash} != {proposed_hash}"
            )

    workbook = Workbook()
    comparison = workbook.active
    comparison.title = "OverallComparison"
    comparison.append(
        [
            "Dataset",
            "Baseline OA", "Proposed OA", "OA delta (pp)",
            "Baseline AA", "Proposed AA", "AA delta (pp)",
            "Baseline Kappa", "Proposed Kappa", "Kappa delta",
            "Baseline best val OA", "Proposed best val OA",
            "Baseline best epoch", "Proposed best epoch",
        ]
    )
    for display, key in DATASETS:
        baseline = runs[key]["baseline"]["summary"]
        dynamic = runs[key][args.proposed_method]["summary"]
        btest, dtest = baseline["official_test"], dynamic["official_test"]
        comparison.append(
            [
                display,
                btest["OA"], dtest["OA"], (dtest["OA"] - btest["OA"]) * 100.0,
                btest["AA"], dtest["AA"], (dtest["AA"] - btest["AA"]) * 100.0,
                btest["kappa"], dtest["kappa"], dtest["kappa"] - btest["kappa"],
                baseline["best_val_OA"], dynamic["best_val_OA"],
                baseline["best_epoch"], dynamic["best_epoch"],
            ]
        )
    style_header(comparison)
    percent_columns(comparison, [2, 3, 5, 6, 8, 9, 11, 12])
    for column in (4, 7):
        for row in range(2, comparison.max_row + 1):
            comparison.cell(row, column).number_format = "+0.0000;-0.0000;0.0000"
    for row in range(2, comparison.max_row + 1):
        comparison.cell(row, 10).number_format = "+0.000000;-0.000000;0.000000"
    comparison.conditional_formatting.add(
        f"D2:D{comparison.max_row}",
        ColorScaleRule(start_type="min", start_color="F8696B", mid_type="num", mid_value=0,
                       mid_color="FFEB84", end_type="max", end_color="63BE7B"),
    )
    comparison.conditional_formatting.add(
        f"G2:G{comparison.max_row}",
        ColorScaleRule(start_type="min", start_color="F8696B", mid_type="num", mid_value=0,
                       mid_color="FFEB84", end_type="max", end_color="63BE7B"),
    )
    autosize(comparison)

    delta_chart = BarChart()
    delta_chart.type = "col"
    delta_chart.style = 10
    delta_chart.title = "Proposed minus baseline (percentage points)"
    delta_chart.y_axis.title = "Delta (pp)"
    delta_chart.x_axis.title = "Dataset"
    delta_chart.add_data(Reference(comparison, min_col=4, max_col=7, min_row=1, max_row=4), titles_from_data=True)
    delta_chart.set_categories(Reference(comparison, min_col=1, min_row=2, max_row=4))
    delta_chart.height = 8
    delta_chart.width = 16
    comparison.add_chart(delta_chart, "A7")

    all_runs = workbook.create_sheet("AllRuns")
    all_runs.append(
        [
            "Dataset", "Method", "Seed", "Best epoch", "Best val OA",
            "Test OA", "Test AA", "Test Kappa", "Test samples",
            "Actual 300-epoch seconds", "Actual hours", "Test seconds",
            "Parameters", "MACs", "Source Excel",
        ]
    )
    for display, key in DATASETS:
        for method in methods:
            run = runs[key][method]
            summary = run["summary"]
            test = summary["official_test"]
            seconds = run["actual_training_seconds"]
            all_runs.append(
                [
                    display, method, summary["seed"], summary["best_epoch"], summary["best_val_OA"],
                    test["OA"], test["AA"], test["kappa"], test["samples"],
                    seconds, seconds / 3600.0, test["elapsed_seconds"],
                    summary["parameters"], summary["macs"], str(run["source_excel"]),
                ]
            )
    style_header(all_runs)
    percent_columns(all_runs, [5, 6, 7, 8])
    for row in range(2, all_runs.max_row + 1):
        all_runs.cell(row, 11).number_format = "0.000"
    autosize(all_runs)

    timing = workbook.create_sheet("TrainingTime")
    timing.append(
        [
            "Dataset", "Baseline seconds", "Proposed seconds", "Extra seconds",
            "Baseline hours", "Proposed hours", "Overhead ratio", "Overhead (%)",
        ]
    )
    for display, key in DATASETS:
        baseline_seconds = runs[key]["baseline"]["actual_training_seconds"]
        dynamic_seconds = runs[key][args.proposed_method]["actual_training_seconds"]
        ratio = dynamic_seconds / baseline_seconds
        timing.append(
            [
                display, baseline_seconds, dynamic_seconds, dynamic_seconds - baseline_seconds,
                baseline_seconds / 3600.0, dynamic_seconds / 3600.0, ratio, ratio - 1.0,
            ]
        )
    style_header(timing)
    for row in range(2, timing.max_row + 1):
        timing.cell(row, 5).number_format = "0.000"
        timing.cell(row, 6).number_format = "0.000"
        timing.cell(row, 7).number_format = "0.000x"
        timing.cell(row, 8).number_format = "0.00%"
    autosize(timing)

    per_class = workbook.create_sheet("PerClassComparison")
    per_class.append(
        ["Dataset", "Class", "Test count", "Baseline accuracy", "Proposed accuracy", "Delta (pp)"]
    )
    for display, key in DATASETS:
        baseline_rows = runs[key]["baseline"]["per_class"]
        dynamic_rows = runs[key][args.proposed_method]["per_class"]
        if len(baseline_rows) != len(dynamic_rows):
            raise RuntimeError(f"Per-class row mismatch for {key}")
        for baseline_row, dynamic_row in zip(baseline_rows, dynamic_rows):
            per_class.append(
                [
                    display,
                    baseline_row["class"],
                    baseline_row["test"],
                    baseline_row["accuracy"],
                    dynamic_row["accuracy"],
                    (dynamic_row["accuracy"] - baseline_row["accuracy"]) * 100.0,
                ]
            )
    style_header(per_class)
    percent_columns(per_class, [4, 5])
    for row in range(2, per_class.max_row + 1):
        per_class.cell(row, 6).number_format = "+0.0000;-0.0000;0.0000"
    per_class.conditional_formatting.add(
        f"F2:F{per_class.max_row}",
        ColorScaleRule(start_type="min", start_color="F8696B", mid_type="num", mid_value=0,
                       mid_color="FFEB84", end_type="max", end_color="63BE7B"),
    )
    autosize(per_class)

    curves = workbook.create_sheet("EpochCurves")
    curves.append(
        [
            "Epoch",
            "Houston baseline val OA", "Houston proposed val OA",
            "R1 baseline val OA", "R1 proposed val OA",
            "R2 baseline val OA", "R2 proposed val OA",
        ]
    )
    for epoch_index in range(300):
        row = [epoch_index + 1]
        for _, key in DATASETS:
            row.append(float(runs[key]["baseline"]["epochs"][epoch_index]["val_OA"]))
            row.append(
                float(runs[key][args.proposed_method]["epochs"][epoch_index]["val_OA"])
            )
        curves.append(row)
    style_header(curves)
    percent_columns(curves, range(2, 8))
    autosize(curves)
    chart = LineChart()
    chart.title = "Validation OA across 300 epochs"
    chart.y_axis.title = "Validation OA"
    chart.x_axis.title = "Epoch"
    chart.add_data(Reference(curves, min_col=2, max_col=7, min_row=1, max_row=301), titles_from_data=True)
    chart.set_categories(Reference(curves, min_col=1, min_row=2, max_row=301))
    chart.height = 11
    chart.width = 24
    curves.add_chart(chart, "I2")

    notes = workbook.create_sheet("Notes")
    notes.append(["Item", "Explanation"])
    notes.append(["Comparison protocol", f"Baseline and {args.proposed_method} use identical seed-4 splits for each dataset; split hashes were checked before workbook generation."])
    notes.append(["Houston split", "90% of the official training split for training, 10% for validation; official test used once after selection."])
    notes.append(["SZUTree split", "Per-class 1% development sample; 10% of that sample for validation; remaining labeled pixels for test."])
    notes.append(["Training time correction", "Training seconds are recomputed by summing all 300 epoch_log.csv rows. Individual workbooks previously stored time only up to the selected best epoch."])
    notes.append(["Statistical limitation", "These are single seed-4 runs. Mean±standard deviation requires additional seeds."])
    style_header(notes)
    notes.column_dimensions["A"].width = 28
    notes.column_dimensions["B"].width = 110
    for row in notes.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    notes.append(["Proposed method", args.proposed_method])
    workbook.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
