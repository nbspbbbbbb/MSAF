# MSAF3 Gradient Diagnostics

This folder contains a side-car diagnostic runner for the unmodified
`msaf3/MSAF_3.py` model. It reuses the original MSAF3 model and dataset
pipeline, but runs its own training loop so gradient-balance and
gradient-conflict values can be written to CSV.

The script does not edit the original MSAF3 files.

## What It Writes

Each run creates a timestamped directory under `msaf3_diagnostics/results`
unless `--output-dir` is provided.

- `diagnostic_summary.xlsx`: the main file to open first.
- `diagnostic_summary.md`: compact text summary.
- `final_decision.csv`: one row per probed stage with the final diagnosis.
- `epoch_overview.csv`: one row per epoch with modality-level CGE and compact
  conflict summaries.
- `pair_conflict_epoch.csv`: one row per epoch and modality pair. This contains
  only cosine/conflict values, not pairwise CGE.
- `masked_eval_summary.csv`: full test accuracy and accuracy after zeroing
  each modality.
- `masked_eval_per_class.csv`: per-class masked-evaluation accuracy.
- `run_config.json`: arguments, dataset metadata, and output paths.

Raw per-batch tables are not written by default. Add `--save-raw-batch` only
when you need trace-level debugging; it writes `raw_batch_metrics.csv` and
`raw_bcp_pairs.csv`.

## Example

```powershell
E:\miniconda\python.exe D:\ADMIN\Documents\GIT\MSAF\msaf3_diagnostics\diagnose_msaf3_gradients.py --dataset houston2018 --data-dir D:\DATA_3\Houston2018 --epochs 20 --batch-size 16 --probe-stage stage2 --pin-memory
```

For SZUTree:

```powershell
E:\miniconda\python.exe D:\ADMIN\Documents\GIT\MSAF\msaf3_diagnostics\diagnose_msaf3_gradients.py --dataset szutree --data-dir D:\DATA_3\SZUTreeData2.0\SZUTreeData_R2_2.0 --epochs 20 --batch-size 16 --probe-stage both --pin-memory
```

The runner refuses tiny training splits by default. This prevents accidentally
diagnosing toy caches such as a 3-sample prepared directory. Use
`--allow-small-data` only for a smoke test.

## Reading The Key Columns

- `hsi_cge_share`, `lidar_cge_share`, `rgb_cge_share`: long-run contribution
  share for each modality.
- `cge_gap`: `max(CGE share) - min(CGE share)`. Large values suggest modality
  imbalance.
- `dominance_ratio`: strongest modality share divided by weakest modality
  share.
- `effective_modalities`: near 3 means balanced use of all modalities; near 1
  means one modality dominates.
- `min_pair_cos`, `worst_conflict_pair`, `mean_sample_conflict_rate`: compact
  three-modality conflict summary.
- `joint_problem_level`: the main answer for whether imbalance and conflict
  co-exist in the current run.
