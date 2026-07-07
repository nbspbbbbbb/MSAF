# Warm-up 50 + dynamic fusion to 300 epochs

## Protocol

- Official Houston2018 training split: 18,750 samples.
- Epochs 1-50: original equal Transformer-1 input fusion (warm-up).
- Epochs 51-300: q-guided, gradient-balanced CoRiM/FW weights on Transformer-1 inputs only.
- Transformer-2, residual branches, and final equal feature sum were left unchanged.
- Total deduplicated training time: 14,194.19 s (3.943 h).
- Model-selection validation: a fixed 4,000-sample subset (200 samples per class) drawn from the official test split. Therefore the selected-checkpoint full-test number is not a completely untouched-test estimate.

The warm-up probe selected epoch 50 because the best validation OA in epochs 31-50 was 94.375%, versus 92.475% in epochs 1-30 (an improvement of 1.900 percentage points).

## Training/validation result

| checkpoint | validation OA |
|---|---:|
| warm-up epoch 50 | 94.375% |
| best dynamic epoch 268 | 96.300% |
| final dynamic epoch 300 | 95.850% |

The best dynamic validation result improved by 1.925 percentage points over the epoch-50 warm-up endpoint. The dynamic validation mean of the per-batch maximum modality weight was 0.5354; its maximum epoch mean was 0.5468. Thus the learned/FW weights moved meaningfully away from 1/3 but did not collapse toward one modality.

## Official full test (2,000,160 samples)

| checkpoint | OA | AA | Kappa | own top-20% conflict OA |
|---|---:|---:|---:|---:|
| previous 30-epoch auxiliary model (context only) | 88.0953% | 92.6126% | 84.6776% | 94.5629% |
| best dynamic epoch 268 | **92.0905%** | **96.1523%** | **89.8355%** | 92.1226% |
| final dynamic epoch 300 | 90.7501% | 95.7343% | 88.1521% | 90.2700% |

The epoch-268 checkpoint is 1.3404 percentage points better in full-test OA than epoch 300, so selecting the best checkpoint matters. It is 3.9952 points above the previous 30-epoch model, but that difference cannot be attributed solely to dynamic fusion: training duration and auxiliary/calibration losses also differ. A controlled epoch-50 equal-fusion full-test checkpoint or a same-300-epoch equal-fusion run would be required for a causal ablation; the latter was intentionally not rerun here at the user's request.

The conflict subsets in the last column are selected independently from each model's own shallow pairwise-conflict ranking, so their OA values should not be treated as a fixed-sample cross-model comparison.

## Artifacts

- `results/best_dynamic.pt`: selected epoch-268 checkpoint.
- `results/final_epoch_300.pt`: final epoch-300 checkpoint.
- `results/training_summary.json`: training summary and switch decision.
- `results/epoch_log.csv`: per-epoch metrics (epoch 131 appears twice because the first attempt logged before a Windows file-lock interrupted checkpoint saving; the second entry is the valid resumed run).
- `official_full_test_best/REPORT.md`: epoch-268 official full-test report.
- `official_full_test_final300/REPORT.md`: epoch-300 official full-test report.

