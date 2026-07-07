# Houston2018 clean 90/10 comparison

This experiment leaves the original `msaf3` source unchanged.  Both commands
use the same seed-controlled, per-class stratified split of the 18,750 official
training samples: 16,875 for training and 1,875 for validation.  The official
2,000,160-pixel test split is used only after checkpoint selection.

Run the original three-modal MSAF3 baseline from Windows CMD:

```cmd
E:\miniconda\python.exe D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\run_houston_clean_split.py --method baseline --seed 4
```

Run the complete proposed method with the identical split:

```cmd
E:\miniconda\python.exe D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\run_houston_clean_split.py --method dynamic --seed 4
```

Each command is resumable.  Its output directory contains `best_model.pt`,
`latest.pt`, `epoch_log.csv`, an MSAF-style `*_Report.txt`, and one `*.xlsx`
workbook containing Summary, PerClass, ConfusionMatrix, EpochLog and Config
sheets.  Do not run both commands at the same time on one GPU.

## Remaining five experiments, then shut down

`run_remaining_five_then_shutdown.cmd` sequentially runs:

1. Original MSAF3 on SZUTree R1.
2. Original MSAF3 on SZUTree R2.
3. Proposed method on Houston2018.
4. Proposed method on SZUTree R1.
5. Proposed method on SZUTree R2.

The batch no longer shuts Windows down.  When all commands succeed and all
five expected Excel workbooks exist, it prints a completion message and exits.
A failed run stops the remaining commands and also exits without changing the
Windows power state.
