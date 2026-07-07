# Step 17: final-fusion dynamic MSAF3

This experiment preserves the existing Step 15 implementation. Step 15
applies q-guided CoRiM/FW weights to Transformer-1 inputs; Step 17 applies the
same weights only to the final three-branch feature sum.

The clean-split runner method name is `dynamic_final`. Its output directory is
separate from both `baseline` and the earlier `dynamic` experiment, so their
source, checkpoints and results are not overwritten.

Run all three datasets sequentially (there is no shutdown command):

```cmd
D:\ADMIN\Documents\GIT\MSAF\experiments\step17_final_fusion_dynamic\run_three_datasets.cmd
```

Or run them separately:

```cmd
E:\miniconda\python.exe -u -B D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\run_houston_clean_split.py --method dynamic_final --dataset houston2018 --seed 4 --epochs 300 --warmup-epochs 50
E:\miniconda\python.exe -u -B D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\run_houston_clean_split.py --method dynamic_final --dataset szutree-r1 --seed 4 --epochs 300 --warmup-epochs 50
E:\miniconda\python.exe -u -B D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\run_houston_clean_split.py --method dynamic_final --dataset szutree-r2 --seed 4 --epochs 300 --warmup-epochs 50
```
