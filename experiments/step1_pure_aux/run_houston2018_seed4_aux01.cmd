@echo off
set STEP1_TRAIN_PER_CLASS=50
set STEP1_TEST_PER_CLASS=200
"E:\miniconda\python.exe" -u -B "D:\ADMIN\Documents\GIT\MSAF\experiments\step1_pure_aux\run_step1_pure_aux.py" ^
  --dataset houston2018 ^
  --run-name houston2018_pilot50x200_seed4_aux01 ^
  --data-dir "D:\DATA_3\Houston2018\prepared_msaf3" ^
  --output-dir "D:\ADMIN\Documents\GIT\MSAF\experiments\step1_pure_aux\results" ^
  --seed 4 ^
  --epochs 30 ^
  --batch-size 16 ^
  --patch-size 11 ^
  --num-workers 0 ^
  --pin-memory ^
  --aux-loss-weight 0.1 ^
  --grad-log-interval 0 ^
  1>"D:\ADMIN\Documents\GIT\MSAF\experiments\step1_pure_aux\logs\houston2018_seed4_aux01.out.log" ^
  2>"D:\ADMIN\Documents\GIT\MSAF\experiments\step1_pure_aux\logs\houston2018_seed4_aux01.err.log"

exit /b %ERRORLEVEL%
