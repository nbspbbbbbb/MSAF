@echo off
setlocal
cd /d D:\ADMIN\Documents\GIT\MSAF

set "PY=E:\miniconda\python.exe"
set "RUN=D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\run_houston_clean_split.py"
set "OUT=D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\results"

echo ============================================================
echo [1/5] Original MSAF3 - SZUTree R1
echo ============================================================
"%PY%" "%RUN%" --dataset szutree-r1 --method baseline --seed 4
if errorlevel 1 goto :failed
if not exist "%OUT%\szutree_r1_baseline_seed4\szutree_r1_3mod_baseline_seed4.xlsx" goto :missing

echo ============================================================
echo [2/5] Original MSAF3 - SZUTree R2
echo ============================================================
"%PY%" "%RUN%" --dataset szutree-r2 --method baseline --seed 4
if errorlevel 1 goto :failed
if not exist "%OUT%\szutree_r2_baseline_seed4\szutree_r2_3mod_baseline_seed4.xlsx" goto :missing

echo ============================================================
echo [3/5] Proposed method - Houston2018
echo ============================================================
"%PY%" "%RUN%" --dataset houston2018 --method dynamic --seed 4
if errorlevel 1 goto :failed
if not exist "%OUT%\houston2018_dynamic_seed4\houston2018_3mod_dynamic_seed4.xlsx" goto :missing

echo ============================================================
echo [4/5] Proposed method - SZUTree R1
echo ============================================================
"%PY%" "%RUN%" --dataset szutree-r1 --method dynamic --seed 4
if errorlevel 1 goto :failed
if not exist "%OUT%\szutree_r1_dynamic_seed4\szutree_r1_3mod_dynamic_seed4.xlsx" goto :missing

echo ============================================================
echo [5/5] Proposed method - SZUTree R2
echo ============================================================
"%PY%" "%RUN%" --dataset szutree-r2 --method dynamic --seed 4
if errorlevel 1 goto :failed
if not exist "%OUT%\szutree_r2_dynamic_seed4\szutree_r2_3mod_dynamic_seed4.xlsx" goto :missing

echo ============================================================
echo All five experiments completed and all five Excel files exist.
echo Automatic shutdown is disabled. Windows will remain running.
echo ============================================================
exit /b 0

:missing
echo.
echo ERROR: An experiment returned success but its expected Excel file is missing.
exit /b 2

:failed
echo.
echo ERROR: One experiment failed. Remaining experiments were stopped.
exit /b 1
