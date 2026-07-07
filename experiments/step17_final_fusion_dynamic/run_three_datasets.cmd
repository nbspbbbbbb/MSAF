@echo off
setlocal

REM ================================================================
REM 当前最终版（dynamic_final）三数据集顺序运行脚本
REM 1. 每个数据集训练300轮，前50轮等权warm-up；
REM 2. 每轮保存latest.pt，可中断后重新执行本脚本自动续训；
REM 3. 每个数据集完成后立刻全样本测试并生成TXT和Excel；
REM 4. 任一实验报错时跳到:failed并停止，避免错误状态下继续下一组；
REM 5. 本脚本没有shutdown命令，全部完成后只正常退出。
REM ================================================================

REM 本机训练使用的Python环境，以及三个数据集共用的实验流程入口。
set "PY=E:\miniconda\python.exe"
set "RUNNER=D:\ADMIN\Documents\GIT\MSAF\experiments\step16_clean_split_comparison\run_houston_clean_split.py"

REM --method dynamic_final：在两个Transformer完成后、最终求和前动态加权。
REM 输出目录会包含数据集名、方法名和seed，所以三组结果不会互相覆盖。
echo [1/3] Houston2018 - final-fusion dynamic
"%PY%" -u -B "%RUNNER%" --method dynamic_final --dataset houston2018 --seed 4 --epochs 300 --warmup-epochs 50
REM errorlevel>=1表示Python异常，立即停止批处理。
if errorlevel 1 goto :failed

echo [2/3] SZUTree R1 - final-fusion dynamic
"%PY%" -u -B "%RUNNER%" --method dynamic_final --dataset szutree-r1 --seed 4 --epochs 300 --warmup-epochs 50
if errorlevel 1 goto :failed

echo [3/3] SZUTree R2 - final-fusion dynamic
"%PY%" -u -B "%RUNNER%" --method dynamic_final --dataset szutree-r2 --seed 4 --epochs 300 --warmup-epochs 50
if errorlevel 1 goto :failed

echo All three final-fusion dynamic experiments completed.
exit /b 0

:failed
REM 已经写完的checkpoint仍保留；排除错误后重新运行会从latest.pt下一轮继续。
echo Experiment stopped because one training command failed.
exit /b 1
