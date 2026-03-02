@echo off
cd /d "%~dp0"

:: Log start time
echo [%date% %time%] daily_run.bat started >> logs\dailyrun_bat.log

:: Wait for network to be ready (max 120 seconds)
echo Waiting for network...
set /a RETRY=0
:WAIT_NET
ping -n 1 -w 2000 8.8.8.8 >nul 2>&1
if %errorlevel%==0 goto NET_OK
set /a RETRY+=1
if %RETRY% GEQ 24 (
    echo [%date% %time%] ERROR: Network not available after 120s. Aborting. >> logs\dailyrun_bat.log
    exit /b 1
)
timeout /t 5 /nobreak >nul
goto WAIT_NET

:NET_OK
echo [%date% %time%] Network ready after ~%RETRY% retries >> logs\dailyrun_bat.log

:: Run in batch mode (run-once + force) for Task Scheduler
:: Use full path to venv python so Task Scheduler doesn't depend on PATH
"%~dp0.venv\Scripts\python.exe" -m ancser_quant.execution.main_loop --run-once
set EXIT_CODE=%errorlevel%

echo [%date% %time%] daily_run.bat finished with exit code %EXIT_CODE% >> logs\dailyrun_bat.log
exit /b %EXIT_CODE%
