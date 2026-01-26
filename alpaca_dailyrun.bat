@echo off
:: ==========================================
:: 設定區 (請依據您的電腦修改這裡)
:: ==========================================

:: 1. 設定專案所在的資料夾路徑 (例如 F:\ancserQuant\ancserAlpca)
set PROJECT_DIR=F:\ancserQuant\ancserAlpacaLab

:: 2. 設定 Python 執行檔 (如果您有虛擬環境，請填寫 venv\Scripts\python.exe 的絕對路徑)
set PYTHON_EXE=python

:: ==========================================
:: 執行區
:: ==========================================

:: 切換到專案硬碟與目錄 (/d 參數允許跨磁碟切換)
cd /d "%PROJECT_DIR%"

:: 建立 logs 資料夾 (如果不存在)
if not exist logs mkdir logs

echo. >> logs\dailyrun.log
echo [START] %date% %time% - Starting Alpaca Bot >> logs\dailyrun.log

:: 執行 Python 腳本
:: 參數說明: 
:: --paper : 使用模擬交易
:: >> logs\... : 將輸出寫入日誌檔
"%PYTHON_EXE%" alpaca_execute.py --paper >> logs\dailyrun.log 2>&1

:: 檢查執行結果
if %errorlevel% neq 0 (
    echo [ERROR] Script failed! Check logs for details.
    echo [ERROR] Script failed at %date% %time% >> logs\dailyrun.log
) else (
    echo [SUCCESS] Script finished successfully.
    echo [SUCCESS] Finished at %date% %time% >> logs\dailyrun.log
)

:: 讓視窗停留 10 秒，方便您手動執行時查看結果
timeout /t 10