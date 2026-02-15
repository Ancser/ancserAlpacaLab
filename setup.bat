@echo off
cd /d "%~dp0"
echo ==========================================
echo   Setting up Ancser Alpaca Lab Environment
echo ==========================================

:: 1. Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.10+ and check "Add Python to PATH".
    pause
    exit /b
)

:: 2. Create Virtual Environment if missing
if not exist ".venv" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [1/3] Virtual environment already exists.
)

:: 3. Activate and Install Dependencies
echo [2/3] Installing dependencies...
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

:: 4. Check Config
if not exist ".env" (
    echo [3/3] Creating .env template...
    echo APCA_API_KEY_ID=YOUR_KEY_HERE > .env
    echo APCA_API_SECRET_KEY=YOUR_SECRET_HERE >> .env
    echo.
    echo [IMPORTANT] Please edit '.env' file with your API keys!
) else (
    echo [3/3] .env found.
)

echo.
echo ==========================================
echo   Setup Complete!
echo   You can now run 'daily_run.bat'
echo ==========================================
pause
