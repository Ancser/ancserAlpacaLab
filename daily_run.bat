@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat || echo "Virtualenv not found, using global python"

:: Load PYTHON_EXEC from .env if it exists
set PYTHON_EXEC=python
if exist .env (
    for /f "tokens=1,2 delims==" %%a in ('type .env ^| findstr /i "^PYTHON_EXEC="') do (
        set PYTHON_EXEC=%%b
    )
)

"%PYTHON_EXEC%" -m ancser_quant.execution.main_loop --run-once