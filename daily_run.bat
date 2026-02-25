@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat || echo "Virtualenv not found, using global python"
python -m ancser_quant.execution.main_loop
