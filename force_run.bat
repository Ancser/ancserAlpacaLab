@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat || echo "Virtualenv not found, using global python"
echo "WARNING: Forcing Execution ignoring market hours!"
start "ancserAlpacaLab Force" python -m ancser_quant.execution.main_loop --run-once --force
pause
