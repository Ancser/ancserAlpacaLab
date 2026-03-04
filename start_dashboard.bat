@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat || echo "Virtualenv not found, using global python"
start "ancserAlpacaLab" python -m streamlit run frontend/app.py
