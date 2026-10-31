@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment not found. Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m streamlit run frontend\streamlit_app.py --server.port 8501
pause
