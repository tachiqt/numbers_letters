@echo off
REM Launch Numbers+Letters specialist Streamlit (port 8505)
cd /d "%~dp0..\.."
set PYTHONPATH=%CD%
set PYTHONUTF8=1
".\.venv\Scripts\python.exe" -m streamlit run "specialists\numbers_letters\streamlit_app.py" --server.port 8505 --server.headless true
