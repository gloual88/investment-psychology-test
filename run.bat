@echo off
setlocal

set "PY=%~dp0..\..\pykrx_venv\Scripts\python.exe"
set "APP=%~dp0app.py"

"%PY%" -m streamlit run "%APP%" --server.port 8533

endlocal
pause
