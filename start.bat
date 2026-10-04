@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INIT] Creating Python virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto :error
)

".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error

if not exist ".env" (
    echo [ERROR] .env was not found. Copy .env.example to .env and configure it.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo AI PC Agent failed. Check the error above and logs\agent.log.
pause
exit /b 1
