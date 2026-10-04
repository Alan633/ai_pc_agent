@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_ollama_qwen16k.ps1"
if errorlevel 1 (
  echo.
  echo 创建 16K Qwen 模型失败，请查看上面的错误。
  pause
  exit /b 1
)
echo.
pause
