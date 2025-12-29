@echo off
chcp 65001 >nul
echo Testing user creation...
echo.

cd /d "%~dp0"

python quick_fix.py
echo.
python capture_actual_error.py
echo.
pause
