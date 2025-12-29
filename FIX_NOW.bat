@echo off
chcp 65001 >nul
cls
echo ========================================
echo FIXING DATABASE - PLEASE WAIT
echo ========================================
echo.

cd /d "%~dp0"

echo Stopping any running servers...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul
echo.

echo Running complete fix...
echo.
python FINAL_FIX_ALL.py

echo.
echo ========================================
echo FIX COMPLETE!
echo ========================================
echo.
echo Starting server...
echo.
python manage.py runserver
