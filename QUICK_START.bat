@echo off
chcp 65001 >nul
echo ========================================
echo QUICK START - Fix and Run Server
echo ========================================
echo.

cd /d "%~dp0"

echo Fixing database...
python quick_fix.py
echo.

echo Starting server...
echo.
python manage.py runserver
