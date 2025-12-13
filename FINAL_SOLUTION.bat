@echo off
chcp 65001 >nul
echo ========================================
echo FINAL SOLUTION - Complete Fix
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 1 /nobreak >nul

echo.
echo [2/3] Fixing database (ensuring access_type column exists)...
python quick_fix.py
if errorlevel 1 (
    echo ERROR: Database fix failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Django server...
echo.
echo Server will be at: http://localhost:8000
echo.
echo IMPORTANT: Check the server console for any errors!
echo.
python manage.py runserver
