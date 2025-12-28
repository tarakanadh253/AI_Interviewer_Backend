@echo off
chcp 65001 >nul
echo ========================================
echo FIX DATABASE AND START SERVER
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping any running servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 1 /nobreak >nul

echo.
echo [2/3] Fixing database...
python quick_fix.py
if errorlevel 1 (
    echo ERROR: Database fix failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Django server...
echo.
echo ========================================
echo SERVER STARTING
echo ========================================
echo.
echo Backend URL: http://localhost:8000
echo API Test: http://localhost:8000/api/users/
echo.
echo IMPORTANT: Watch this window for error messages!
echo If you see errors, copy them and share them.
echo.
python manage.py runserver
