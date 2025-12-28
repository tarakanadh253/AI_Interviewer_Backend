@echo off
chcp 65001 >nul
echo ========================================
echo Starting Django Server
echo ========================================
echo.

cd /d "%~dp0"

echo [1/2] Ensuring database is ready...
python quick_fix.py
if errorlevel 1 (
    echo ERROR: Database fix failed
    pause
    exit /b 1
)

echo.
echo [2/2] Starting Django server...
echo.
echo Server will start at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python manage.py runserver
