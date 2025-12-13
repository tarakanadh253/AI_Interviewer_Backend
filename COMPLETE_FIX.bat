@echo off
chcp 65001 >nul
echo ========================================
echo COMPLETE DATABASE FIX
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Stopping servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2/3] Fixing database schema (recreating table if needed)...
python FIX_DATABASE_SCHEMA.py
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
echo The database has been fixed!
echo Test: http://localhost:8000/api/users/
echo.
python manage.py runserver
