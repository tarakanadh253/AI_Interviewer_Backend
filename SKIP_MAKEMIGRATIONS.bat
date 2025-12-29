@echo off
chcp 65001 >nul
echo ========================================
echo SKIP MAKEMIGRATIONS - Direct Database Fix
echo ========================================
echo.
echo This script will fix the database directly without
echo running makemigrations (which is causing issues).
echo.

cd /d "%~dp0"

echo [1/3] Stopping Django server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 1 /nobreak >nul

echo.
echo [2/3] Fixing database schema directly...
python quick_fix.py
if errorlevel 1 (
    echo ERROR: quick_fix.py failed
    pause
    exit /b 1
)

echo.
echo [3/3] Running comprehensive fix...
python FIX_ALL_ISSUES.py
if errorlevel 1 (
    echo ERROR: FIX_ALL_ISSUES.py failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo FIX COMPLETE!
echo ========================================
echo.
echo The database has been fixed directly.
echo You can now restart your Django server:
echo   python manage.py runserver
echo.
echo NOTE: If you see migration warnings, you can ignore them
echo since we've fixed the database schema directly.
echo.
pause
