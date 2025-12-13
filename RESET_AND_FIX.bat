@echo off
chcp 65001 >nul
echo ========================================
echo RESET DATABASE AND FIX
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Stopping servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2/4] Running complete database fix...
python COMPLETE_FIX.py
if errorlevel 1 (
    echo ERROR: Database fix failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo DATABASE RESET COMPLETE!
echo ========================================
echo.
echo Starting server...
echo.
python manage.py runserver
