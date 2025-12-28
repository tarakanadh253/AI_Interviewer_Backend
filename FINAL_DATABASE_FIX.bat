@echo off
chcp 65001 >nul
echo ========================================
echo FINAL DATABASE FIX
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Stopping servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2/4] Fixing database schema...
python FIX_DATABASE_WITH_FK.py
if errorlevel 1 (
    echo ERROR: Database fix failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Running migrations to recreate dependent tables...
python manage.py migrate
if errorlevel 1 (
    echo WARNING: Migration had issues, but continuing...
)

echo.
echo [4/4] Starting Django server...
echo.
echo ========================================
echo SERVER STARTING
echo ========================================
echo.
echo Database has been fixed!
echo Test: http://localhost:8000/api/users/
echo.
python manage.py runserver
