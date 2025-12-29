@echo off
chcp 65001 >nul
cls
title Starting Django Server (Fixed)
cd /d "%~dp0"

echo.
echo ========================================
echo   STARTING DJANGO SERVER
echo ========================================
echo.

REM Check if port 8000 is in use
netstat -ano | findstr :8000 >nul
if %errorlevel% == 0 (
    echo WARNING: Port 8000 is already in use!
    echo.
    echo Options:
    echo 1. Stop the other process using port 8000
    echo 2. Use a different port: python manage.py runserver 8001
    echo.
    choice /C YN /M "Kill processes on port 8000"
    if errorlevel 2 goto skip_kill
    if errorlevel 1 (
        echo Stopping processes on port 8000...
        for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%a 2>nul
        timeout /t 2 /nobreak >nul
    )
)
:skip_kill

echo [1/3] Ensuring django_session table exists...
python create_session_table.py >nul 2>&1
echo   Done.

echo.
echo [2/3] Running migrations...
python manage.py migrate >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Migrations failed!
    echo   Running with verbose output...
    python manage.py migrate
    pause
    exit /b 1
)
echo   Done.

echo.
echo [3/3] Starting Django development server...
echo.
echo ========================================
echo   SERVER INFORMATION
echo ========================================
echo.
echo Backend URL:    http://localhost:8000
echo Admin Panel:    http://localhost:8000/admin/
echo API Endpoints:  http://localhost:8000/api/
echo.
echo ========================================
echo.
echo IMPORTANT: Keep this window open!
echo Press Ctrl+C to stop the server.
echo.
echo Starting server...
echo.

python manage.py runserver

pause
