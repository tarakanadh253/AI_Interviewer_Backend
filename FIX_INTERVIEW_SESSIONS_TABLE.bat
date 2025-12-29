@echo off
chcp 65001 >nul
cls
title Fix interview_sessions Table
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING interview_sessions TABLE
echo ========================================
echo.

echo This will create the interview_sessions table if it doesn't exist.
echo.

echo [Step 1] Ensuring interview_sessions table exists...
python ensure_interview_sessions_table.py
if errorlevel 1 (
    echo ERROR: Failed to create table!
    pause
    exit /b 1
)
echo.

echo [Step 2] Ensuring M2M join table exists...
python fix_m2m_simple.py
echo.

echo [Step 3] Running migrations...
python manage.py migrate --noinput
echo.

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
echo Restart your Django server and try creating a session.
echo.
pause
