@echo off
chcp 65001 >nul
cls
title Create Missing Tables
cd /d "%~dp0"

echo.
echo ========================================
echo   CREATING MISSING DATABASE TABLES
echo ========================================
echo.

echo [Step 1] Creating migrations (non-interactive)...
python manage.py makemigrations --noinput
if errorlevel 1 (
    echo.
    echo Migration creation failed. Trying with default values...
    python manage.py makemigrations interview --name create_missing_tables --empty
    echo.
    echo Please run migrations manually: python manage.py migrate
    pause
    exit /b 1
)
echo.

echo [Step 2] Running migrations...
python manage.py migrate --noinput
echo.

echo [Step 3] Verifying interview_sessions table exists...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='interview_sessions'\"); result = c.fetchone(); print('interview_sessions table:', 'EXISTS' if result else 'MISSING'); conn.close()"

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
echo Restart your server and try again.
echo.
pause
