@echo off
chcp 65001 >nul
title Django Server - Port 8000
cd /d "%~dp0"

echo.
echo ========================================
echo   STARTING DJANGO SERVER
echo ========================================
echo.

REM Quick fix for django_session table
python -c "import os, sys, sqlite3; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); import django; django.setup(); from django.conf import settings; db_path = str(settings.DATABASES['default']['NAME']); conn = sqlite3.connect(db_path); c = conn.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='django_session'\"); exists = c.fetchone(); conn.close(); exit(0 if exists else 1)" 2>nul
if errorlevel 1 (
    echo Creating django_session table...
    python create_session_table.py >nul 2>&1
)

echo Running migrations...
python manage.py migrate >nul 2>&1

echo.
echo ========================================
echo   SERVER STARTING
echo ========================================
echo.
echo URL: http://localhost:8000
echo API: http://localhost:8000/api/
echo Admin: http://localhost:8000/admin/
echo.
echo Keep this window open!
echo Press Ctrl+C to stop.
echo.
echo ========================================
echo.

python manage.py runserver

pause
