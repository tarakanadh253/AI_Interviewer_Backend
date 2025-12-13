@echo off
chcp 65001 >nul
cls
title Fixing Database and Starting Server
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING DATABASE AND STARTING SERVER
echo ========================================
echo.

echo [Step 1/4] Fixing django_session table...
python create_session_table.py
if errorlevel 1 (
    echo WARNING: Session table fix had issues, continuing anyway...
)
echo.

echo [Step 2/4] Running all migrations...
python manage.py migrate
if errorlevel 1 (
    echo ERROR: Migrations failed!
    pause
    exit /b 1
)
echo.

echo [Step 3/4] Verifying database...
python -c "import os, sys, sqlite3; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); import django; django.setup(); from django.conf import settings; db_path = str(settings.DATABASES['default']['NAME']); conn = sqlite3.connect(db_path); c = conn.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='django_session'\"); exists = c.fetchone(); print('django_session table:', 'EXISTS ✓' if exists else 'MISSING ✗'); conn.close()"
echo.

echo [Step 4/4] Starting Django server...
echo.
echo ========================================
echo   SERVER STARTING
echo ========================================
echo.
echo Backend URL: http://localhost:8000
echo Admin Panel: http://localhost:8000/admin/
echo API Base: http://localhost:8000/api/
echo.
echo IMPORTANT: Keep this window open!
echo Press Ctrl+C to stop the server.
echo.
echo ========================================
echo.

python manage.py runserver

pause
