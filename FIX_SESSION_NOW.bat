@echo off
chcp 65001 >nul
cls
title Fixing django_session Table - DO NOT CLOSE
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING DJANGO_SESSION TABLE
echo ========================================
echo.

echo Step 1: Running all Django migrations...
python manage.py migrate
if errorlevel 1 (
    echo ERROR: Migration failed!
    pause
    exit /b 1
)
echo.

echo Step 2: Specifically migrating sessions app...
python manage.py migrate sessions
echo.

echo Step 3: Creating django_session table if it doesn't exist...
python -c "import os, sys, sqlite3; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); import django; django.setup(); from django.conf import settings; db_path = str(settings.DATABASES['default']['NAME']); conn = sqlite3.connect(db_path); c = conn.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='django_session'\"); exists = c.fetchone(); print('Table exists:', 'YES' if exists else 'NO'); conn.close()"
echo.

echo Step 4: Verifying table was created...
python create_session_table.py
echo.

echo ========================================
echo   FIX COMPLETE!
echo ========================================
echo.
echo IMPORTANT: Restart your Django server now!
echo 1. Stop the server (Ctrl+C in the server window)
echo 2. Run: python manage.py runserver
echo 3. Go to: http://localhost:8000/admin/
echo.
pause
