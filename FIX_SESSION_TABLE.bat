@echo off
chcp 65001 >nul
cls
title Fixing django_session Table
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING DJANGO_SESSION TABLE
echo ========================================
echo.

echo [1] Running Django migrations...
python manage.py migrate
echo.

echo [2] Verifying django_session table...
python -c "import os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from django.db import connection; c = connection.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='django_session'\"); result = c.fetchone(); print('SUCCESS: django_session table exists!' if result else 'ERROR: django_session table NOT found')"
echo.

echo ========================================
echo   FIX COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Restart your Django server (Ctrl+C if running)
echo 2. Run: python manage.py runserver
echo 3. Go to http://localhost:8000/admin/
echo.
pause
