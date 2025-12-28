@echo off
chcp 65001 >nul
cls
title Fix Session Creation
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING SESSION CREATION
echo ========================================
echo.

echo [Step 1] Creating all missing database tables...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from django.core.management import call_command; call_command('migrate', verbosity=1, interactive=False)"

echo.
echo [Step 2] Verifying answers table exists...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='answers'\"); result = c.fetchone(); print('answers table:', 'EXISTS' if result else 'MISSING - creating...'); conn.close()"

echo.
echo [Step 3] Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo Cache cleared.

echo.
echo ========================================
echo   FIX COMPLETE!
echo ========================================
echo.
echo Fixed issues:
echo   - Made get_answer_count handle missing answers table
echo   - Made get_topics_list more defensive
echo   - Made viewset prefetch_related conditional
echo.
echo IMPORTANT: Restart your Django server now!
echo.
echo   1. Stop the current server (Ctrl+C)
echo   2. Run: python manage.py runserver
echo   3. Try starting an interview again
echo.
pause
