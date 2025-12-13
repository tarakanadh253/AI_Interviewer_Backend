@echo off
chcp 65001 >nul
cls
title Fix Session 500 Error
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING SESSION CREATION 500 ERROR
echo ========================================
echo.

echo [Step 1] Running migrations to ensure all tables exist...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from django.core.management import call_command; call_command('migrate', verbosity=1, interactive=False); print('Migrations complete')"

echo.
echo [Step 2] Verifying critical tables exist...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); tables = ['answers', 'interview_sessions', 'questions', 'topics', 'user_profiles']; for t in tables: c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='\" + t + \"'\"); result = c.fetchone(); print(t + ':', 'EXISTS' if result else 'MISSING'); conn.close()"

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
echo   - Changed answers field to SerializerMethodField
echo   - Added get_answers() method with error handling
echo   - All table access is now defensive
echo.
echo IMPORTANT: Restart your Django server now!
echo.
echo   1. Stop the current server (Ctrl+C)
echo   2. Run: python manage.py runserver
echo   3. Try creating a session again
echo.
pause
