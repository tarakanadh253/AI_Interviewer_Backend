@echo off
chcp 65001 >nul
cls
title Fix All Database Issues
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING ALL DATABASE ISSUES
echo ========================================
echo.

echo [Step 1] Fixing missing tables (answers, django_session, etc.)...
python fix_all_tables.py

echo.
echo [Step 2] Running all migrations...
python manage.py migrate --noinput

echo.
echo [Step 3] Verifying answers table exists...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='answers'\"); result = c.fetchone(); print('✓ answers table exists!' if result else '✗ answers table NOT found'); conn.close()"

echo.
echo [Step 4] Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo Cache cleared.

echo.
echo ========================================
echo   FIXES COMPLETE!
echo ========================================
echo.
echo IMPORTANT: Restart your Django server now!
echo.
echo   1. Stop the current server (Ctrl+C)
echo   2. Run: python manage.py runserver
echo   3. Test the Question Bank tab
echo.
pause
