@echo off
chcp 65001 >nul
cls
title Complete Question Bank Fix
cd /d "%~dp0"

echo.
echo ========================================
echo   COMPLETE QUESTION BANK FIX
echo ========================================
echo.

echo [Step 1] Creating all missing database tables...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from django.core.management import call_command; call_command('migrate', verbosity=1, interactive=False)"

echo.
echo [Step 2] Verifying tables exist...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\"); tables = [r[0] for r in c.fetchall()]; print('All tables:', ', '.join(sorted(tables))); print('answers table:', 'EXISTS' if 'answers' in tables else 'MISSING'); print('questions table:', 'EXISTS' if 'questions' in tables else 'MISSING'); conn.close()"

echo.
echo [Step 3] Adding missing columns to questions table...
python add_columns_if_missing.py

echo.
echo [Step 4] Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo Cache cleared.

echo.
echo ========================================
echo   FIX COMPLETE!
echo ========================================
echo.
echo Fixed issues:
echo   - Topic variable error in serializer
echo   - Missing answers table handling
echo   - Missing columns in questions table
echo.
echo IMPORTANT: Restart your Django server now!
echo.
echo   1. Stop the current server (Ctrl+C)
echo   2. Run: python manage.py runserver
echo   3. Test the Question Bank tab
echo.
pause
