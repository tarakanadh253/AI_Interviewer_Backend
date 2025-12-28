@echo off
chcp 65001 >nul
cls
title Fixing Question Bank - Adding Missing Columns
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING QUESTION BANK TAB
echo ========================================
echo.

echo [1/3] Running migrations...
python manage.py migrate
echo.

echo [2/3] Adding missing columns to questions table...
python fix_questions_schema.py
echo.

echo [3/3] Verifying fix...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from django.db import connection; c = connection.cursor(); c.execute('PRAGMA table_info(questions)'); cols = [r[1] for r in c.fetchall()]; print('Columns:', ', '.join(cols)); print('Has source_type:', 'source_type' in cols); print('Has reference_links:', 'reference_links' in cols)"
echo.

echo ========================================
echo   FIX COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Restart your Django server
echo 2. Refresh the Question Bank tab
echo.
pause
