@echo off
chcp 65001 >nul
cls
title Fixing Questions Table Schema
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING QUESTIONS TABLE
echo ========================================
echo.

echo Step 1: Adding missing columns...
python add_missing_columns.py

echo.
echo Step 2: Running migrations...
python manage.py migrate

echo.
echo Step 3: Verifying columns exist...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute('PRAGMA table_info(questions)'); cols = [r[1] for r in c.fetchall()]; print('Columns:', ', '.join(cols)); print(''); print('Has source_type:', 'source_type' in cols); print('Has reference_links:', 'reference_links' in cols); conn.close()"

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
echo Restart your Django server now.
echo.
pause
