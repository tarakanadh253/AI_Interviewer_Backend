@echo off
chcp 65001 >nul
cls
title Fix Question Bank - Simple Fix
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING QUESTION BANK
echo ========================================
echo.

echo Adding missing columns...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute('PRAGMA table_info(questions)'); cols = [r[1] for r in c.fetchall()]; print('Current columns:', ', '.join(cols)); exec('c.execute(\"ALTER TABLE questions ADD COLUMN source_type VARCHAR(10) DEFAULT ''MANUAL''\") or conn.commit() or print(''Added source_type'')' if 'source_type' not in cols else 'print(''source_type exists'')'); exec('c.execute(\"ALTER TABLE questions ADD COLUMN reference_links TEXT\") or conn.commit() or print(''Added reference_links'')' if 'reference_links' not in cols else 'print(''reference_links exists'')'); c.execute('PRAGMA table_info(questions)'); final = [r[1] for r in c.fetchall()]; print('Final:', ', '.join(final)); conn.close()"

echo.
echo Running migrations...
python manage.py migrate

echo.
echo ========================================
echo   DONE! Restart your server.
echo ========================================
echo.
pause
