@echo off
chcp 65001 >nul
cls
title Fixing Question Bank - DO THIS NOW
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING QUESTION BANK TAB
echo ========================================
echo.

echo Step 1: Adding missing columns using SQLite directly...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute('PRAGMA table_info(questions)'); cols = [r[1] for r in c.fetchall()]; print('Current columns:', ', '.join(cols)); missing = []; missing.append('source_type') if 'source_type' not in cols else None; missing.append('reference_links') if 'reference_links' not in cols else None; print('Missing columns:', missing if missing else 'None'); [c.execute('ALTER TABLE questions ADD COLUMN source_type VARCHAR(10) DEFAULT ''MANUAL''') or conn.commit() or print('Added source_type') for _ in [1] if 'source_type' not in cols]; [c.execute('ALTER TABLE questions ADD COLUMN reference_links TEXT') or conn.commit() or print('Added reference_links') for _ in [1] if 'reference_links' not in cols]; c.execute('PRAGMA table_info(questions)'); final = [r[1] for r in c.fetchall()]; print('Final columns:', ', '.join(final)); print('SUCCESS!' if 'source_type' in final and 'reference_links' in final else 'WARNING: Check columns'); conn.close()"

echo.
echo Step 2: Running migrations...
python manage.py migrate

echo.
echo Step 3: Testing the endpoint...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from interview.models import Question; from interview.serializers import AdminQuestionSerializer; qs = Question.objects.all()[:2]; print(f'Testing {qs.count()} questions...'); [print(f'Q{q.id}: OK') if AdminQuestionSerializer(q).data else None for q in qs]; print('Test complete!')"

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
echo IMPORTANT: Restart your Django server now!
echo.
pause
