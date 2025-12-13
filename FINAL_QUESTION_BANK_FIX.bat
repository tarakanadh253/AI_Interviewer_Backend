@echo off
chcp 65001 >nul
cls
title Final Question Bank Fix
cd /d "%~dp0"

echo.
echo ========================================
echo   FINAL QUESTION BANK FIX
echo ========================================
echo.

echo [1/3] Verifying database schema...
python -c "import sqlite3; conn = sqlite3.connect('db.sqlite3'); c = conn.cursor(); c.execute('PRAGMA table_info(questions)'); cols = [r[1] for r in c.fetchall()]; print('Columns:', ', '.join(cols)); print('Has source_type:', 'source_type' in cols); print('Has reference_links:', 'reference_links' in cols); conn.close()"

echo.
echo [2/3] Running migrations...
python manage.py migrate

echo.
echo [3/3] Testing endpoint...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from rest_framework.test import APIRequestFactory; from interview.views import AdminQuestionViewSet; factory = APIRequestFactory(); request = factory.get('/api/admin/questions/'); viewset = AdminQuestionViewSet(); viewset.request = request; viewset.format_kwarg = None; response = viewset.list(request); print('Status:', response.status_code); print('Questions:', len(response.data) if response.status_code == 200 else 'ERROR')"

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
echo Restart your Django server and test!
echo.
pause
