@echo off
chcp 65001 >nul
cls
title Fixing Database...
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING DATABASE - DO NOT CLOSE
echo ========================================
echo.

echo [1] Stopping servers...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul
echo Done.
echo.

echo [2] Dropping all tables...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from django.db import connection; c = connection.cursor(); c.execute('PRAGMA foreign_keys = OFF'); [c.execute(f'DROP TABLE IF EXISTS {t[0]}') for t in c.execute('SELECT name FROM sqlite_master WHERE type=\"table\" AND name NOT LIKE \"sqlite_%%\"').fetchall()]; c.execute('PRAGMA foreign_keys = ON'); print('Tables dropped.')"
echo.

echo [3] Running migrations...
python manage.py migrate
echo.

echo [4] Seeding data...
python manage.py seed_data
echo.

echo [5] Testing user creation...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from interview.models import UserProfile; from interview.serializers import UserProfileCreateSerializer; UserProfile.objects.filter(username='test').delete(); s = UserProfileCreateSerializer(data={'username': 'test', 'password': 'test123', 'email': 't@t.com', 'access_type': 'TRIAL'}); print('Valid:', s.is_valid()); u = s.save() if s.is_valid() else None; print('Created:', u.username if u else 'FAILED'); u.delete() if u else None"
echo.

echo ========================================
echo   FIX COMPLETE!
echo ========================================
echo.
echo Starting server...
echo.
python manage.py runserver
