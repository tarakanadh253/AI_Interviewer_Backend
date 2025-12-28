@echo off
chcp 65001 >nul
echo ========================================
echo VERIFY DATABASE AND START SERVER
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking database schema...
python CHECK_DB_SCHEMA.py
echo.

echo [2/3] Testing user creation...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from interview.models import UserProfile; from interview.serializers import UserProfileCreateSerializer; data = {'username': 'verify_test', 'password': 'test123', 'email': 'v@test.com', 'access_type': 'TRIAL'}; UserProfile.objects.filter(username='verify_test').delete(); s = UserProfileCreateSerializer(data=data); print('Valid:', s.is_valid()); u = s.save() if s.is_valid() else None; print('Created:', u.username if u else 'FAILED'); u.delete() if u else None"
echo.

echo [3/3] Starting server...
echo.
python manage.py runserver
