@echo off
chcp 65001 >nul
cls
title Verify and Fix Question Bank
cd /d "%~dp0"

echo.
echo ========================================
echo   VERIFYING AND FIXING QUESTION BANK
echo ========================================
echo.

echo [Step 1] Checking and adding missing columns...
python add_columns_if_missing.py

echo.
echo [Step 2] Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo Cache cleared.

echo.
echo [Step 3] Running migrations...
python manage.py migrate --noinput

echo.
echo [Step 4] Testing Django setup...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from interview.models import Question; print('Django setup: OK'); print('Questions in DB:', Question.objects.count())"

echo.
echo ========================================
echo   VERIFICATION COMPLETE
echo ========================================
echo.
echo IMPORTANT: Restart your Django server now!
echo.
echo   1. Stop the current server (Ctrl+C)
echo   2. Run: python manage.py runserver
echo   3. Test the Question Bank tab in your frontend
echo.
pause
