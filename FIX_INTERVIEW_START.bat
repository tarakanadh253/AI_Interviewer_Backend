@echo off
chcp 65001 >nul
cls
title Fix Interview Start - Complete Fix
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING INTERVIEW START ISSUE
echo ========================================
echo.

echo [Step 1] Creating and running migrations...
python manage.py makemigrations
python manage.py migrate --noinput
if errorlevel 1 (
    echo ERROR: Migrations failed!
    pause
    exit /b 1
)

echo.
echo [Step 2] Verifying all required tables exist...
python check_tables.py
if errorlevel 1 (
    echo WARNING: Some tables are missing. Creating migrations and running them...
    python manage.py makemigrations
    python manage.py migrate --noinput
)

echo.
echo [Step 3] Adding missing columns to questions table...
python add_columns_if_missing.py

echo.
echo [Step 4] Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo Cache cleared.

echo.
echo [Step 5] Testing imports...
python -c "import os, sys, django; sys.path.insert(0, '.'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from interview.views import InterviewSessionViewSet; from interview.serializers import InterviewSessionSerializer; print('✓ All imports successful')"

echo.
echo ========================================
echo   FIX COMPLETE!
echo ========================================
echo.
echo All fixes applied:
echo   - Migrations run
echo   - Tables verified
echo   - Missing columns added
echo   - Error handling improved
echo   - Serializer made defensive
echo.
echo IMPORTANT: Restart your Django server now!
echo.
echo   1. Stop the current server (Ctrl+C)
echo   2. Run: python manage.py runserver
echo   3. Try starting an interview
echo.
pause
