@echo off
chcp 65001 >nul
cls
title Final Fix - Interview Start
cd /d "%~dp0"

echo.
echo ========================================
echo   FINAL FIX FOR INTERVIEW START
echo ========================================
echo.

echo [Step 1] Creating migrations...
python manage.py makemigrations
echo.

echo [Step 2] Running migrations...
python manage.py migrate --noinput
echo.

echo [Step 3] Verifying tables...
python check_tables.py
echo.

echo [Step 4] Adding missing columns...
python add_columns_if_missing.py
echo.

echo [Step 5] Clearing cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo Cache cleared.
echo.

echo ========================================
echo   READY!
echo ========================================
echo.
echo All fixes applied. Restart your server:
echo   python manage.py runserver
echo.
pause
