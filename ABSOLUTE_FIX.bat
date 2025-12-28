@echo off
chcp 65001 >nul
echo ========================================
echo ABSOLUTE FIX - Complete Solution
echo ========================================
echo.

cd /d "%~dp0"

echo Step 1: Stopping servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 2: Ensuring access_type column exists...
python quick_fix.py
echo.

echo Step 3: Verifying database...
python SIMPLE_CHECK.py
echo.

echo Step 4: Starting Django server...
echo.
echo ========================================
echo SERVER STARTING
echo ========================================
echo.
echo Test these URLs in your browser:
echo   1. http://localhost:8000/api/users/     (should return JSON)
echo   2. http://localhost:8000/admin/        (Django admin)
echo.
echo If you see errors in this window, copy them!
echo.
python manage.py runserver
