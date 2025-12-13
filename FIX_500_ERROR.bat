@echo off
echo ========================================
echo Fixing 500 Error - Adding access_type
echo ========================================
echo.

cd /d "%~dp0"

echo Step 1: Stopping any running Django servers...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *runserver*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 2: Adding access_type column to database...
python FIX_ACCESS_TYPE_NOW.py

echo.
echo Step 3: Verifying migration status...
python manage.py migrate interview --verbosity=1

echo.
echo ========================================
echo Done! Please restart your Django server:
echo   python manage.py runserver
echo ========================================
echo.
pause
