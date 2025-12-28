@echo off
chcp 65001 >nul
echo ========================================
echo Fixing 500 Error - Adding access_type Column
echo ========================================
echo.

cd /d "%~dp0"

echo Checking database schema...
python quick_fix.py
echo.

echo Running migrations...
python manage.py migrate interview
echo.

echo ========================================
echo Fix complete! Please restart your server:
echo   python manage.py runserver
echo ========================================
echo.
pause
