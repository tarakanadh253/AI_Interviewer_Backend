@echo off
chcp 65001 >nul
echo ========================================
echo Fixing User Creation Error
echo ========================================
echo.

cd /d "%~dp0"

echo Step 1: Adding access_type column if missing...
python quick_fix.py
echo.

echo Step 2: Verifying database and testing...
python verify_and_fix.py
echo.

echo Step 3: Running migrations...
python manage.py migrate interview --verbosity=1
echo.

echo ========================================
echo Fix complete!
echo.
echo Please restart your Django server:
echo   python manage.py runserver
echo ========================================
echo.
pause
