@echo off
chcp 65001 >nul
echo ========================================
echo FIX USER CREATION AND START SERVER
echo ========================================
echo.

cd /d "%~dp0"

echo Step 1: Fixing database...
python quick_fix.py
echo.

echo Step 2: Running complete fix and test...
python COMPLETE_USER_FIX.py
echo.

echo Step 3: Starting Django server...
echo.
echo Server will be available at: http://localhost:8000
echo API endpoint: http://localhost:8000/api/users/
echo.
echo Press Ctrl+C to stop the server
echo.
python manage.py runserver
