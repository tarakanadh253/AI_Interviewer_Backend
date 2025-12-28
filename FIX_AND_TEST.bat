@echo off
chcp 65001 >nul
echo ========================================
echo FIX DATABASE AND TEST
echo ========================================
echo.

cd /d "%~dp0"

echo Stopping any running servers...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Running complete fix...
python COMPLETE_FIX.py
echo.

echo Testing user creation...
python TEST_USER_CREATION.py
echo.

echo.
echo ========================================
echo If tests passed, start server with:
echo   python manage.py runserver
echo ========================================
pause
