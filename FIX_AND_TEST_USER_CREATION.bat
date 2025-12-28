@echo off
chcp 65001 >nul
echo ========================================
echo FIX AND TEST USER CREATION
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Fixing database...
python quick_fix.py
if errorlevel 1 (
    echo ERROR: Database fix failed
    pause
    exit /b 1
)

echo.
echo [2/3] Testing user creation...
python test_user_creation_detailed.py
if errorlevel 1 (
    echo ERROR: Test failed
    pause
    exit /b 1
)

echo.
echo [3/3] If tests passed, you can start the server:
echo   python manage.py runserver
echo.
pause
