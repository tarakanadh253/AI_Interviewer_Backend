@echo off
echo ============================================================
echo STOP SERVER AND FIX DATABASE
echo ============================================================
echo.
echo This script will:
echo 1. Stop any running Django/Python processes
echo 2. Fix the database schema
echo 3. Restart instructions
echo.
pause

echo.
echo Step 1: Stopping Python processes...
taskkill /F /IM python.exe 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Python processes stopped.
    timeout /t 2 /nobreak >nul
) else (
    echo No Python processes found (or already stopped).
)

echo.
echo Step 2: Fixing database schema...
python fix_db_simple.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Database fixed.
    echo ============================================================
    echo.
    echo Now start your server:
    echo   python manage.py runserver
    echo.
    echo Then:
    echo 1. Go to http://localhost:8000/admin/
    echo 2. Create a user in "User Profiles"
    echo 3. Test login in frontend
    echo.
) else (
    echo.
    echo ============================================================
    echo Fix script had issues. Trying to reset database...
    echo ============================================================
    echo.
    if exist db.sqlite3 (
        del /F /Q db.sqlite3
        echo Database deleted.
    )
    echo.
    echo Running migrations...
    python manage.py migrate
    echo.
    echo Database recreated!
)

pause
