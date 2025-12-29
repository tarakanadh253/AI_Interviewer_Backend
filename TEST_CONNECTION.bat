@echo off
chcp 65001 >nul
echo ========================================
echo TESTING REACT-DJANGO CONNECTION
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Running complete diagnostic...
python COMPLETE_DIAGNOSTIC.py
echo.

echo [2/4] Ensuring database is ready...
python GUARANTEED_FIX.py
echo.

echo [3/4] Checking if server is running...
netstat -an | findstr ":8000" >nul
if errorlevel 1 (
    echo Server is NOT running on port 8000
    echo.
    echo [4/4] Starting server...
    echo.
    echo ========================================
    echo SERVER STARTING
    echo ========================================
    echo.
    echo Frontend should connect to: http://localhost:8000/api
    echo.
    echo Test in browser: http://localhost:8000/api/users/
    echo Should return JSON (empty array [] if no users)
    echo.
    python manage.py runserver
) else (
    echo Server IS running on port 8000
    echo.
    echo [4/4] Testing API endpoint...
    echo.
    echo Open in browser: http://localhost:8000/api/users/
    echo.
    echo If you see JSON (or []), the backend is working!
    echo If you see an error, check the Django server console.
    echo.
    pause
)
