@echo off
echo ============================================================
echo RESET DATABASE - This will delete and recreate the database
echo ============================================================
echo.
echo IMPORTANT: Make sure your Django server is STOPPED!
echo (Press Ctrl+C in the server terminal if it's running)
echo.
pause

echo.
echo Step 1: Stopping any processes using the database...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo Step 2: Deleting old database...
if exist db.sqlite3 (
    del /F /Q db.sqlite3
    echo Database deleted.
) else (
    echo Database file not found (might already be deleted).
)

echo.
echo Step 3: Running migrations to create new database...
python manage.py migrate

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Database has been reset.
    echo ============================================================
    echo.
    echo Next steps:
    echo 1. Create admin user (if needed):
    echo    python manage.py createsuperuser
    echo.
    echo 2. Seed data (if you have topics/questions):
    echo    python manage.py seed_data
    echo.
    echo 3. Start your server:
    echo    python manage.py runserver
    echo.
) else (
    echo.
    echo ============================================================
    echo ERROR: Migration failed. Check the error above.
    echo ============================================================
)

pause
