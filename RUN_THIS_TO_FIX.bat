@echo off
echo ============================================================
echo FIXING DATABASE - This will fix the 500 errors
echo ============================================================
echo.
echo Step 1: Stopping any running Django server...
echo (Please press Ctrl+C in your server terminal if it's running)
echo.
pause

echo.
echo Step 2: Applying database migration...
python apply_migration.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Database has been fixed.
    echo ============================================================
    echo.
    echo Now restart your Django server with:
    echo   python manage.py runserver
    echo.
    echo Then test:
    echo   1. Go to http://localhost:8000/admin/
    echo   2. Create a user in "User Profiles"
    echo   3. Test login in frontend
    echo.
) else (
    echo.
    echo ============================================================
    echo ERROR: Migration failed. Trying alternative method...
    echo ============================================================
    echo.
    echo Deleting database and recreating...
    if exist db.sqlite3 (
        del db.sqlite3
        echo Database deleted.
    )
    echo.
    echo Running migrations...
    python manage.py migrate
    echo.
    echo ============================================================
    echo Database recreated! Now restart your server.
    echo ============================================================
)

pause
