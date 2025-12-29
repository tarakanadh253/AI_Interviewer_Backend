@echo off
echo ========================================
echo Running Django Migrations
echo ========================================
cd /d "%~dp0"
python manage.py migrate
echo.
echo ========================================
echo Checking if database was created...
echo ========================================
if exist db.sqlite3 (
    echo SUCCESS: Database file created!
) else (
    echo ERROR: Database file not found!
)
echo.
pause
