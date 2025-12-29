@echo off
chcp 65001 >nul
echo ========================================
echo STARTING SERVER (After Database Fix)
echo ========================================
echo.

cd /d "%~dp0"

echo Verifying database...
python CHECK_DB_SCHEMA.py
echo.

echo Starting Django server...
echo.
echo ========================================
echo SERVER STARTING
echo ========================================
echo.
echo Test these URLs:
echo   - http://localhost:8000/api/users/     (should return JSON)
echo   - http://localhost:8000/admin/        (Django admin)
echo.
echo If you see errors, check this window!
echo.
python manage.py runserver
