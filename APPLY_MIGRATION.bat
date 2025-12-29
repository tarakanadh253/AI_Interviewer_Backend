@echo off
echo ========================================
echo Applying Migration for access_type
echo ========================================
echo.

cd /d "%~dp0"

echo Running migration...
python manage.py migrate interview 0003_add_access_type --verbosity=2

echo.
echo ========================================
echo Migration complete!
echo ========================================
echo.
echo If you see errors, try:
echo   python manage.py migrate
echo.
pause
