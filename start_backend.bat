@echo off
echo ========================================
echo Starting AI Interview Buddy Backend
echo ========================================
echo.

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found. Using system Python.
)

echo.
echo Running migrations...
python manage.py migrate

echo.
echo Seeding topics and questions...
python manage.py seed_data

echo.
echo Starting Django development server...
echo Backend will be available at: http://localhost:8000/api/
echo.
python manage.py runserver

pause
