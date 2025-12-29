@echo off
echo ============================================================
echo SETUP AND CHECK TOPICS
echo ============================================================
echo.
echo This will:
echo 1. Check if topics exist
echo 2. Seed topics if needed
echo 3. Verify setup
echo.
pause

echo.
echo Step 1: Checking current topics...
python check_topics.py

echo.
echo Step 2: Seeding topics (if needed)...
python manage.py seed_data

echo.
echo Step 3: Verifying topics again...
python check_topics.py

echo.
echo ============================================================
echo Done!
echo ============================================================
echo.
echo Now make sure your server is running:
echo   python manage.py runserver
echo.
echo Then test the API:
echo   http://localhost:8000/api/topics/
echo.
pause
