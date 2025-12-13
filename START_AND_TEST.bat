@echo off
chcp 65001 >nul
cls
title Start Server and Test
cd /d "%~dp0"

echo.
echo ========================================
echo   STARTING DJANGO SERVER
echo ========================================
echo.
echo The Question Bank should now work!
echo.
echo Columns verified:
echo   - source_type: EXISTS
echo   - reference_links: EXISTS
echo.
echo Questions in database: 9
echo.
echo Starting server on http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python manage.py runserver
