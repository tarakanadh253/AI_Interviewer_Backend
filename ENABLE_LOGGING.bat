@echo off
chcp 65001 >nul
cls
title Enable Logging for Debugging
cd /d "%~dp0"

echo.
echo ========================================
echo   ENABLING DETAILED LOGGING
echo ========================================
echo.
echo This will help identify the exact error.
echo.
echo Starting server with detailed logging...
echo.
python manage.py runserver --verbosity 2
