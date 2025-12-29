@echo off
chcp 65001 >nul
cls
title Extract Questions (with Python Path Fix)
cd /d "%~dp0"

echo.
echo ========================================
echo   EXTRACT QUESTIONS (WITH PATH FIX)
echo ========================================
echo.

echo Running extraction with Python path fix...
echo This script will automatically add D:\python-packages to Python path
echo if packages are installed there.
echo.

python fix_import_issue.py

echo.
pause


