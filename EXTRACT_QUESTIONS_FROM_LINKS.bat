@echo off
chcp 65001 >nul
cls
title Extract Questions from Link Questions
cd /d "%~dp0"

echo.
echo ========================================
echo   EXTRACTING QUESTIONS FROM LINKS
echo ========================================
echo.

echo This will extract questions from all LINK-type questions
echo and create individual MANUAL questions for each extracted Q&A.
echo.

echo [Checking packages...]
python -c "import bs4" 2>nul || (
    echo.
    echo ERROR: beautifulsoup4 is not installed!
    echo.
    echo Installing packages first...
    python -m pip install beautifulsoup4 requests
    echo.
)

echo.
echo [Starting extraction...]
echo.
python verify_and_extract.py

echo.
pause
