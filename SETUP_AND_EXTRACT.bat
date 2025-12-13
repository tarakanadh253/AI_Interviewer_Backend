@echo off
chcp 65001 >nul
cls
title Setup and Extract Questions from Links
cd /d "%~dp0"

echo.
echo ========================================
echo   SETUP AND EXTRACT QUESTIONS
echo ========================================
echo.

echo [Step 1] Installing required packages...
echo.
python -m pip install --upgrade pip >nul 2>&1
python -m pip install beautifulsoup4 requests

echo.
echo [Step 2] Verifying installation...
echo.
python -c "import bs4; print('  OK: beautifulsoup4')" 2>nul || (
    echo   ERROR: beautifulsoup4 not found!
    echo   Please install manually: pip install beautifulsoup4
    pause
    exit /b 1
)
python -c "import requests; print('  OK: requests')" 2>nul || (
    echo   ERROR: requests not found!
    echo   Please install manually: pip install requests
    pause
    exit /b 1
)

echo.
echo [Step 3] Running extraction...
echo.
python verify_and_extract.py

echo.
pause
