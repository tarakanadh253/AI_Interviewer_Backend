@echo off
chcp 65001 >nul
cls
title Fix Installation and Extract Questions
cd /d "%~dp0"

echo.
echo ========================================
echo   FIX INSTALLATION AND EXTRACT
echo ========================================
echo.

echo Step 1: Installing beautifulsoup4 and requests...
python -m pip install --upgrade --force-reinstall beautifulsoup4 requests

echo.
echo Step 2: Verifying installation...
python -c "from bs4 import BeautifulSoup; print('SUCCESS: beautifulsoup4 works!')"
if errorlevel 1 (
    echo ERROR: beautifulsoup4 still not working!
    echo.
    echo Please try installing manually:
    echo   python -m pip install --user beautifulsoup4 requests
    echo.
    pause
    exit /b 1
)

python -c "import requests; print('SUCCESS: requests works!')"
if errorlevel 1 (
    echo ERROR: requests still not working!
    pause
    exit /b 1
)

echo.
echo Step 3: Running extraction script...
echo.
python verify_and_extract.py

echo.
pause
