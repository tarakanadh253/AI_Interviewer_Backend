@echo off
chcp 65001 >nul
cls
title Install Packages and Extract Questions
cd /d "%~dp0"

echo.
echo ========================================
echo   INSTALL PACKAGES AND EXTRACT
echo ========================================
echo.

echo [Step 1] Installing packages...
echo.
python -m pip install beautifulsoup4 requests

echo.
echo [Step 2] Waiting for installation to complete...
timeout /t 2 /nobreak >nul

echo.
echo [Step 3] Running extraction...
echo.
python verify_and_extract.py

echo.
pause
