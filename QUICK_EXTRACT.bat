@echo off
chcp 65001 >nul
cls
title Quick Extract Questions
cd /d "%~dp0"

echo.
echo ========================================
echo   QUICK QUESTION EXTRACTION
echo ========================================
echo.

echo Running extraction script...
echo.

python verify_and_extract.py

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
pause
