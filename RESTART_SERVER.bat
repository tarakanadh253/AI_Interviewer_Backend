@echo off
chcp 65001 >nul
cls
title Restart Django Server
cd /d "%~dp0"

echo.
echo ========================================
echo   RESTARTING DJANGO SERVER
echo ========================================
echo.
echo Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul

echo.
echo Starting Django server...
echo.
python manage.py runserver
