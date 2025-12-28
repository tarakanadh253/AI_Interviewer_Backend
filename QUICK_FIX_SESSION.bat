@echo off
chcp 65001 >nul
cls
title Quick Fix Session Creation
cd /d "%~dp0"

echo.
echo ========================================
echo   QUICK FIX FOR SESSION CREATION
echo ========================================
echo.

echo [1/2] Clearing cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo Cache cleared.

echo.
echo [2/2] Verifying code compiles...
python -m py_compile interview\views.py interview\serializers.py
if errorlevel 1 (
    echo ERROR: Code has syntax errors!
    pause
    exit /b 1
)
echo Code compiles successfully.

echo.
echo ========================================
echo   READY!
echo ========================================
echo.
echo Fixed:
echo   - Improved error handling in serialization
echo   - Made user_email and user_name access safer
echo   - Added fallback response if serialization fails
echo   - Added debug logging
echo.
echo Restart your server and try again.
echo Check the server console for [DEBUG] messages.
echo.
pause
