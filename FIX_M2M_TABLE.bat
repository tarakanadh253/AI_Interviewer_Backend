@echo off
chcp 65001 >nul
cls
title Fix M2M Table for interview_sessions_topics
cd /d "%~dp0"

echo.
echo ========================================
echo   FIXING M2M TABLE
echo ========================================
echo.

echo Creating the ManyToMany join table...
python create_m2m_table.py

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
echo Restart your Django server and try creating a session.
echo.
pause
