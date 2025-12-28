@echo off
echo ============================================================
echo Fixing Database Schema
echo ============================================================
echo.

python apply_migration.py

echo.
echo ============================================================
echo Done! Please restart your Django server.
echo ============================================================
pause
