@echo off
chcp 65001 >nul
cls
title Install Required Packages
cd /d "%~dp0"

echo.
echo ========================================
echo   INSTALLING REQUIRED PACKAGES
echo ========================================
echo.

echo Installing beautifulsoup4 and requests...
echo.

python -m pip install --upgrade pip
python -m pip install beautifulsoup4 requests

echo.
echo ========================================
echo   VERIFYING INSTALLATION
echo ========================================
echo.

python -c "from bs4 import BeautifulSoup; print('✓ beautifulsoup4 is installed and working')" 2>nul || echo "✗ beautifulsoup4 import failed - but package may be installed"
python -c "import requests; print('✓ requests is installed')" 2>nul || echo "✗ requests installation failed"

echo.
echo ========================================
echo   DONE!
echo ========================================
echo.
pause
