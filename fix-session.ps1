# Fix django_session table issue
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fixing Django Session Table" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot

Write-Host "Running Django migrations..." -ForegroundColor Yellow
python manage.py migrate

Write-Host ""
Write-Host "Checking database file..." -ForegroundColor Yellow
if (Test-Path "db.sqlite3") {
    Write-Host "✓ Database file exists" -ForegroundColor Green
    
    # Check if django_session table exists
    Write-Host ""
    Write-Host "Checking django_session table..." -ForegroundColor Yellow
    python -c "import os, django; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings'); django.setup(); from django.db import connection; c = connection.cursor(); c.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='django_session'\"); result = c.fetchone(); print('✓ django_session table exists!' if result else '✗ django_session table NOT found')"
} else {
    Write-Host "✗ Database file not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Done! Restart your Django server." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
