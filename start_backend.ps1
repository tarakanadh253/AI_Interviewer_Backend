# PowerShell script to start the backend server
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting AI Interview Buddy Backend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Warning: Virtual environment not found. Using system Python." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Installing/updating required packages..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Verifying NLP libraries..." -ForegroundColor Yellow
python -c "from sentence_transformers import SentenceTransformer; from sklearn.metrics.pairwise import cosine_similarity; import numpy as np; import nltk; print('✓ All NLP libraries available')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: NLP libraries not properly installed!" -ForegroundColor Red
    Write-Host "Please run: pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Running migrations..." -ForegroundColor Yellow
python manage.py migrate

Write-Host ""
Write-Host "Seeding topics and questions..." -ForegroundColor Yellow
python manage.py seed_data

Write-Host ""
Write-Host "Starting Django development server..." -ForegroundColor Green
Write-Host "Backend will be available at: http://localhost:8000/api/" -ForegroundColor Green
Write-Host ""
python manage.py runserver
