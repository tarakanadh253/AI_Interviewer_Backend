#!/usr/bin/env bash
# exit on error
set -o errexit


python -m pip install --upgrade pip
# Install CPU-only torch first to avoid downloading 800MB+ CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python manage.py collectstatic --no-input

echo "Running migrations..."
python manage.py migrate || echo "WARNING: Migration failed. This is likely due to the DATABASE_URL being unreachable during build. Ensure you are using the EXTERNAL Database URL in Render dashboard."
