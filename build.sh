#!/usr/bin/env bash
# exit on error
set -o errexit


python -m pip install --upgrade pip
# Install CPU-only torch first to avoid downloading 800MB+ CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate
