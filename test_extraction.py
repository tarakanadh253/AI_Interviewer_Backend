#!/usr/bin/env python
"""Test question extraction"""
import sys

print("Testing imports...")
try:
    import bs4
    print("✓ beautifulsoup4 imported")
except ImportError as e:
    print(f"✗ beautifulsoup4 import failed: {e}")
    sys.exit(1)

try:
    import requests
    print("✓ requests imported")
except ImportError as e:
    print(f"✗ requests import failed: {e}")
    sys.exit(1)

print("\nTesting Django setup...")
import os, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    django.setup()
    print("✓ Django setup complete")
except Exception as e:
    print(f"✗ Django setup failed: {e}")
    sys.exit(1)

print("\nTesting models...")
try:
    from interview.models import Question
    link_count = Question.objects.filter(source_type='LINK').count()
    manual_count = Question.objects.filter(source_type='MANUAL').count()
    print(f"✓ Models imported")
    print(f"  LINK questions: {link_count}")
    print(f"  MANUAL questions: {manual_count}")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting extraction utility...")
try:
    from interview.utils.question_extractor import process_link_question
    print("✓ Extraction utility imported")
except Exception as e:
    print(f"✗ Extraction utility import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed! Ready to extract questions.")
print("="*50)
