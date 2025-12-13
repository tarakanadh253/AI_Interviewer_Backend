#!/usr/bin/env python
"""Test the question bank endpoint to see what error occurs"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from rest_framework.test import APIRequestFactory
from interview.views import AdminQuestionViewSet
from interview.models import Question
from interview.serializers import AdminQuestionSerializer

print("\n" + "="*70)
print("TESTING QUESTION BANK ENDPOINT")
print("="*70 + "\n")

# Check database columns
print("1. Checking database schema...")
import sqlite3
conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()
c.execute('PRAGMA table_info(questions)')
cols = {row[1]: row for row in c.fetchall()}
print(f"   Columns: {', '.join(sorted(cols.keys()))}")
print(f"   Has source_type: {'source_type' in cols}")
print(f"   Has reference_links: {'reference_links' in cols}\n")
conn.close()

# Test model access
print("2. Testing Question model...")
try:
    questions = Question.objects.all()
    print(f"   Total questions: {questions.count()}")
    if questions.exists():
        q = questions.first()
        print(f"   First question ID: {q.id}")
        print(f"   Has source_type attr: {hasattr(q, 'source_type')}")
        try:
            print(f"   source_type value: {q.source_type}")
        except Exception as e:
            print(f"   Error accessing source_type: {e}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test serializer
print("\n3. Testing AdminQuestionSerializer...")
try:
    if questions.exists():
        q = questions.first()
        serializer = AdminQuestionSerializer(q)
        data = serializer.data
        print(f"   ✓ Serialization successful")
        print(f"   Keys: {', '.join(data.keys())}")
        print(f"   source_type: {data.get('source_type')}")
        print(f"   reference_links: {data.get('reference_links', 'N/A')[:50] if data.get('reference_links') else 'None'}")
    else:
        print("   ⚠ No questions to test")
except Exception as e:
    print(f"   ✗ Serialization error: {e}")
    import traceback
    traceback.print_exc()

# Test viewset
print("\n4. Testing AdminQuestionViewSet.list()...")
try:
    factory = APIRequestFactory()
    request = factory.get('/api/admin/questions/')
    
    viewset = AdminQuestionViewSet()
    viewset.request = request
    viewset.format_kwarg = None
    
    response = viewset.list(request)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✓ Success! Questions returned: {len(response.data)}")
    else:
        print(f"   ✗ Error response: {response.data}")
except Exception as e:
    print(f"   ✗ ViewSet error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")
