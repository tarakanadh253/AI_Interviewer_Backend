#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test listing questions to find the 500 error
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from rest_framework.test import APIRequestFactory
from interview.views import AdminQuestionViewSet
from interview.models import Question, Topic
from interview.serializers import AdminQuestionSerializer

print("\n" + "="*70)
print("TESTING LIST QUESTIONS (GET)")
print("="*70 + "\n")

# Check if questions exist
questions = Question.objects.all()
print(f"Total questions in database: {questions.count()}")

if questions.exists():
    print("\n1. Testing serializer on existing questions...")
    for q in questions[:3]:  # Test first 3
        try:
            print(f"\n   Question ID {q.id}:")
            print(f"   - Topic: {q.topic_id if hasattr(q, 'topic_id') else 'N/A'}")
            print(f"   - Has topic object: {hasattr(q, 'topic') and q.topic is not None}")
            
            serializer = AdminQuestionSerializer(q)
            data = serializer.data
            print(f"   ✓ Serialization successful")
            print(f"   - topic_name: {data.get('topic_name')}")
            print(f"   - answer_count: {data.get('answer_count')}")
        except Exception as e:
            print(f"   ✗ Serialization failed: {e}")
            import traceback
            traceback.print_exc()
            break
else:
    print("   ⚠ No questions in database")

# Test the viewset
print("\n2. Testing AdminQuestionViewSet.list()...")
try:
    factory = APIRequestFactory()
    request = factory.get('/api/admin/questions/')
    
    viewset = AdminQuestionViewSet()
    viewset.request = request
    viewset.format_kwarg = None
    
    response = viewset.list(request)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   ✓ List successful")
        print(f"   Questions returned: {len(response.data)}")
    else:
        print(f"   ✗ Error: {response.data}")
except Exception as e:
    print(f"   ✗ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")
