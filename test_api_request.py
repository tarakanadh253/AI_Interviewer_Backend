#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the API endpoint directly to see what error is returned
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from rest_framework.test import APIRequestFactory
from interview.views import AdminQuestionViewSet
from interview.models import Topic

print("\n" + "="*70)
print("TESTING API ENDPOINT DIRECTLY")
print("="*70 + "\n")

# Get or create a topic
topic = Topic.objects.first()
if not topic:
    topic = Topic.objects.create(name="Test Topic", description="Test")
    print(f"Created test topic: {topic.name} (ID: {topic.id})")
else:
    print(f"Using existing topic: {topic.name} (ID: {topic.id})")

# Create test data similar to what frontend sends
test_data = {
    'topic': topic.id,  # Integer ID
    'source_type': 'MANUAL',
    'question_text': 'What is Python?',
    'ideal_answer': 'Python is a programming language.',
    'difficulty': 'EASY',
    'is_active': True
}

print(f"\nTest data: {json.dumps(test_data, indent=2)}")

# Create request
factory = APIRequestFactory()
request = factory.post(
    '/api/admin/questions/',
    data=json.dumps(test_data),
    content_type='application/json'
)

# Create viewset and test
viewset = AdminQuestionViewSet()
viewset.request = request
viewset.format_kwarg = None

print("\n1. Testing serializer validation...")
serializer = viewset.get_serializer(data=test_data)
if serializer.is_valid():
    print("   ✓ Validation passed")
    print(f"   Validated data keys: {list(serializer.validated_data.keys())}")
else:
    print(f"   ✗ Validation failed:")
    for field, errors in serializer.errors.items():
        print(f"      {field}: {errors}")

print("\n2. Testing full create endpoint...")
try:
    response = viewset.create(request)
    print(f"   Status: {response.status_code}")
    if response.status_code == 201:
        print("   ✓ Question created successfully!")
        print(f"   Response data: {json.dumps(response.data, indent=2, default=str)}")
    else:
        print(f"   ✗ Error response:")
        print(f"   {json.dumps(response.data, indent=2, default=str)}")
except Exception as e:
    print(f"   ✗ Exception: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70 + "\n")
