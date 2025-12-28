#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug question creation - simulate what frontend sends
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from interview.serializers import AdminQuestionSerializer
from interview.models import Topic

print("\n" + "="*70)
print("DEBUGGING QUESTION CREATION")
print("="*70 + "\n")

# Get a topic
topic = Topic.objects.first()
if not topic:
    topic = Topic.objects.create(name="General", description="Default")
    print(f"Created topic: {topic.name} (ID: {topic.id})")
else:
    print(f"Using topic: {topic.name} (ID: {topic.id})")

# Simulate frontend data (with undefined values as None)
test_cases = [
    {
        'name': 'Full MANUAL question',
        'data': {
            'topic': topic.id,
            'source_type': 'MANUAL',
            'question_text': 'What is Python?',
            'ideal_answer': 'Python is a programming language.',
            'difficulty': 'EASY',
            'is_active': True
        }
    },
    {
        'name': 'MANUAL with None values',
        'data': {
            'topic': topic.id,
            'source_type': 'MANUAL',
            'question_text': 'What is Python?',
            'ideal_answer': 'Python is a programming language.',
            'difficulty': 'EASY',
            'is_active': True,
            'reference_links': None  # Simulating undefined from frontend
        }
    },
    {
        'name': 'Missing question_text',
        'data': {
            'topic': topic.id,
            'source_type': 'MANUAL',
            'ideal_answer': 'Python is a programming language.',
            'difficulty': 'EASY',
        }
    }
]

for test_case in test_cases:
    print(f"\n{'='*70}")
    print(f"Test: {test_case['name']}")
    print(f"{'='*70}")
    print(f"Data: {json.dumps(test_case['data'], indent=2, default=str)}")
    
    serializer = AdminQuestionSerializer(data=test_case['data'])
    if serializer.is_valid():
        print("✓ Validation PASSED")
        try:
            question = serializer.save()
            print(f"✓ Question created: ID {question.id}")
            question.delete()  # Clean up
            print("✓ Test question deleted")
        except Exception as e:
            print(f"✗ Creation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ Validation FAILED")
        for field, errors in serializer.errors.items():
            print(f"  {field}: {errors}")

print("\n" + "="*70 + "\n")
