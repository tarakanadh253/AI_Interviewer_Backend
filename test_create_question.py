#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test creating a question to diagnose the 500 error
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from interview.models import Question, Topic
from interview.serializers import AdminQuestionSerializer

print("\n" + "="*70)
print("TESTING QUESTION CREATION")
print("="*70 + "\n")

# Check topics
topics = Topic.objects.all()
print(f"Available topics: {topics.count()}")
for topic in topics:
    print(f"  - {topic.name} (ID: {topic.id})")

if topics.count() == 0:
    print("\n⚠ No topics found! Creating a default topic...")
    default_topic = Topic.objects.create(name="General", description="Default topic")
    print(f"✓ Created topic: {default_topic.name} (ID: {default_topic.id})")
    topic_id = default_topic.id
else:
    topic_id = topics.first().id

# Test data
test_data = {
    'topic': topic_id,
    'source_type': 'MANUAL',
    'question_text': 'What is Python?',
    'ideal_answer': 'Python is a high-level programming language.',
    'difficulty': 'EASY',
    'is_active': True
}

print(f"\nTest data: {test_data}")

# Test serialization
print("\n1. Testing serializer validation...")
try:
    serializer = AdminQuestionSerializer(data=test_data)
    if serializer.is_valid():
        print("   ✓ Validation passed")
        print(f"   Validated data: {serializer.validated_data}")
    else:
        print(f"   ✗ Validation failed: {serializer.errors}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Validation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test creation
print("\n2. Testing question creation...")
try:
    question = serializer.save()
    print(f"   ✓ Question created successfully!")
    print(f"   ID: {question.id}")
    print(f"   Topic: {question.topic.name}")
    print(f"   Question: {question.question_text[:50]}...")
    
    # Clean up
    question.delete()
    print("   ✓ Test question deleted")
except Exception as e:
    print(f"   ✗ Creation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("TEST PASSED!")
print("="*70 + "\n")
