#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix Question Bank tab - ensure questions can be listed and serialized
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from interview.models import Question, Topic, Answer
from interview.serializers import AdminQuestionSerializer
from django.db import connection

print("\n" + "="*70)
print("FIXING QUESTION BANK TAB")
print("="*70 + "\n")

# Step 1: Ensure topics exist
print("1. Checking topics...")
topics = Topic.objects.all()
if topics.count() == 0:
    print("   Creating default topics...")
    default_topics = [
        ("Python", "Python programming questions"),
        ("SQL", "SQL database questions"),
        ("DSA", "Data Structures and Algorithms"),
    ]
    for name, desc in default_topics:
        Topic.objects.get_or_create(name=name, defaults={'description': desc})
    print(f"   ✓ Created {len(default_topics)} topics")
else:
    print(f"   ✓ {topics.count()} topics exist")

# Step 2: Fix questions with invalid topics
print("\n2. Checking questions for issues...")
questions = Question.objects.all()
print(f"   Total questions: {questions.count()}")

if questions.exists():
    fixed = 0
    for q in questions:
        try:
            # Test if topic is accessible
            topic = q.topic
            if not topic:
                # Fix orphaned question
                default_topic = Topic.objects.first()
                if default_topic:
                    q.topic = default_topic
                    q.save()
                    fixed += 1
                    print(f"   ✓ Fixed question {q.id} - assigned to topic {default_topic.name}")
        except Exception as e:
            # Question has invalid topic_id
            default_topic = Topic.objects.first()
            if default_topic:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("UPDATE questions SET topic_id = ? WHERE id = ?", [default_topic.id, q.id])
                    fixed += 1
                    print(f"   ✓ Fixed question {q.id} - updated topic_id")
                except Exception as e2:
                    print(f"   ✗ Could not fix question {q.id}: {e2}")
    
    if fixed > 0:
        print(f"\n   ✓ Fixed {fixed} questions")
    else:
        print("   ✓ All questions are valid")

# Step 3: Test serialization
print("\n3. Testing question serialization...")
questions = Question.objects.all()[:5]
if questions.exists():
    errors = []
    for q in questions:
        try:
            serializer = AdminQuestionSerializer(q)
            data = serializer.data
            # Verify all required fields are present
            required_fields = ['id', 'topic', 'topic_name', 'source_type', 'question_text', 'difficulty', 'is_active']
            missing = [f for f in required_fields if f not in data]
            if missing:
                errors.append((q.id, f"Missing fields: {missing}"))
        except Exception as e:
            errors.append((q.id, str(e)))
    
    if errors:
        print(f"   ✗ Found {len(errors)} serialization errors:")
        for qid, error in errors:
            print(f"      Question {qid}: {error}")
    else:
        print(f"   ✓ All {questions.count()} questions serialize correctly")
else:
    print("   ⚠ No questions to test")

# Step 4: Test the API endpoint
print("\n4. Testing API endpoint...")
try:
    from rest_framework.test import APIRequestFactory
    from interview.views import AdminQuestionViewSet
    
    factory = APIRequestFactory()
    request = factory.get('/api/admin/questions/')
    
    viewset = AdminQuestionViewSet()
    viewset.request = request
    viewset.format_kwarg = None
    
    response = viewset.list(request)
    if response.status_code == 200:
        print(f"   ✓ API endpoint works!")
        print(f"   Questions returned: {len(response.data)}")
    else:
        print(f"   ✗ API endpoint returned status {response.status_code}")
        print(f"   Response: {response.data}")
except Exception as e:
    print(f"   ✗ API endpoint error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("FIX COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Restart your Django server")
print("2. Refresh the Question Bank tab in your frontend")
print("="*70 + "\n")
