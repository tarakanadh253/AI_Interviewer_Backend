#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnose 500 errors in admin endpoints
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from django.db import connection
from interview.models import Question, Topic, Answer

print("\n" + "="*70)
print("DIAGNOSING 500 ERRORS")
print("="*70 + "\n")

# Check 1: Verify tables exist
print("1. Checking database tables...")
with connection.cursor() as cursor:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]
    required_tables = ['questions', 'topics', 'answers', 'django_session']
    missing = [t for t in required_tables if t not in tables]
    if missing:
        print(f"   ✗ Missing tables: {missing}")
    else:
        print(f"   ✓ All required tables exist")

# Check 2: Questions and their topics
print("\n2. Checking questions and topics...")
try:
    questions = Question.objects.all()
    print(f"   Total questions: {questions.count()}")
    
    if questions.exists():
        # Check for questions without topics
        questions_without_topic = []
        for q in questions:
            try:
                if not q.topic:
                    questions_without_topic.append(q.id)
            except Exception as e:
                print(f"   ✗ Error accessing question {q.id}: {e}")
                questions_without_topic.append(q.id)
        
        if questions_without_topic:
            print(f"   ✗ Questions without valid topics: {questions_without_topic}")
        else:
            print("   ✓ All questions have valid topics")
        
        # Test serialization
        print("\n3. Testing AdminQuestionSerializer...")
        from interview.serializers import AdminQuestionSerializer
        try:
            first_q = questions.first()
            serializer = AdminQuestionSerializer(first_q)
            data = serializer.data
            print(f"   ✓ Serialization successful for question {first_q.id}")
            print(f"   Topic: {data.get('topic_name', 'N/A')}")
        except Exception as e:
            print(f"   ✗ Serialization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠ No questions in database")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Check 3: Topics
print("\n4. Checking topics...")
try:
    topics = Topic.objects.all()
    print(f"   Total topics: {topics.count()}")
    if topics.exists():
        for topic in topics:
            print(f"   - {topic.name} (ID: {topic.id})")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Check 4: Test the viewset directly
print("\n5. Testing AdminQuestionViewSet...")
try:
    from interview.views import AdminQuestionViewSet
    from rest_framework.test import APIRequestFactory
    from rest_framework.request import Request
    
    factory = APIRequestFactory()
    request = factory.get('/api/admin/questions/')
    django_request = Request(request)
    
    viewset = AdminQuestionViewSet()
    viewset.request = django_request
    viewset.format_kwarg = None
    
    queryset = viewset.get_queryset()
    print(f"   Queryset count: {queryset.count()}")
    
    if queryset.exists():
        try:
            serializer = viewset.get_serializer(queryset, many=True)
            data = serializer.data
            print(f"   ✓ ViewSet serialization successful ({len(data)} items)")
        except Exception as e:
            print(f"   ✗ ViewSet serialization failed: {e}")
            import traceback
            traceback.print_exc()
            
except Exception as e:
    print(f"   ✗ Error testing ViewSet: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70 + "\n")
