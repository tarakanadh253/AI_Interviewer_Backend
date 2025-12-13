#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix 500 errors in admin endpoints
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
print("FIXING 500 ERRORS IN ADMIN ENDPOINTS")
print("="*70 + "\n")

# Fix 1: Ensure all questions have valid topics
print("1. Checking questions for orphaned topics...")
try:
    questions = Question.objects.all()
    orphaned = []
    for q in questions:
        try:
            # Try to access topic
            topic = q.topic
            if not topic:
                orphaned.append(q.id)
        except Exception:
            orphaned.append(q.id)
    
    if orphaned:
        print(f"   Found {len(orphaned)} questions with invalid topics")
        # Get first available topic or create a default one
        default_topic = Topic.objects.first()
        if not default_topic:
            print("   Creating default topic...")
            default_topic = Topic.objects.create(name="General", description="Default topic")
        
        # Fix orphaned questions
        for qid in orphaned:
            try:
                q = Question.objects.get(id=qid)
                q.topic = default_topic
                q.save()
                print(f"   ✓ Fixed question {qid}")
            except Exception as e:
                print(f"   ✗ Could not fix question {qid}: {e}")
    else:
        print("   ✓ All questions have valid topics")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Fix 2: Ensure database integrity
print("\n2. Verifying database integrity...")
try:
    with connection.cursor() as cursor:
        # Check for questions with NULL topic_id
        cursor.execute("SELECT COUNT(*) FROM questions WHERE topic_id IS NULL;")
        null_topics = cursor.fetchone()[0]
        if null_topics > 0:
            print(f"   Found {null_topics} questions with NULL topic_id")
            # Get first topic
            cursor.execute("SELECT id FROM topics LIMIT 1;")
            result = cursor.fetchone()
            if result:
                default_topic_id = result[0]
                cursor.execute("UPDATE questions SET topic_id = ? WHERE topic_id IS NULL;", [default_topic_id])
                connection.commit()
                print(f"   ✓ Fixed {null_topics} questions")
            else:
                print("   ⚠ No topics available to fix with")
        else:
            print("   ✓ No NULL topic_ids found")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Fix 3: Test serialization
print("\n3. Testing serialization...")
try:
    from interview.serializers import AdminQuestionSerializer
    
    questions = Question.objects.all()[:5]  # Test first 5
    if questions.exists():
        errors = []
        for q in questions:
            try:
                serializer = AdminQuestionSerializer(q)
                data = serializer.data
            except Exception as e:
                errors.append((q.id, str(e)))
        
        if errors:
            print(f"   ✗ Found {len(errors)} serialization errors:")
            for qid, error in errors:
                print(f"      Question {qid}: {error}")
        else:
            print(f"   ✓ Serialization test passed for {questions.count()} questions")
    else:
        print("   ⚠ No questions to test")
        
except Exception as e:
    print(f"   ✗ Serialization test error: {e}")
    import traceback
    traceback.print_exc()

# Fix 4: Ensure at least one topic exists
print("\n4. Ensuring topics exist...")
try:
    if Topic.objects.count() == 0:
        print("   Creating default topics...")
        topics_data = [
            ("Python", "Python programming questions"),
            ("SQL", "SQL database questions"),
            ("DSA", "Data Structures and Algorithms"),
        ]
        for name, desc in topics_data:
            Topic.objects.get_or_create(name=name, defaults={'description': desc})
        print("   ✓ Default topics created")
    else:
        print(f"   ✓ {Topic.objects.count()} topics exist")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*70)
print("FIX COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Restart your Django server")
print("2. Try accessing /api/admin/questions/ again")
print("="*70 + "\n")
