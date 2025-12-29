#!/usr/bin/env python
"""Comprehensive diagnosis of session creation error"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

output = []

try:
    django.setup()
    output.append("✓ Django setup successful")
    
    # Check database tables
    import sqlite3
    conn = sqlite3.connect('db.sqlite3')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    conn.close()
    output.append(f"✓ Database tables: {', '.join(sorted(tables))}")
    
    required_tables = ['answers', 'interview_sessions', 'questions', 'topics', 'user_profiles']
    missing = [t for t in required_tables if t not in tables]
    if missing:
        output.append(f"✗ Missing tables: {', '.join(missing)}")
    else:
        output.append("✓ All required tables exist")
    
    # Check models
    from interview.models import UserProfile, Topic, InterviewSession
    output.append("✓ Models imported successfully")
    
    # Check users
    users = UserProfile.objects.all()
    output.append(f"✓ Users in DB: {users.count()}")
    if users.exists():
        user = users.first()
        output.append(f"  - First user: {user.username} (ID: {user.id}, Active: {user.is_active})")
    else:
        output.append("✗ NO USERS FOUND - This is a problem!")
    
    # Check topics
    topics = Topic.objects.all()
    output.append(f"✓ Topics in DB: {topics.count()}")
    if topics.exists():
        topic_ids = [t.id for t in topics[:3]]
        output.append(f"  - First 3 topic IDs: {topic_ids}")
    else:
        output.append("✗ NO TOPICS FOUND - This is a problem!")
    
    # Test session creation
    if users.exists() and topics.exists():
        user = users.first()
        topic_ids = [t.id for t in topics[:2]]
        
        output.append("\nTesting session creation...")
        try:
            from rest_framework.test import APIRequestFactory
            from interview.views import InterviewSessionViewSet
            
            factory = APIRequestFactory()
            request = factory.post('/api/sessions/', {
                'username': user.username,
                'topic_ids': topic_ids
            }, format='json')
            
            viewset = InterviewSessionViewSet()
            viewset.request = request
            viewset.format_kwarg = None
            
            response = viewset.create(request)
            output.append(f"✓ Response status: {response.status_code}")
            
            if response.status_code == 201:
                output.append("✓ SUCCESS! Session created")
                output.append(f"  - Session ID: {response.data.get('id')}")
                output.append(f"  - Status: {response.data.get('status')}")
            else:
                output.append(f"✗ ERROR: Status {response.status_code}")
                output.append(f"  - Error data: {response.data}")
                
        except Exception as e:
            output.append(f"✗ EXCEPTION during session creation: {e}")
            import traceback
            output.append("Traceback:")
            for line in traceback.format_exc().split('\n'):
                output.append(f"  {line}")
    
    # Test serializer directly
    if users.exists() and topics.exists():
        user = users.first()
        topic_ids = [t.id for t in topics[:2]]
        
        output.append("\nTesting serializer directly...")
        try:
            session = InterviewSession.objects.create(user=user, status='IN_PROGRESS')
            session.topics.set(topic_ids)
            output.append(f"✓ Session created directly: {session.id}")
            
            from interview.serializers import InterviewSessionSerializer
            serializer = InterviewSessionSerializer(session)
            data = serializer.data
            output.append(f"✓ Serialization successful")
            output.append(f"  - Keys: {', '.join(data.keys())}")
            output.append(f"  - Answers: {len(data.get('answers', []))} items")
            output.append(f"  - Answer count: {data.get('answer_count')}")
            
            # Clean up test session
            session.delete()
            output.append("✓ Test session cleaned up")
            
        except Exception as e:
            output.append(f"✗ EXCEPTION during serialization: {e}")
            import traceback
            output.append("Traceback:")
            for line in traceback.format_exc().split('\n'):
                output.append(f"  {line}")
    
except Exception as e:
    output.append(f"✗ FATAL ERROR: {e}")
    import traceback
    for line in traceback.format_exc().split('\n'):
        output.append(f"  {line}")

# Print and save
result = '\n'.join(output)
print(result)
with open('diagnosis_results.txt', 'w', encoding='utf-8') as f:
    f.write(result)
