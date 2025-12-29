#!/usr/bin/env python
"""Capture the exact error when creating a session"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import traceback

try:
    django.setup()
    print("=" * 70)
    print("TESTING SESSION CREATION - CAPTURING ERROR")
    print("=" * 70)
    print()
    
    from interview.models import UserProfile, Topic, InterviewSession
    from interview.serializers import InterviewSessionSerializer, InterviewSessionCreateSerializer
    from rest_framework.test import APIRequestFactory
    from interview.views import InterviewSessionViewSet
    
    # Get user and topics
    user = UserProfile.objects.first()
    if not user:
        print("ERROR: No users found!")
        sys.exit(1)
    
    topics = Topic.objects.all()[:2]
    topic_ids = [t.id for t in topics]
    if not topic_ids:
        print("ERROR: No topics found!")
        sys.exit(1)
    
    print(f"User: {user.username} (ID: {user.id})")
    print(f"Topics: {topic_ids}")
    print()
    
    # Test 1: Create session directly
    print("Test 1: Creating session directly...")
    try:
        session = InterviewSession.objects.create(user=user, status='IN_PROGRESS')
        session.topics.set(topic_ids)
        print(f"✓ Session created: {session.id}")
    except Exception as e:
        print(f"✗ ERROR creating session: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test 2: Serialize the session
    print("\nTest 2: Serializing session...")
    try:
        serializer = InterviewSessionSerializer(session)
        data = serializer.data
        print(f"✓ Serialization successful")
        print(f"  Keys: {list(data.keys())}")
        print(f"  user_email: {data.get('user_email')}")
        print(f"  user_name: {data.get('user_name')}")
        print(f"  answers: {data.get('answers')}")
        print(f"  answer_count: {data.get('answer_count')}")
    except Exception as e:
        print(f"✗ ERROR serializing: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test 3: Use viewset
    print("\nTest 3: Using viewset create method...")
    try:
        factory = APIRequestFactory()
        request = factory.post('/api/sessions/', {
            'username': user.username,
            'topic_ids': topic_ids
        }, format='json')
        
        viewset = InterviewSessionViewSet()
        viewset.request = request
        viewset.format_kwarg = None
        
        response = viewset.create(request)
        print(f"✓ Response status: {response.status_code}")
        if response.status_code == 201:
            print("✓ SUCCESS!")
        else:
            print(f"✗ Error response: {response.data}")
    except Exception as e:
        print(f"✗ EXCEPTION in viewset: {e}")
        traceback.print_exc()
    
    # Cleanup
    session.delete()
    print("\n✓ Test session cleaned up")
    
except Exception as e:
    print(f"\n✗ FATAL ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
