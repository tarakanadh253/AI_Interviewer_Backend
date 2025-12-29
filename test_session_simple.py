#!/usr/bin/env python
"""Simple test to see what happens"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    django.setup()
    print("Django setup: OK")
    
    from interview.models import UserProfile, Topic, InterviewSession
    from interview.serializers import InterviewSessionSerializer
    
    user = UserProfile.objects.first()
    if not user:
        print("ERROR: No user found")
        sys.exit(1)
    
    topics = Topic.objects.all()[:2]
    topic_ids = [t.id for t in topics]
    if not topic_ids:
        print("ERROR: No topics found")
        sys.exit(1)
    
    print(f"User: {user.username}")
    print(f"Topics: {topic_ids}")
    
    # Try to create a session directly
    print("\nCreating session...")
    session = InterviewSession.objects.create(user=user, status='IN_PROGRESS')
    session.topics.set(topic_ids)
    print(f"Session created: {session.id}")
    
    # Try to serialize
    print("\nSerializing session...")
    serializer = InterviewSessionSerializer(session)
    data = serializer.data
    print(f"Serialization: SUCCESS")
    print(f"Keys: {list(data.keys())}")
    print(f"Answers: {data.get('answers')}")
    print(f"Answer count: {data.get('answer_count')}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
