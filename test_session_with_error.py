#!/usr/bin/env python
"""Test session creation and capture full error"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    django.setup()
    print("Django setup: OK\n")
    
    from interview.models import UserProfile, Topic
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
    print(f"Topics: {topic_ids}\n")
    
    # Create request
    factory = APIRequestFactory()
    request = factory.post('/api/sessions/', {
        'username': user.username,
        'topic_ids': topic_ids
    }, format='json')
    
    viewset = InterviewSessionViewSet()
    viewset.request = request
    viewset.format_kwarg = None
    
    print("Calling create()...")
    try:
        response = viewset.create(request)
        print(f"\n✓ Status: {response.status_code}")
        if response.status_code == 201:
            print("✓ SUCCESS!")
        else:
            print(f"✗ Error: {response.data}")
    except Exception as e:
        print(f"\n✗ EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
except Exception as e:
    print(f"FATAL: {e}")
    import traceback
    traceback.print_exc()
