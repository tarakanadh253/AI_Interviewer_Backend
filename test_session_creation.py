#!/usr/bin/env python
"""Test session creation to find the error"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from rest_framework.test import APIRequestFactory
from interview.views import InterviewSessionViewSet
from interview.models import UserProfile, Topic

print("=" * 70)
print("TESTING SESSION CREATION")
print("=" * 70)
print()

# Check if answers table exists
import sqlite3
conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='answers'")
has_answers = c.fetchone() is not None
conn.close()
print(f"answers table exists: {has_answers}")
print()

# Get a test user
try:
    user = UserProfile.objects.first()
    if not user:
        print("ERROR: No users found in database!")
        sys.exit(1)
    print(f"Test user: {user.username}")
except Exception as e:
    print(f"ERROR getting user: {e}")
    sys.exit(1)

# Get topics
try:
    topics = Topic.objects.all()[:3]
    topic_ids = [t.id for t in topics]
    if not topic_ids:
        print("ERROR: No topics found in database!")
        sys.exit(1)
    print(f"Test topic IDs: {topic_ids}")
except Exception as e:
    print(f"ERROR getting topics: {e}")
    sys.exit(1)

print()
print("Testing session creation...")
print()

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
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 201:
        print("✓ SUCCESS! Session created")
        print(f"Session ID: {response.data.get('id')}")
    else:
        print(f"✗ ERROR: {response.data}")
        
except Exception as e:
    print(f"✗ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
