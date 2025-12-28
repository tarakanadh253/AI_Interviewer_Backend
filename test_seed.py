#!/usr/bin/env python
"""Test if topics are seeded and API is working"""
import os
import sys
import django
import requests

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import Topic
from django.core.management import call_command

print("=" * 60)
print("Testing Topics Setup")
print("=" * 60)

# Check database
topics = Topic.objects.all()
topic_count = topics.count()

print(f"\n1. Database check: {topic_count} topics found")

if topic_count == 0:
    print("\n   ⚠️  No topics in database! Seeding now...")
    try:
        call_command('seed_data', verbosity=1)
        topics = Topic.objects.all()
        topic_count = topics.count()
        print(f"   ✅ Seeded {topic_count} topics successfully!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
else:
    print("   ✅ Topics already exist in database")

# List topics
print("\n2. Topics in database:")
for topic in topics:
    q_count = topic.questions.filter(is_active=True).count()
    print(f"   - {topic.name}: {q_count} questions")

# Test API
print("\n3. Testing API endpoint...")
try:
    response = requests.get('http://localhost:8000/api/topics/', timeout=5)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            print(f"   ✅ API is working! Returned {len(data)} topics")
            for topic in data:
                print(f"      - {topic.get('name')}: {topic.get('question_count', 0)} questions")
        else:
            print("   ⚠️  API returned empty array")
    else:
        print(f"   ❌ API returned status code: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ Cannot connect to API - server may not be running")
    print("   💡 Start server with: python manage.py runserver")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
if topic_count > 0:
    print("✅ Setup complete! Refresh your frontend to see topics.")
else:
    print("❌ Setup incomplete. Please check errors above.")
print("=" * 60)
