#!/usr/bin/env python
"""Comprehensive diagnostic for topics not showing"""
import os
import sys
import django
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import Topic, Question
from interview.serializers import TopicSerializer
from django.core.management import call_command
import urllib.request

print("=" * 70)
print("DIAGNOSTIC: Topics Not Showing Issue")
print("=" * 70)

# 1. Check database
print("\n1. DATABASE CHECK")
print("-" * 70)
topics = Topic.objects.all()
topic_count = topics.count()
print(f"   Topics in database: {topic_count}")

if topic_count == 0:
    print("   ⚠️  NO TOPICS FOUND! Seeding now...")
    try:
        call_command('seed_data', verbosity=1)
        topics = Topic.objects.all()
        topic_count = topics.count()
        print(f"   ✅ Seeded {topic_count} topics")
    except Exception as e:
        print(f"   ❌ Seeding failed: {e}")
        sys.exit(1)
else:
    print("   ✅ Topics exist in database")
    for topic in topics:
        q_count = topic.questions.filter(is_active=True).count()
        print(f"      - {topic.name} (ID: {topic.id}): {q_count} questions")

# 2. Check serializer
print("\n2. SERIALIZER CHECK")
print("-" * 70)
if topic_count > 0:
    topic = topics.first()
    serializer = TopicSerializer(topic)
    serialized_data = serializer.data
    print(f"   Serialized topic '{topic.name}':")
    print(f"      {json.dumps(serialized_data, indent=6)}")
    
    # Check all topics
    all_serialized = TopicSerializer(topics, many=True).data
    print(f"\n   All topics serialized: {len(all_serialized)} items")
    if len(all_serialized) > 0:
        print(f"   First topic keys: {list(all_serialized[0].keys())}")

# 3. Test API endpoint
print("\n3. API ENDPOINT CHECK")
print("-" * 70)
try:
    url = 'http://localhost:8000/api/topics/'
    print(f"   Testing: {url}")
    
    with urllib.request.urlopen(url, timeout=5) as response:
        status = response.getcode()
        data = json.loads(response.read().decode())
        
        print(f"   Status code: {status}")
        print(f"   Response type: {type(data)}")
        print(f"   Response length: {len(data) if isinstance(data, list) else 'N/A'}")
        
        if isinstance(data, list):
            if len(data) > 0:
                print(f"   ✅ API is working! Found {len(data)} topics:")
                for item in data:
                    print(f"      - {item.get('name', 'N/A')}: {item.get('question_count', 0)} questions")
            else:
                print("   ⚠️  API returned EMPTY ARRAY!")
                print("   This is the problem - API is working but returning no data")
        else:
            print(f"   ⚠️  API returned non-list: {type(data)}")
            print(f"   Response: {json.dumps(data, indent=6)}")
            
except urllib.error.URLError as e:
    print(f"   ❌ Cannot connect to API: {e}")
    print("   💡 Make sure backend server is running: python manage.py runserver")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Check ViewSet
print("\n4. VIEWSET CHECK")
print("-" * 70)
from interview.views import TopicViewSet
viewset = TopicViewSet()
queryset = viewset.get_queryset()
print(f"   ViewSet queryset count: {queryset.count()}")
print(f"   ViewSet permission classes: {viewset.permission_classes}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if topic_count > 0:
    print("✅ Topics exist in database")
    print("✅ Serializer should work")
    print("\n💡 Next steps:")
    print("   1. Check browser console (F12) for errors")
    print("   2. Check Network tab to see API response")
    print("   3. Verify CORS is allowing requests from frontend")
    print("   4. Test API directly: http://localhost:8000/api/topics/")
else:
    print("❌ No topics in database - seeding failed")
print("=" * 70)
