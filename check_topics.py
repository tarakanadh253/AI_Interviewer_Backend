#!/usr/bin/env python
"""Quick script to check if topics exist in the database"""
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import Topic

print("="*60)
print("Checking Topics in Database")
print("="*60)

topics = Topic.objects.all()
print(f"\nTotal topics: {topics.count()}")

if topics.count() > 0:
    print("\nTopics found:")
    for topic in topics:
        print(f"  - {topic.name} (ID: {topic.id})")
        print(f"    Description: {topic.description or 'None'}")
        print(f"    Questions: {topic.questions.count()}")
else:
    print("\n⚠️  No topics found!")
    print("\nTo seed topics, run:")
    print("  python manage.py seed_data")

print("\n" + "="*60)
