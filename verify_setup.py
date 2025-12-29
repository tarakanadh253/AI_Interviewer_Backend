#!/usr/bin/env python
"""Verify backend setup and seed topics if needed"""
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.core.management import call_command
from interview.models import Topic, Question

print("Checking backend setup...")
print("-" * 50)

# Check topics
topics = Topic.objects.all()
topic_count = topics.count()

print(f"Topics in database: {topic_count}")

if topic_count == 0:
    print("\n⚠️  No topics found! Seeding now...")
    try:
        call_command('seed_data', verbosity=1)
        topics = Topic.objects.all()
        topic_count = topics.count()
        print(f"\n✅ Seeded {topic_count} topics successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

print("\nTopics:")
for topic in topics:
    q_count = topic.questions.filter(is_active=True).count()
    print(f"  - {topic.name} ({q_count} questions)")

total_questions = Question.objects.filter(is_active=True).count()
print(f"\nTotal active questions: {total_questions}")

print("\n" + "-" * 50)
print("✅ Setup complete!")
print("\nTo start the server, run:")
print("  python manage.py runserver")
print("\nThen test the API at:")
print("  http://localhost:8000/api/topics/")
