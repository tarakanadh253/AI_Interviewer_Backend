#!/usr/bin/env python
"""Setup script to ensure database is ready and topics are seeded"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.core.management import call_command
from interview.models import Topic, Question

print("=" * 60)
print("AI Interview Buddy - Backend Setup Check")
print("=" * 60)

# Check migrations
print("\n1. Checking database migrations...")
try:
    call_command('migrate', verbosity=0, interactive=False)
    print("   ✅ Migrations are up to date")
except Exception as e:
    print(f"   ❌ Migration error: {e}")
    sys.exit(1)

# Check topics
print("\n2. Checking topics...")
topics = Topic.objects.all()
topic_count = topics.count()

if topic_count == 0:
    print("   ⚠️  No topics found. Seeding topics and questions...")
    try:
        call_command('seed_data', verbosity=1)
        topics = Topic.objects.all()
        topic_count = topics.count()
        print(f"   ✅ Successfully seeded {topic_count} topics")
    except Exception as e:
        print(f"   ❌ Error seeding data: {e}")
        sys.exit(1)
else:
    print(f"   ✅ Found {topic_count} topics in database")

# Display topics
print("\n3. Topics in database:")
for topic in topics:
    question_count = topic.questions.filter(is_active=True).count()
    print(f"   - {topic.name}: {question_count} active questions")

# Summary
total_questions = Question.objects.filter(is_active=True).count()
print(f"\n📊 Summary:")
print(f"   - Topics: {topic_count}")
print(f"   - Active Questions: {total_questions}")

print("\n" + "=" * 60)
print("✅ Backend is ready!")
print("\nNext steps:")
print("   1. Start the backend server: python manage.py runserver")
print("   2. The API will be available at: http://localhost:8000/api/")
print("   3. Test topics endpoint: http://localhost:8000/api/topics/")
print("=" * 60)
