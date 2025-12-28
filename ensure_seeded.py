#!/usr/bin/env python
"""Ensure topics are seeded - with visible output"""
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import Topic, Question
from django.core.management import call_command

# Force output
sys.stdout.write("Checking topics...\n")
sys.stdout.flush()

topics = Topic.objects.all()
count = topics.count()

sys.stdout.write(f"Found {count} topics in database\n")
sys.stdout.flush()

if count == 0:
    sys.stdout.write("Seeding topics now...\n")
    sys.stdout.flush()
    call_command('seed_data', verbosity=2)
    topics = Topic.objects.all()
    count = topics.count()
    sys.stdout.write(f"Seeded {count} topics!\n")
    sys.stdout.flush()

sys.stdout.write("\nTopics:\n")
for topic in topics:
    q_count = topic.questions.filter(is_active=True).count()
    sys.stdout.write(f"  - {topic.name} ({q_count} questions)\n")
    sys.stdout.flush()

sys.stdout.write(f"\nTotal: {count} topics, {Question.objects.filter(is_active=True).count()} questions\n")
sys.stdout.flush()
