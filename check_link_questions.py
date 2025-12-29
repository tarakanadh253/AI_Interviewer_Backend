#!/usr/bin/env python
"""Quick check for LINK-type questions"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

django.setup()

from interview.models import Question

link_questions = Question.objects.filter(source_type='LINK')
manual_questions = Question.objects.filter(source_type='MANUAL')

print("="*60)
print(f"LINK-type questions: {link_questions.count()}")
print(f"MANUAL-type questions: {manual_questions.count()}")
print("="*60)

if link_questions.exists():
    print("\nLINK questions found:")
    for q in link_questions:
        links = q.get_reference_links_list()
        print(f"  ID {q.id}: Topic={q.topic.name}, Links={len(links)}")
        for link in links[:2]:  # Show first 2 links
            print(f"    - {link[:60]}...")
else:
    print("\nNo LINK-type questions found.")
