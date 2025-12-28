#!/usr/bin/env python
"""
Fix import issue by adding custom package directory to path
"""
import sys
import os

# Add the custom packages directory to Python path
custom_packages = r"D:\python-packages"
if os.path.exists(custom_packages) and custom_packages not in sys.path:
    sys.path.insert(0, custom_packages)
    print(f"Added {custom_packages} to Python path")

# Now try to import
try:
    import bs4
    from bs4 import BeautifulSoup
    print("✓ beautifulsoup4 imported successfully")
except ImportError as e:
    print(f"✗ beautifulsoup4 import failed: {e}")
    sys.exit(1)

try:
    import requests
    print("✓ requests imported successfully")
except ImportError as e:
    print(f"✗ requests import failed: {e}")
    sys.exit(1)

print("\nAll packages imported! Now running extraction...")
print("="*60 + "\n")

# Continue with the extraction script
import django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import Question
from interview.utils.question_extractor import process_link_question

link_count = Question.objects.filter(source_type='LINK').count()
manual_count = Question.objects.filter(source_type='MANUAL').count()

print(f"Database connection successful")
print(f"  - LINK-type questions: {link_count}")
print(f"  - MANUAL-type questions: {manual_count}")

if link_count > 0:
    print("\n" + "-"*60)
    print("Extracting questions from links...")
    print("-"*60)
    
    link_questions = Question.objects.filter(source_type='LINK')
    total_extracted = 0
    
    for link_question in link_questions:
        print(f"\nProcessing Question ID {link_question.id}:")
        print(f"  Topic: {link_question.topic.name}")
        print(f"  Difficulty: {link_question.difficulty}")
        
        links = link_question.get_reference_links_list()
        print(f"  URLs: {len(links)}")
        for i, link in enumerate(links[:3], 1):
            print(f"    {i}. {link[:70]}...")
        
        try:
            created_count = process_link_question(link_question)
            total_extracted += created_count
            if created_count > 0:
                print(f"  ✓ Successfully extracted {created_count} questions")
            else:
                print(f"  ⚠ No questions extracted (check URLs or website structure)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "-"*60)
    print(f"Total questions extracted: {total_extracted}")
    
    new_manual_count = Question.objects.filter(source_type='MANUAL').count()
    print(f"Total MANUAL questions now: {new_manual_count} (was {manual_count})")
else:
    print("\nNo LINK-type questions found.")
    print("Create a LINK-type question in Admin Dashboard to extract questions from URLs.")

print("\n" + "="*60)
print("Done!")
print("="*60)


