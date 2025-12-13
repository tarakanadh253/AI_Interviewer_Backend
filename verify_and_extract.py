#!/usr/bin/env python
"""
Comprehensive script to verify installation and extract questions
"""
import sys
import os

# Fix: Add custom packages directory to path if it exists
custom_packages = r"D:\python-packages"
if os.path.exists(custom_packages) and custom_packages not in sys.path:
    sys.path.insert(0, custom_packages)

print("="*60)
print("Question Extraction - Setup Verification")
print("="*60)

# Step 1: Check Python packages
print("\n[Step 1] Checking Python packages...")
try:
    import bs4
    from bs4 import BeautifulSoup
    print(f"  ✓ beautifulsoup4 is installed")
except ImportError:
    print("  ✗ beautifulsoup4 is NOT installed")
    print("  Please run: python -m pip install beautifulsoup4 requests")
    print("  Then run this script again.")
    sys.exit(1)

try:
    import requests
    print(f"  ✓ requests is installed")
except ImportError:
    print("  ✗ requests is NOT installed")
    print("  Please run: python -m pip install beautifulsoup4 requests")
    print("  Then run this script again.")
    sys.exit(1)

# Step 2: Setup Django
print("\n[Step 2] Setting up Django...")
try:
    import django
    sys.path.insert(0, '.')
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
    django.setup()
    print("  ✓ Django setup complete")
except Exception as e:
    print(f"  ✗ Django setup failed: {e}")
    sys.exit(1)

# Step 3: Check models
print("\n[Step 3] Checking database...")
try:
    from interview.models import Question
    link_count = Question.objects.filter(source_type='LINK').count()
    manual_count = Question.objects.filter(source_type='MANUAL').count()
    print(f"  ✓ Database connection successful")
    print(f"  - LINK-type questions: {link_count}")
    print(f"  - MANUAL-type questions: {manual_count}")
except Exception as e:
    print(f"  ✗ Database check failed: {e}")
    sys.exit(1)

# Step 4: Check extraction utility
print("\n[Step 4] Checking extraction utility...")
try:
    from interview.utils.question_extractor import process_link_question
    print("  ✓ Extraction utility is ready")
except Exception as e:
    print(f"  ✗ Extraction utility failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Process LINK questions if any exist
if link_count > 0:
    print("\n[Step 5] Extracting questions from links...")
    print("-"*60)
    
    link_questions = Question.objects.filter(source_type='LINK')
    total_extracted = 0
    
    for link_question in link_questions:
        print(f"\nProcessing Question ID {link_question.id}:")
        print(f"  Topic: {link_question.topic.name}")
        print(f"  Difficulty: {link_question.difficulty}")
        
        links = link_question.get_reference_links_list()
        print(f"  URLs: {len(links)}")
        for i, link in enumerate(links[:3], 1):  # Show first 3
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
    
    # Show updated counts
    new_manual_count = Question.objects.filter(source_type='MANUAL').count()
    print(f"Total MANUAL questions now: {new_manual_count} (was {manual_count})")
else:
    print("\n[Step 5] No LINK-type questions found.")
    print("  Create a LINK-type question in Admin Dashboard to extract questions from URLs.")

print("\n" + "="*60)
print("Verification complete!")
print("="*60)
print("\nNext steps:")
print("1. Check Admin Dashboard → Questions tab for extracted questions")
print("2. Start an interview to see the questions (they'll be shuffled)")
print("="*60)
