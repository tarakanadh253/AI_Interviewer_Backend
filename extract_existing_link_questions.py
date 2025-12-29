#!/usr/bin/env python
"""
Script to extract questions from existing LINK-type questions that haven't been processed yet
"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

django.setup()

from interview.models import Question
from interview.utils.question_extractor import process_link_question
import logging

logger = logging.getLogger(__name__)

def main():
    import sys
    
    # Find all LINK-type questions
    link_questions = Question.objects.filter(source_type='LINK')
    
    count = link_questions.count()
    sys.stdout.write(f"Found {count} LINK-type questions\n")
    sys.stdout.flush()
    
    if count == 0:
        sys.stdout.write("\nNo LINK-type questions found. Nothing to extract.\n")
        sys.stdout.flush()
        return
    
    total_extracted = 0
    for link_question in link_questions:
        sys.stdout.write(f"\nProcessing question ID {link_question.id} (Topic: {link_question.topic.name})...\n")
        sys.stdout.flush()
        
        links = link_question.get_reference_links_list()
        sys.stdout.write(f"  Links: {len(links)}\n")
        sys.stdout.flush()
        
        for link in links:
            sys.stdout.write(f"    - {link}\n")
            sys.stdout.flush()
        
        try:
            created_count = process_link_question(link_question)
            total_extracted += created_count
            sys.stdout.write(f"  ✓ Extracted {created_count} questions\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"  ✗ Error: {e}\n")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
    
    sys.stdout.write("\n" + "="*50 + "\n")
    sys.stdout.write(f"Done! Extracted {total_extracted} total questions.\n")
    sys.stdout.write("Check the questions table for extracted questions.\n")
    sys.stdout.write("="*50 + "\n")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
