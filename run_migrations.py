#!/usr/bin/env python
"""Run Django migrations to fix the django_session table issue."""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.core.management import execute_from_command_line

if __name__ == '__main__':
    print("Running Django migrations...")
    print("=" * 50)
    
    # Run migrations
    execute_from_command_line(['manage.py', 'migrate', '--verbosity', '2'])
    
    print("=" * 50)
    print("Migrations completed!")
    
    # Check if database file exists
    from django.conf import settings
    db_path = settings.DATABASES['default']['NAME']
    if os.path.exists(db_path):
        print(f"✓ Database file created at: {db_path}")
    else:
        print(f"✗ Database file not found at: {db_path}")
