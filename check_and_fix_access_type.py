#!/usr/bin/env python
"""Check and fix access_type field in UserProfile table"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from django.core.management import execute_from_command_line

def check_access_type_field():
    """Check if access_type field exists in user_profiles table"""
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(user_profiles);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print("Current columns in user_profiles table:")
        for col in column_names:
            print(f"  - {col}")
        
        if 'access_type' in column_names:
            print("\n✓ access_type field exists!")
            return True
        else:
            print("\n✗ access_type field is MISSING!")
            return False

def apply_migration():
    """Apply the migration"""
    print("\nApplying migration...")
    try:
        execute_from_command_line(['manage.py', 'migrate', 'interview', '0003_add_access_type'])
        print("Migration applied successfully!")
        return True
    except Exception as e:
        print(f"Error applying migration: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Checking access_type field in UserProfile")
    print("=" * 60)
    
    has_field = check_access_type_field()
    
    if not has_field:
        print("\nField is missing. Attempting to apply migration...")
        if apply_migration():
            print("\nRe-checking...")
            check_access_type_field()
        else:
            print("\nMigration failed. You may need to run:")
            print("  python manage.py migrate interview")
    else:
        print("\nDatabase is up to date!")
