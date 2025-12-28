#!/usr/bin/env python
"""Fix access_type field - check database and apply migration"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from django.core.management import call_command

def main():
    print("=" * 70)
    print("Checking and fixing access_type field")
    print("=" * 70)
    
    # Check current schema
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(user_profiles);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print("\nCurrent columns in user_profiles table:")
        for col in column_names:
            print(f"  - {col}")
        
        if 'access_type' in column_names:
            print("\n✓ access_type field already exists!")
            return
        
    print("\n✗ access_type field is MISSING!")
    print("\nApplying migration...")
    
    try:
        # Run migration
        call_command('migrate', 'interview', '0003_add_access_type', verbosity=2)
        print("\n✓ Migration applied!")
        
        # Verify
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA table_info(user_profiles);")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'access_type' in column_names:
                print("✓ Verified: access_type field now exists!")
            else:
                print("✗ ERROR: access_type field still missing after migration!")
                
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTrying alternative: direct SQL...")
        
        # Try direct SQL as fallback
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    ALTER TABLE user_profiles 
                    ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';
                """)
                print("✓ Added access_type column via direct SQL")
        except Exception as sql_error:
            print(f"✗ SQL error: {sql_error}")

if __name__ == '__main__':
    main()
