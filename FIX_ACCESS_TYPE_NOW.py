#!/usr/bin/env python
"""Directly fix access_type column - bypasses migration issues"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("FIXING access_type FIELD")
print("=" * 70)

try:
    with connection.cursor() as cursor:
        # Check if column exists
        cursor.execute("PRAGMA table_info(user_profiles);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print(f"\nCurrent columns: {', '.join(column_names)}")
        
        if 'access_type' in column_names:
            print("\n✓ access_type column already exists!")
            print("Database is ready.")
        else:
            print("\n✗ access_type column is MISSING!")
            print("Adding column...")
            
            # Add the column
            cursor.execute("""
                ALTER TABLE user_profiles 
                ADD COLUMN access_type VARCHAR(10);
            """)
            
            # Set default for existing rows
            cursor.execute("""
                UPDATE user_profiles 
                SET access_type = 'TRIAL' 
                WHERE access_type IS NULL;
            """)
            
            print("✓ Column added successfully!")
            
            # Verify
            cursor.execute("PRAGMA table_info(user_profiles);")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            print(f"\nUpdated columns: {', '.join(column_names)}")
            print("\n✓ Database is now ready!")
            
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("SUCCESS! You can now restart your Django server.")
print("=" * 70)
