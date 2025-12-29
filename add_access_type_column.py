#!/usr/bin/env python
"""Directly add access_type column to user_profiles table"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

def main():
    print("Checking user_profiles table schema...")
    
    with connection.cursor() as cursor:
        # Check current columns
        cursor.execute("PRAGMA table_info(user_profiles);")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print(f"Current columns: {', '.join(column_names)}")
        
        if 'access_type' in column_names:
            print("✓ access_type column already exists!")
            return
        
        print("\nAdding access_type column...")
        
        try:
            # SQLite doesn't support adding columns with DEFAULT in some versions
            # So we add it first, then update existing rows
            cursor.execute("""
                ALTER TABLE user_profiles 
                ADD COLUMN access_type VARCHAR(10);
            """)
            
            # Set default value for existing rows
            cursor.execute("""
                UPDATE user_profiles 
                SET access_type = 'TRIAL' 
                WHERE access_type IS NULL;
            """)
            
            print("✓ Successfully added access_type column!")
            print("✓ Set default value 'TRIAL' for existing users")
            
            # Verify
            cursor.execute("PRAGMA table_info(user_profiles);")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            print(f"\nUpdated columns: {', '.join(column_names)}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
