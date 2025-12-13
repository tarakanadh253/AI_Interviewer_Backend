#!/usr/bin/env python
"""Fix database schema - recreate user_profiles table if needed"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("FIXING DATABASE SCHEMA")
print("=" * 70)

with connection.cursor() as cursor:
    # Check current schema
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    
    print(f"\nCurrent columns: {', '.join(col_names)}")
    
    # Check if username exists
    has_username = 'username' in col_names
    has_access_type = 'access_type' in col_names
    
    print(f"\nHas username: {has_username}")
    print(f"Has access_type: {has_access_type}")
    
    if not has_username:
        print("\n✗ CRITICAL: username column is MISSING!")
        print("The table schema is completely wrong.")
        print("\nRecreating table...")
        
        # Drop old table
        cursor.execute("DROP TABLE IF EXISTS user_profiles")
        print("✓ Old table dropped")
        
        # Create new table with correct schema
        cursor.execute("""
            CREATE TABLE user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(150) UNIQUE NOT NULL,
                password VARCHAR(128) NOT NULL,
                email VARCHAR(254) NOT NULL,
                name VARCHAR(255),
                is_active BOOLEAN NOT NULL DEFAULT 1,
                access_type VARCHAR(10) DEFAULT 'TRIAL',
                has_used_trial BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        print("✓ New table created")
        
        # Create index
        cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles(username)")
        print("✓ Index created")
        
        # Verify
        cursor.execute("PRAGMA table_info(user_profiles)")
        new_cols = [col[1] for col in cursor.fetchall()]
        print(f"\nNew columns: {', '.join(new_cols)}")
        
        if 'username' in new_cols:
            print("\n✓ Table recreated successfully!")
        else:
            print("\n✗ ERROR: Table recreation failed!")
            sys.exit(1)
    else:
        print("\n✓ username column exists")
        
        # Just add access_type if missing
        if not has_access_type:
            print("Adding access_type column...")
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("✓ access_type added!")

print("\n" + "=" * 70)
print("✓ DATABASE SCHEMA FIXED!")
print("=" * 70)
print("\nThe table now has the correct schema.")
print("You can start the server: python manage.py runserver")
