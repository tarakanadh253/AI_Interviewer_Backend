#!/usr/bin/env python
"""Recreate user_profiles table with correct schema"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("RECREATING user_profiles TABLE")
print("=" * 70)

with connection.cursor() as cursor:
    # Check current schema
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    
    print(f"\nCurrent columns: {', '.join(col_names)}")
    
    # Check if we need to recreate
    if 'username' not in col_names:
        print("\n✗ username column is MISSING!")
        print("Recreating table with correct schema...")
        
        # Backup existing data if any
        try:
            cursor.execute("SELECT COUNT(*) FROM user_profiles")
            count = cursor.fetchone()[0]
            print(f"Found {count} existing records (will be lost)")
        except:
            count = 0
        
        # Drop old table
        print("\nDropping old table...")
        cursor.execute("DROP TABLE IF EXISTS user_profiles")
        
        # Create new table with correct schema
        print("Creating new table...")
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
        
        # Create index
        cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles(username)")
        
        print("✓ Table recreated successfully!")
        
        # Verify
        cursor.execute("PRAGMA table_info(user_profiles)")
        new_cols = [col[1] for col in cursor.fetchall()]
        print(f"\nNew columns: {', '.join(new_cols)}")
        
        if 'username' in new_cols and 'access_type' in new_cols:
            print("\n✓ Table is now correct!")
        else:
            print("\n✗ ERROR: Table still incorrect!")
            sys.exit(1)
    else:
        print("\n✓ username column exists")
        
        # Just ensure access_type exists
        if 'access_type' not in col_names:
            print("Adding access_type column...")
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("✓ access_type added!")

print("\n" + "=" * 70)
print("DATABASE FIX COMPLETE!")
print("=" * 70)
print("\nYou can now start the server:")
print("  python manage.py runserver")
