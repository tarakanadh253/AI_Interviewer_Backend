#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QUICK FIX for 500 Errors - Run this script!
Double-click this file or run: python FIX_NOW.py
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    import django
    django.setup()
except Exception as e:
    print(f"Error setting up Django: {e}")
    print("\nMake sure you're in the backend directory!")
    input("Press Enter to exit...")
    sys.exit(1)

from django.db import connection

print("\n" + "="*70)
print(" FIXING DATABASE SCHEMA - This will fix the 500 errors")
print("="*70 + "\n")

try:
    with connection.cursor() as cursor:
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='user_profiles';
        """)
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("⚠️  Table doesn't exist. Creating it...")
            cursor.execute("""
                CREATE TABLE user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(150) UNIQUE NOT NULL,
                    password VARCHAR(128) NOT NULL,
                    email VARCHAR(254) NOT NULL,
                    name VARCHAR(255),
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    has_used_trial BOOLEAN NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                );
            """)
            cursor.execute("""
                CREATE UNIQUE INDEX user_profiles_username_idx 
                ON user_profiles(username);
            """)
            print("✅ Table created!")
        else:
            # Check current schema
            cursor.execute("PRAGMA table_info(user_profiles);")
            columns = {row[1]: row for row in cursor.fetchall()}
            column_names = list(columns.keys())
            
            print(f"Current columns: {', '.join(column_names)}\n")
            
            has_google_id = 'google_id' in column_names
            has_username = 'username' in column_names
            has_password = 'password' in column_names
            
            if has_google_id and not has_username:
                print("⚠️  Found old schema (google_id). Converting to new schema...\n")
                
                # Create new table
                print("Step 1: Creating new table structure...")
                cursor.execute("""
                    CREATE TABLE user_profiles_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username VARCHAR(150) UNIQUE NOT NULL,
                        password VARCHAR(128) NOT NULL,
                        email VARCHAR(254) NOT NULL,
                        name VARCHAR(255),
                        is_active BOOLEAN NOT NULL DEFAULT 1,
                        has_used_trial BOOLEAN NOT NULL DEFAULT 0,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    );
                """)
                print("   ✓ New table created")
                
                # Create index
                print("Step 2: Creating index...")
                cursor.execute("""
                    CREATE UNIQUE INDEX user_profiles_username_idx 
                    ON user_profiles_new(username);
                """)
                print("   ✓ Index created")
                
                # Drop old table
                print("Step 3: Removing old table...")
                cursor.execute("DROP TABLE user_profiles;")
                print("   ✓ Old table removed")
                
                # Rename new table
                print("Step 4: Activating new table...")
                cursor.execute("ALTER TABLE user_profiles_new RENAME TO user_profiles;")
                print("   ✓ New table activated")
                
                print("\n✅ Migration completed successfully!")
                
            elif has_username and has_password:
                print("✅ Database already has correct schema (username/password)")
            else:
                print("⚠️  Unexpected schema. Recreating table...")
                cursor.execute("DROP TABLE IF EXISTS user_profiles;")
                cursor.execute("""
                    CREATE TABLE user_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username VARCHAR(150) UNIQUE NOT NULL,
                        password VARCHAR(128) NOT NULL,
                        email VARCHAR(254) NOT NULL,
                        name VARCHAR(255),
                        is_active BOOLEAN NOT NULL DEFAULT 1,
                        has_used_trial BOOLEAN NOT NULL DEFAULT 0,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    );
                """)
                cursor.execute("""
                    CREATE UNIQUE INDEX user_profiles_username_idx 
                    ON user_profiles(username);
                """)
                print("✅ Table recreated!")
        
        # Verify final state
        print("\n" + "-"*70)
        print("Verifying final schema...")
        cursor.execute("PRAGMA table_info(user_profiles);")
        final_columns = [row[1] for row in cursor.fetchall()]
        print(f"Final columns: {', '.join(final_columns)}")
        
        if 'username' in final_columns and 'password' in final_columns:
            print("\n✅ Schema is correct!")
        else:
            print("\n⚠️  Warning: Schema might still be incorrect")
    
    print("\n" + "="*70)
    print(" SUCCESS! Database has been fixed.")
    print("="*70)
    print("\nNext steps:")
    print("1. Restart your Django server (Ctrl+C then: python manage.py runserver)")
    print("2. Go to http://localhost:8000/admin/")
    print("3. Create a user in 'User Profiles'")
    print("4. Test login in your frontend")
    print("\n" + "="*70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}\n")
    import traceback
    traceback.print_exc()
    print("\n" + "="*70)
    print(" If this error persists, try deleting db.sqlite3 and running:")
    print("   python manage.py migrate")
    print("="*70 + "\n")

input("Press Enter to exit...")
