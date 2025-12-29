#!/usr/bin/env python
"""Fix user_profiles table schema directly"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("FIXING user_profiles TABLE SCHEMA")
print("=" * 70)

with connection.cursor() as cursor:
    # Check current schema
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    
    print(f"\nCurrent columns: {', '.join(col_names)}")
    
    required_cols = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
        'username': 'VARCHAR(150) UNIQUE NOT NULL',
        'password': 'VARCHAR(128) NOT NULL',
        'email': 'VARCHAR(254) NOT NULL',
        'name': 'VARCHAR(255)',
        'is_active': 'BOOLEAN NOT NULL DEFAULT 1',
        'access_type': "VARCHAR(10) DEFAULT 'TRIAL'",
        'has_used_trial': 'BOOLEAN NOT NULL DEFAULT 0',
        'created_at': 'DATETIME NOT NULL',
        'updated_at': 'DATETIME NOT NULL'
    }
    
    missing = [c for c in required_cols.keys() if c not in col_names]
    
    if missing or 'username' not in col_names or 'password' not in col_names:
        print(f"\n✗ Missing or incorrect columns. Recreating table...")
        print(f"Missing: {missing}")
        
        # Disable foreign keys
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        # Drop dependent tables first
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND (name LIKE '%session%' OR name LIKE '%answer%')
        """)
        dependent = [r[0] for r in cursor.fetchall()]
        for table in dependent:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"  Dropped {table}")
            except:
                pass
        
        # Drop old table
        cursor.execute("DROP TABLE IF EXISTS user_profiles")
        
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
        
        # Create index
        cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles(username)")
        
        # Re-enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        print("✓ Table recreated with correct schema")
        
        # Verify
        cursor.execute("PRAGMA table_info(user_profiles)")
        new_cols = [c[1] for c in cursor.fetchall()]
        print(f"New columns: {', '.join(new_cols)}")
        
        if 'username' in new_cols and 'password' in new_cols:
            print("\n✓ Table fixed successfully!")
        else:
            print("\n✗ ERROR: Table recreation failed!")
            sys.exit(1)
    else:
        print("\n✓ Table schema is correct!")

print("\n" + "=" * 70)
print("SCHEMA FIX COMPLETE")
print("=" * 70)
