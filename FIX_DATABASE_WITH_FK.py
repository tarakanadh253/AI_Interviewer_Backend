#!/usr/bin/env python
"""Fix database schema - handle foreign key constraints"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("FIXING DATABASE SCHEMA (WITH FOREIGN KEY HANDLING)")
print("=" * 70)

with connection.cursor() as cursor:
    # Check current schema
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    
    print(f"\nCurrent columns: {', '.join(col_names)}")
    
    has_username = 'username' in col_names
    has_google_id = 'google_id' in col_names
    
    print(f"\nHas username: {has_username}")
    print(f"Has google_id: {has_google_id}")
    
    if not has_username and has_google_id:
        print("\n✗ CRITICAL: Table has old schema (google_id instead of username)")
        print("\nFixing schema...")
        
        # Disable foreign key checks temporarily
        cursor.execute("PRAGMA foreign_keys = OFF")
        print("✓ Foreign key checks disabled")
        
        # Check for dependent tables that reference user_profiles
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND (name LIKE '%session%' OR name LIKE '%answer%' OR name LIKE '%interview%')
        """)
        dependent_tables = [row[0] for row in cursor.fetchall()]
        print(f"Found dependent tables: {dependent_tables}")
        
        # Drop dependent tables first (they will be recreated by migrations)
        for table in dependent_tables:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"✓ Dropped {table}")
            except Exception as e:
                print(f"  Note: Could not drop {table} - {e}")
        
        # Drop indexes first
        try:
            cursor.execute("DROP INDEX IF EXISTS user_profiles_username_idx")
            print("✓ Dropped index")
        except:
            pass
        
        # Now drop user_profiles
        try:
            cursor.execute("DROP TABLE IF EXISTS user_profiles")
            print("✓ Dropped user_profiles")
        except Exception as e:
            print(f"  Error dropping table: {e}")
            # Try to delete all rows instead
            try:
                cursor.execute("DELETE FROM user_profiles")
                print("  Deleted all rows instead")
            except:
                pass
        
        # Re-enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create new table with correct schema
        print("\nCreating new user_profiles table...")
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
        print("✓ Table created")
        
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
    elif has_username:
        print("\n✓ username column exists - table is correct!")
        
        # Just ensure access_type exists
        if 'access_type' not in col_names:
            print("Adding access_type column...")
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("✓ access_type added!")

print("\n" + "=" * 70)
print("✓ DATABASE SCHEMA FIXED!")
print("=" * 70)
print("\nNext steps:")
print("  1. Run migrations to recreate dependent tables:")
print("     python manage.py migrate")
print("  2. Seed data if needed:")
print("     python manage.py seed_data")
print("  3. Start server:")
print("     python manage.py runserver")
