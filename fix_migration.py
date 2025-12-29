#!/usr/bin/env python
"""
Script to fix the database migration issue
Run this from the backend directory: python fix_migration.py
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.core.management import call_command
from django.db import connection

def check_table_schema():
    """Check what columns exist in user_profiles table"""
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(user_profiles);")
        columns = cursor.fetchall()
        print("\nCurrent database columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        return [col[1] for col in columns]

def main():
    print("=" * 60)
    print("Fixing UserProfile Migration")
    print("=" * 60)
    
    # Check current schema
    print("\n1. Checking current database schema...")
    columns = check_table_schema()
    
    has_google_id = 'google_id' in columns
    has_username = 'username' in columns
    
    print(f"\n   Has google_id: {has_google_id}")
    print(f"   Has username: {has_username}")
    
    if has_google_id and not has_username:
        print("\n2. Database needs migration. Applying migrations...")
        try:
            call_command('migrate', 'interview', verbosity=2)
            print("\n✅ Migration applied successfully!")
        except Exception as e:
            print(f"\n❌ Error applying migration: {e}")
            print("\nTrying to fix manually...")
            # Try to manually fix the schema
            try:
                with connection.cursor() as cursor:
                    # Delete existing users
                    cursor.execute("DELETE FROM user_profiles;")
                    # Drop google_id column (SQLite doesn't support DROP COLUMN directly)
                    # We'll need to recreate the table
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
                    cursor.execute("DROP TABLE user_profiles;")
                    cursor.execute("ALTER TABLE user_profiles_new RENAME TO user_profiles;")
                    cursor.execute("CREATE INDEX user_profiles_username_idx ON user_profiles(username);")
                print("✅ Database schema fixed manually!")
            except Exception as e2:
                print(f"❌ Manual fix failed: {e2}")
                print("\nPlease run: python manage.py migrate interview")
                return False
    elif has_username:
        print("\n✅ Database schema is already correct!")
    else:
        print("\n⚠️  Unexpected schema state. Please check manually.")
    
    # Verify final state
    print("\n3. Verifying final schema...")
    columns = check_table_schema()
    print(f"\n   Final columns: {', '.join(columns)}")
    
    print("\n" + "=" * 60)
    print("Migration fix complete!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
