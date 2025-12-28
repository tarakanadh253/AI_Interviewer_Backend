#!/usr/bin/env python
"""
Direct script to apply the migration and fix the database
Run: python apply_migration.py
"""
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

def apply_migration():
    print("=" * 60)
    print("Applying UserProfile Migration")
    print("=" * 60)
    
    try:
        with connection.cursor() as cursor:
            # Check current schema
            cursor.execute("PRAGMA table_info(user_profiles);")
            columns = [row[1] for row in cursor.fetchall()]
            
            print(f"\nCurrent columns: {', '.join(columns)}")
            
            has_google_id = 'google_id' in columns
            has_username = 'username' in columns
            
            if has_google_id and not has_username:
                print("\n⚠️  Database has old schema (google_id). Fixing...")
                
                # Create new table
                print("Creating new table structure...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles_new (
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
                
                # Create index
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS user_profiles_username_idx 
                    ON user_profiles_new(username);
                """)
                
                # Drop old table
                print("Removing old table...")
                cursor.execute("DROP TABLE IF EXISTS user_profiles;")
                
                # Rename new table
                print("Renaming new table...")
                cursor.execute("ALTER TABLE user_profiles_new RENAME TO user_profiles;")
                
                print("\n✅ Migration applied successfully!")
                print("\nNew columns: id, username, password, email, name, is_active, has_used_trial, created_at, updated_at")
                
            elif has_username:
                print("\n✅ Database already has correct schema (username/password)")
            else:
                print("\n⚠️  Unexpected schema. Creating table from scratch...")
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
                print("✅ Table created successfully!")
        
        # Verify
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA table_info(user_profiles);")
            final_columns = [row[1] for row in cursor.fetchall()]
            print(f"\nFinal schema: {', '.join(final_columns)}")
        
        print("\n" + "=" * 60)
        print("✅ Migration complete! Restart your Django server.")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = apply_migration()
    sys.exit(0 if success else 1)
