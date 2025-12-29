import os, sys, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("Fixing database schema...")

with connection.cursor() as cursor:
    # Check current schema
    cursor.execute("PRAGMA table_info(user_profiles);")
    columns = [row[1] for row in cursor.fetchall()]
    
    print(f"Current columns: {columns}")
    
    has_google_id = 'google_id' in columns
    has_username = 'username' in columns
    
    if has_google_id and not has_username:
        print("Converting from google_id to username/password...")
        
        # Create new table
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
        
        cursor.execute("""
            CREATE UNIQUE INDEX user_profiles_username_idx 
            ON user_profiles_new(username);
        """)
        
        cursor.execute("DROP TABLE user_profiles;")
        cursor.execute("ALTER TABLE user_profiles_new RENAME TO user_profiles;")
        
        print("✅ Fixed!")
    elif has_username:
        print("✅ Database already has correct schema")
    else:
        print("Creating table from scratch...")
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
        print("✅ Created!")

# Verify
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles);")
    final = [r[1] for r in cursor.fetchall()]
    print(f"Final columns: {final}")

print("\n✅ Done! Restart your Django server now.")
