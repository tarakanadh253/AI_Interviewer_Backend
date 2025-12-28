import os, sys, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

output_lines = []
def log(msg):
    print(msg)
    output_lines.append(msg)

log("="*70)
log("FIXING DATABASE SCHEMA")
log("="*70)

try:
    with connection.cursor() as cursor:
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profiles';")
        if not cursor.fetchone():
            log("\nTable doesn't exist. Creating...")
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
            cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles(username);")
            log("✅ Table created!")
        else:
            # Check schema
            cursor.execute("PRAGMA table_info(user_profiles);")
            columns = [row[1] for row in cursor.fetchall()]
            log(f"\nCurrent columns: {', '.join(columns)}")
            
            has_google_id = 'google_id' in columns
            has_username = 'username' in columns
            
            if has_google_id and not has_username:
                log("\n⚠️  Found old schema. Converting...")
                
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
                cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles_new(username);")
                cursor.execute("DROP TABLE user_profiles;")
                cursor.execute("ALTER TABLE user_profiles_new RENAME TO user_profiles;")
                log("✅ Conversion complete!")
            elif has_username:
                log("✅ Schema is already correct!")
            else:
                log("⚠️  Unexpected schema. Recreating...")
                cursor.execute("DROP TABLE user_profiles;")
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
                cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles(username);")
                log("✅ Recreated!")
        
        # Verify
        cursor.execute("PRAGMA table_info(user_profiles);")
        final = [r[1] for r in cursor.fetchall()]
        log(f"\nFinal columns: {', '.join(final)}")
        
        if 'username' in final and 'password' in final:
            log("\n✅ SUCCESS! Database is fixed.")
            log("\nNext: Restart your Django server with: python manage.py runserver")
        else:
            log("\n⚠️  WARNING: Schema might still be incorrect")
    
    # Write to file
    with open('fix_result.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    log("\nResults saved to fix_result.txt")
    
except Exception as e:
    log(f"\n❌ ERROR: {e}")
    import traceback
    log(traceback.format_exc())
    with open('fix_result.txt', 'w') as f:
        f.write('\n'.join(output_lines))
