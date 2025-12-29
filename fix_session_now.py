#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IMMEDIATE FIX for django_session table - Run this now!
"""
import os
import sys

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from django.core.management import call_command
from django.db import connection
from django.conf import settings
import sqlite3

print("\n" + "="*70)
print(" FIXING DJANGO_SESSION TABLE - RUNNING NOW")
print("="*70 + "\n")

db_path = str(settings.DATABASES['default']['NAME'])
print(f"Database: {db_path}")
print(f"Exists: {os.path.exists(db_path)}\n")

# Step 1: Run all migrations
print("="*70)
print("STEP 1: Running Django migrations...")
print("="*70)
try:
    call_command('migrate', verbosity=1, interactive=False, run_syncdb=True)
    print("✓ Migrations completed\n")
except Exception as e:
    print(f"✗ Error: {e}\n")
    import traceback
    traceback.print_exc()

# Step 2: Specifically migrate sessions app
print("="*70)
print("STEP 2: Migrating sessions app specifically...")
print("="*70)
try:
    call_command('migrate', 'sessions', verbosity=1, interactive=False)
    print("✓ Sessions migration completed\n")
except Exception as e:
    print(f"⚠ Warning: {e}\n")

# Step 3: Verify table exists
print("="*70)
print("STEP 3: Verifying django_session table...")
print("="*70)

if not os.path.exists(db_path):
    print("✗ Database file does not exist!")
    print("Creating database...")
    # Force create by running syncdb
    try:
        call_command('migrate', '--run-syncdb', verbosity=1, interactive=False)
    except:
        pass

# Check using direct SQLite connection
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
    result = cursor.fetchone()
    
    if result:
        print("✓ django_session table EXISTS!")
        
        # Show table structure
        cursor.execute("PRAGMA table_info(django_session);")
        columns = cursor.fetchall()
        print(f"\nTable structure ({len(columns)} columns):")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    else:
        print("✗ django_session table NOT FOUND!")
        print("\nAttempting manual creation...")
        
        # Try to create the table manually based on Django's session model
        try:
            cursor.execute("""
                CREATE TABLE django_session (
                    session_key VARCHAR(40) PRIMARY KEY,
                    session_data TEXT NOT NULL,
                    expire_date DATETIME NOT NULL
                );
            """)
            cursor.execute("CREATE INDEX django_session_expire_date ON django_session(expire_date);")
            conn.commit()
            print("✓ Table created manually!")
        except Exception as e:
            print(f"✗ Failed to create manually: {e}")
    
    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    all_tables = [row[0] for row in cursor.fetchall()]
    print(f"\nAll tables in database ({len(all_tables)} total):")
    for table in all_tables:
        marker = "✓" if table == "django_session" else " "
        print(f"  {marker} {table}")
    
    conn.close()
    
except Exception as e:
    print(f"✗ Error checking database: {e}")
    import traceback
    traceback.print_exc()

# Final verification using Django
print("\n" + "="*70)
print("STEP 4: Final verification using Django ORM...")
print("="*70)
try:
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
        result = cursor.fetchone()
        if result:
            print("✓ VERIFIED: django_session table is accessible via Django!")
        else:
            print("✗ VERIFICATION FAILED: Table still not found via Django")
except Exception as e:
    print(f"✗ Verification error: {e}")

print("\n" + "="*70)
if os.path.exists(db_path):
    print(" SUCCESS! Database file exists.")
    print(" If django_session table exists, restart your server.")
    print("="*70)
    print("\nNext steps:")
    print("1. Stop your Django server (Ctrl+C)")
    print("2. Run: python manage.py runserver")
    print("3. Go to: http://localhost:8000/admin/")
else:
    print(" WARNING: Database file was not created.")
    print("="*70)
    print("\nTry running: python manage.py migrate")
print("="*70 + "\n")
