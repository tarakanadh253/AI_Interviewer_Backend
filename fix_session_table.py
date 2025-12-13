#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix django_session table issue by running migrations.
Run this script: python fix_session_table.py
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
    print(f"❌ Error setting up Django: {e}")
    print("\nMake sure you're in the backend directory and Django is installed!")
    input("Press Enter to exit...")
    sys.exit(1)

from django.core.management import call_command
from django.db import connection
from django.conf import settings

print("\n" + "="*70)
print(" FIXING DJANGO_SESSION TABLE ISSUE")
print("="*70 + "\n")

db_path = settings.DATABASES['default']['NAME']
print(f"Database path: {db_path}")
print(f"Database exists: {os.path.exists(db_path)}\n")

# Check if django_session table exists
with connection.cursor() as cursor:
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
        result = cursor.fetchone()
        if result:
            print("✅ django_session table already exists!")
            print("\nThe table is present. If you're still getting errors,")
            print("try restarting your Django server.")
            input("\nPress Enter to exit...")
            sys.exit(0)
        else:
            print("❌ django_session table does not exist.")
            print("Running migrations to create it...\n")
    except Exception as e:
        print(f"⚠️  Error checking table: {e}")
        print("Running migrations anyway...\n")

# Run migrations
print("="*70)
print("Running Django migrations...")
print("="*70)
try:
    call_command('migrate', verbosity=1, interactive=False)
    print("\n✅ Migrations completed!")
except Exception as e:
    print(f"\n❌ Migration error: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
    sys.exit(1)

# Verify django_session table was created
print("\n" + "="*70)
print("Verifying django_session table...")
print("="*70)
with connection.cursor() as cursor:
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
        result = cursor.fetchone()
        if result:
            print("✅ django_session table created successfully!")
        else:
            print("❌ django_session table still not found after migrations!")
            print("\nTrying to create it manually...")
            # Try to run sessions migration specifically
            try:
                call_command('migrate', 'sessions', verbosity=1, interactive=False)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
                result = cursor.fetchone()
                if result:
                    print("✅ django_session table created manually!")
                else:
                    print("❌ Still unable to create table.")
            except Exception as e2:
                print(f"❌ Error creating table manually: {e2}")
    except Exception as e:
        print(f"❌ Error verifying table: {e}")

# List all Django tables
print("\n" + "="*70)
print("All Django tables in database:")
print("="*70)
with connection.cursor() as cursor:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = cursor.fetchall()
    django_tables = [t[0] for t in tables if t[0].startswith('django_') or t[0] in ['user_profiles', 'topics', 'questions', 'interview_sessions', 'answers']]
    for table in django_tables:
        print(f"  ✓ {table}")

if 'django_session' in [t[0] for t in tables]:
    print("\n" + "="*70)
    print(" SUCCESS! django_session table is now available.")
    print("="*70)
    print("\nNext steps:")
    print("1. Restart your Django server (if it's running, press Ctrl+C)")
    print("2. Run: python manage.py runserver")
    print("3. Go to http://localhost:8000/admin/")
    print("="*70 + "\n")
else:
    print("\n" + "="*70)
    print(" WARNING: django_session table not found.")
    print("="*70)
    print("\nTry these steps:")
    print("1. Delete db.sqlite3 if it exists")
    print("2. Run: python manage.py migrate")
    print("3. Restart your Django server")
    print("="*70 + "\n")

input("Press Enter to exit...")
