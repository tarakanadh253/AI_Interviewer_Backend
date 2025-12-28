#!/usr/bin/env python
"""Fix all missing database tables"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.core.management import call_command
from django.db import connection
import sqlite3

print("=" * 70)
print("FIXING ALL MISSING DATABASE TABLES")
print("=" * 70)
print()

# Check which tables exist
conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
existing_tables = {row[0] for row in c.fetchall()}
conn.close()

print("Existing tables:", ', '.join(sorted(existing_tables)))
print()

# Required tables
required_tables = {
    'django_session',
    'answers',
    'questions',
    'topics',
    'user_profiles',
    'interview_sessions',
}

missing_tables = required_tables - existing_tables

if missing_tables:
    print(f"Missing tables: {', '.join(missing_tables)}")
    print()
    print("Running migrations to create missing tables...")
    print()
    
    # Run migrations
    try:
        call_command('migrate', verbosity=2, interactive=False)
        print()
        print("✓ Migrations completed!")
    except Exception as e:
        print(f"✗ Migration error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✓ All required tables exist!")

# Verify
print()
print("Verifying tables...")
conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
final_tables = {row[0] for row in c.fetchall()}
conn.close()

for table in required_tables:
    status = "✓" if table in final_tables else "✗"
    print(f"{status} {table}")

print()
print("=" * 70)
print("DONE! Restart your Django server.")
print("=" * 70)
