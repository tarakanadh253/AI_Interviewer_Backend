#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create django_session table - Run this to fix the error immediately
"""
import os
import sys
import sqlite3

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from django.conf import settings
from django.db import connection

# Get database path
db_path = str(settings.DATABASES['default']['NAME'])

print(f"\n{'='*70}")
print("CREATING DJANGO_SESSION TABLE")
print(f"{'='*70}\n")
print(f"Database: {db_path}")
print(f"Exists: {os.path.exists(db_path)}\n")

# Method 1: Try using Django migrations
print("Method 1: Running Django migrations...")
try:
    from django.core.management import call_command
    call_command('migrate', 'sessions', verbosity=0, interactive=False)
    print("✓ Migrations completed")
except Exception as e:
    print(f"⚠ Migration warning: {e}")

# Method 2: Direct SQL creation if table doesn't exist
print("\nMethod 2: Checking if table exists and creating if needed...")
try:
    # Use direct SQLite connection for reliability
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
    exists = cursor.fetchone()
    
    if exists:
        print("✓ django_session table already exists!")
    else:
        print("✗ Table not found. Creating it...")
        
        # Create the table with the exact schema Django expects
        cursor.execute("""
            CREATE TABLE django_session (
                session_key VARCHAR(40) NOT NULL PRIMARY KEY,
                session_data TEXT NOT NULL,
                expire_date DATETIME NOT NULL
            );
        """)
        
        # Create index
        cursor.execute("""
            CREATE INDEX django_session_expire_date_a5f62663 
            ON django_session(expire_date);
        """)
        
        conn.commit()
        print("✓ django_session table created successfully!")
        
        # Verify
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
        verify = cursor.fetchone()
        if verify:
            print("✓ Verification: Table confirmed in database")
        else:
            print("✗ Verification failed")
    
    # Show table structure
    cursor.execute("PRAGMA table_info(django_session);")
    columns = cursor.fetchall()
    print(f"\nTable structure ({len(columns)} columns):")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    conn.close()
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Method 3: Verify via Django ORM
print("\nMethod 3: Verifying via Django connection...")
try:
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
        result = cursor.fetchone()
        if result:
            print("✓ VERIFIED: Django can see the django_session table!")
        else:
            print("✗ Django cannot see the table")
except Exception as e:
    print(f"✗ Django verification error: {e}")

print(f"\n{'='*70}")
print("FIX COMPLETE!")
print(f"{'='*70}")
print("\nNext steps:")
print("1. Restart your Django server (stop with Ctrl+C, then run: python manage.py runserver)")
print("2. Go to http://localhost:8000/admin/")
print("3. The error should be fixed!")
print(f"{'='*70}\n")
