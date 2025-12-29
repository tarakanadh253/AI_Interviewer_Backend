#!/usr/bin/env python
"""Fix django_session table by running migrations."""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.core.management import call_command
from django.db import connection
from django.conf import settings

def main():
    print("=" * 60)
    print("Fixing Django Session Table Issue")
    print("=" * 60)
    
    db_path = settings.DATABASES['default']['NAME']
    print(f"\nDatabase path: {db_path}")
    print(f"Database exists: {os.path.exists(db_path)}")
    
    # Check if django_session table exists
    with connection.cursor() as cursor:
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
            result = cursor.fetchone()
            if result:
                print("\n✓ django_session table already exists!")
                return
            else:
                print("\n✗ django_session table does not exist. Running migrations...")
        except Exception as e:
            print(f"\n✗ Error checking table: {e}")
            print("Running migrations...")
    
    # Run migrations
    print("\nRunning migrations...")
    try:
        call_command('migrate', verbosity=2, interactive=False)
        print("\n✓ Migrations completed successfully!")
    except Exception as e:
        print(f"\n✗ Migration error: {e}")
        return
    
    # Verify django_session table was created
    with connection.cursor() as cursor:
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='django_session';")
            result = cursor.fetchone()
            if result:
                print("\n✓ django_session table created successfully!")
            else:
                print("\n✗ django_session table still not found after migrations!")
        except Exception as e:
            print(f"\n✗ Error verifying table: {e}")
    
    # List all tables
    print("\n" + "=" * 60)
    print("All tables in database:")
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            print(f"  - {table[0]}")
    
    print("\n" + "=" * 60)
    print("Done! You can now restart your Django server.")
    print("=" * 60)

if __name__ == '__main__':
    main()
