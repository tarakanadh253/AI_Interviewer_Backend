#!/usr/bin/env python
"""Simple fix - drop all tables and recreate"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("FIXING DATABASE - DROPPING ALL TABLES")
print("=" * 70)

with connection.cursor() as cursor:
    # Disable foreign keys
    cursor.execute("PRAGMA foreign_keys = OFF")
    print("✓ Foreign key checks disabled")
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = [row[0] for row in cursor.fetchall()]
    print(f"\nFound tables: {', '.join(all_tables)}")
    
    # Drop all tables
    print("\nDropping all tables...")
    for table in all_tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"✓ Dropped {table}")
        except Exception as e:
            print(f"  Error dropping {table}: {e}")
    
    # Drop all indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
    indexes = [row[0] for row in cursor.fetchall()]
    for idx in indexes:
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {idx}")
        except:
            pass
    
    # Re-enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    print("\n✓ All tables dropped")
    print("\nDatabase is now empty and ready for migrations!")

print("\n" + "=" * 70)
print("DATABASE CLEARED - READY FOR MIGRATIONS")
print("=" * 70)
