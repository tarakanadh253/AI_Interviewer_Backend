#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fix questions table schema - add missing source_type column
"""
import os
import sys
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

import django
django.setup()

from django.conf import settings
from django.db import connection

print("\n" + "="*70)
print("FIXING QUESTIONS TABLE SCHEMA")
print("="*70 + "\n")

db_path = str(settings.DATABASES['default']['NAME'])
print(f"Database: {db_path}")

# Check current schema
print("\n1. Checking current questions table schema...")
try:
    with connection.cursor() as cursor:
        cursor.execute("PRAGMA table_info(questions);")
        columns = {row[1]: row for row in cursor.fetchall()}
        column_names = list(columns.keys())
        print(f"   Current columns: {', '.join(column_names)}")
        
        # Check for missing columns
        required_columns = {
            'source_type': "VARCHAR(10) DEFAULT 'MANUAL'",
            'reference_links': "TEXT",
        }
        
        missing = []
        for col_name, col_def in required_columns.items():
            if col_name not in column_names:
                missing.append((col_name, col_def))
        
        if missing:
            print(f"\n2. Adding {len(missing)} missing columns...")
            for col_name, col_def in missing:
                try:
                    sql = f"ALTER TABLE questions ADD COLUMN {col_name} {col_def};"
                    print(f"   Adding {col_name}...")
                    cursor.execute(sql)
                    connection.commit()
                    print(f"   ✓ Added {col_name}")
                except Exception as e:
                    print(f"   ✗ Failed to add {col_name}: {e}")
        else:
            print("\n2. ✓ All required columns exist")
        
        # Verify
        print("\n3. Verifying schema...")
        cursor.execute("PRAGMA table_info(questions);")
        final_columns = [row[1] for row in cursor.fetchall()]
        print(f"   Final columns: {', '.join(final_columns)}")
        
        # Update existing rows to have default source_type
        if 'source_type' in final_columns:
            cursor.execute("UPDATE questions SET source_type = 'MANUAL' WHERE source_type IS NULL;")
            connection.commit()
            updated = cursor.rowcount
            if updated > 0:
                print(f"   ✓ Updated {updated} questions with default source_type")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("SCHEMA FIX COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Restart your Django server")
print("2. The Question Bank tab should work now")
print("="*70 + "\n")
