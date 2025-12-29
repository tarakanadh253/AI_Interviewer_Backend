#!/usr/bin/env python
"""Verify and fix questions table - run this to fix Question Bank"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
import sqlite3

print("\n" + "="*70)
print("VERIFYING AND FIXING QUESTIONS TABLE")
print("="*70 + "\n")

db_path = str(django.conf.settings.DATABASES['default']['NAME'])
print(f"Database: {db_path}\n")

# Use direct SQLite connection for reliability
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check current columns
    cursor.execute("PRAGMA table_info(questions);")
    existing_cols = {row[1]: row for row in cursor.fetchall()}
    print(f"Current columns: {', '.join(sorted(existing_cols.keys()))}\n")
    
    # Add missing columns
    fixes_applied = False
    
    if 'source_type' not in existing_cols:
        print("Adding source_type column...")
        cursor.execute("ALTER TABLE questions ADD COLUMN source_type VARCHAR(10) DEFAULT 'MANUAL';")
        cursor.execute("UPDATE questions SET source_type = 'MANUAL' WHERE source_type IS NULL;")
        conn.commit()
        print("✓ source_type added\n")
        fixes_applied = True
    else:
        print("✓ source_type exists\n")
    
    if 'reference_links' not in existing_cols:
        print("Adding reference_links column...")
        cursor.execute("ALTER TABLE questions ADD COLUMN reference_links TEXT;")
        conn.commit()
        print("✓ reference_links added\n")
        fixes_applied = True
    else:
        print("✓ reference_links exists\n")
    
    # Verify final state
    cursor.execute("PRAGMA table_info(questions);")
    final_cols = {row[1] for row in cursor.fetchall()}
    print(f"Final columns: {', '.join(sorted(final_cols))}\n")
    
    # Test query
    print("Testing query...")
    cursor.execute("SELECT id, topic_id, source_type FROM questions LIMIT 1;")
    result = cursor.fetchone()
    if result:
        print(f"✓ Query successful - Sample: ID={result[0]}, topic_id={result[1]}, source_type={result[2] if len(result) > 2 else 'N/A'}\n")
    else:
        print("⚠ No questions in database\n")
    
    if fixes_applied:
        print("="*70)
        print("✓ FIXES APPLIED - Restart your Django server!")
        print("="*70)
    else:
        print("="*70)
        print("✓ All columns exist - Issue may be elsewhere")
        print("="*70)
        
except Exception as e:
    print(f"✗ Error: {e}\n")
    import traceback
    traceback.print_exc()
finally:
    conn.close()

print()
