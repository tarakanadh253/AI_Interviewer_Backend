#!/usr/bin/env python
"""Add missing source_type and reference_links columns to questions table"""
import os, sys
import sqlite3

# Use direct SQLite connection - more reliable
db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')

if not os.path.exists(db_path):
    print(f"ERROR: Database file not found at {db_path}")
    sys.exit(1)

print("\n" + "="*70)
print("ADDING MISSING COLUMNS TO QUESTIONS TABLE")
print("="*70 + "\n")
print(f"Database: {db_path}\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Check current columns
    cursor.execute("PRAGMA table_info(questions);")
    existing = {row[1] for row in cursor.fetchall()}
    print(f"Existing columns: {', '.join(sorted(existing))}\n")
    
    fixes_applied = False
    
    # Add source_type if missing
    if 'source_type' not in existing:
        print("Adding source_type column...")
        cursor.execute("ALTER TABLE questions ADD COLUMN source_type VARCHAR(10) DEFAULT 'MANUAL';")
        cursor.execute("UPDATE questions SET source_type = 'MANUAL' WHERE source_type IS NULL;")
        conn.commit()
        print("✓ source_type column added\n")
        fixes_applied = True
    else:
        print("✓ source_type column already exists\n")
    
    # Add reference_links if missing
    if 'reference_links' not in existing:
        print("Adding reference_links column...")
        cursor.execute("ALTER TABLE questions ADD COLUMN reference_links TEXT;")
        conn.commit()
        print("✓ reference_links column added\n")
        fixes_applied = True
    else:
        print("✓ reference_links column already exists\n")
    
    # Verify
    print("Verifying...")
    cursor.execute("PRAGMA table_info(questions);")
    final = {row[1] for row in cursor.fetchall()}
    print(f"Final columns: {', '.join(sorted(final))}\n")
    
    if 'source_type' in final and 'reference_links' in final:
        print("="*70)
        print("✓ SUCCESS! All columns are present.")
        if fixes_applied:
            print("✓ Restart your Django server now!")
        print("="*70)
    else:
        print("="*70)
        print("✗ WARNING: Some columns may still be missing.")
        print("="*70)
        
except Exception as e:
    print(f"\n✗ ERROR: {e}\n")
    import traceback
    traceback.print_exc()
finally:
    conn.close()

print()
