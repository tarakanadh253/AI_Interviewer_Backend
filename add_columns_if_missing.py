#!/usr/bin/env python
"""Add missing columns to questions table if they don't exist"""
import sqlite3

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

# Check existing columns
c.execute('PRAGMA table_info(questions)')
cols = [row[1] for row in c.fetchall()]
print('Current columns:', ', '.join(cols))

# Add source_type if missing
if 'source_type' not in cols:
    print('Adding source_type column...')
    c.execute("ALTER TABLE questions ADD COLUMN source_type VARCHAR(10) DEFAULT 'MANUAL'")
    print('✓ source_type added')
else:
    print('✓ source_type already exists')

# Add reference_links if missing
if 'reference_links' not in cols:
    print('Adding reference_links column...')
    c.execute("ALTER TABLE questions ADD COLUMN reference_links TEXT")
    print('✓ reference_links added')
else:
    print('✓ reference_links already exists')

conn.commit()

# Verify
c.execute('PRAGMA table_info(questions)')
final_cols = [row[1] for row in c.fetchall()]
print('\nFinal columns:', ', '.join(final_cols))
print('source_type exists:', 'source_type' in final_cols)
print('reference_links exists:', 'reference_links' in final_cols)

conn.close()
print('\nDone!')
