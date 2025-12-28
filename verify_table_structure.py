#!/usr/bin/env python
"""Verify interview_sessions table structure matches model"""
import sqlite3

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

# Check if table exists
c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interview_sessions'")
if not c.fetchone():
    print("ERROR: interview_sessions table does not exist!")
    conn.close()
    exit(1)

# Get columns
c.execute("PRAGMA table_info(interview_sessions)")
columns = c.fetchall()

print("Current table columns:")
for col in columns:
    print(f"  {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'} - Default: {col[4]}")

# Check for ManyToMany join table
c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%interview_sessions%topics%'")
m2m_tables = c.fetchall()
print(f"\nManyToMany join tables: {[t[0] for t in m2m_tables]}")

conn.close()
