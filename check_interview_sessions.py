#!/usr/bin/env python
"""Check if interview_sessions table exists"""
import sqlite3

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

# Check if table exists
c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interview_sessions'")
result = c.fetchone()

if result:
    print('interview_sessions table: EXISTS')
    # Get columns
    c.execute('PRAGMA table_info(interview_sessions)')
    cols = [r[1] for r in c.fetchall()]
    print('Columns:', ', '.join(cols))
else:
    print('interview_sessions table: MISSING')

conn.close()
