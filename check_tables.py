#!/usr/bin/env python
"""Check if all required tables exist"""
import sqlite3

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

required = ['answers', 'interview_sessions', 'questions', 'topics', 'user_profiles', 'django_session']
missing = []

for t in required:
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,))
    if not c.fetchone():
        missing.append(t)

if missing:
    print('Missing tables:', ', '.join(missing))
    conn.close()
    exit(1)
else:
    print('All required tables exist!')
    conn.close()
    exit(0)
