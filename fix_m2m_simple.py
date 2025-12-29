#!/usr/bin/env python
"""Create the M2M table - simple version"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

django.setup()

from django.db import connection
from interview.models import InterviewSession

# Get table name
m2m_field = InterviewSession._meta.get_field('topics')
m2m_table = m2m_field.remote_field.through._meta.db_table

print(f"Table name: {m2m_table}")

# Check if exists
with connection.cursor() as cursor:
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{m2m_table}'")
    exists = cursor.fetchone() is not None
    
    if exists:
        print(f"Table {m2m_table} already exists")
    else:
        print(f"Creating table {m2m_table}...")
        
        # Get column names
        through_model = m2m_field.remote_field.through
        session_fk = through_model._meta.get_field('interviewsession')
        topic_fk = through_model._meta.get_field('topic')
        session_col = session_fk.column
        topic_col = topic_fk.column
        
        print(f"Columns: {session_col}, {topic_col}")
        
        # Create table
        cursor.execute(f"""
            CREATE TABLE {m2m_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {session_col} INTEGER NOT NULL,
                {topic_col} INTEGER NOT NULL,
                FOREIGN KEY ({session_col}) REFERENCES interview_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY ({topic_col}) REFERENCES topics(id) ON DELETE CASCADE,
                UNIQUE({session_col}, {topic_col})
            )
        """)
        connection.commit()
        print(f"Table {m2m_table} created successfully!")
