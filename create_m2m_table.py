#!/usr/bin/env python
"""Create the ManyToMany join table for interview_sessions and topics"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    django.setup()
    from django.db import connection
    from interview.models import InterviewSession
    import sys
    
    # Get Django's ManyToMany table name
    m2m_field = InterviewSession._meta.get_field('topics')
    m2m_table = m2m_field.remote_field.through._meta.db_table
    
    # Force output
    sys.stdout.write(f"Django expects M2M table: {m2m_table}\n")
    sys.stdout.flush()
    
    # Check if it exists
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{m2m_table}'")
        exists = cursor.fetchone() is not None
        
        if exists:
            sys.stdout.write(f"✓ Table '{m2m_table}' already exists\n")
            sys.stdout.flush()
            # Show structure
            cursor.execute(f"PRAGMA table_info({m2m_table})")
            cols = [row[1] for row in cursor.fetchall()]
            sys.stdout.write(f"  Columns: {', '.join(cols)}\n")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"✗ Table '{m2m_table}' does NOT exist\n")
            sys.stdout.write(f"Creating it now...\n")
            sys.stdout.flush()
            
            # Get the actual column names Django uses
            through_model = m2m_field.remote_field.through
            session_fk = through_model._meta.get_field('interviewsession')
            topic_fk = through_model._meta.get_field('topic')
            
            session_col = session_fk.column
            topic_col = topic_fk.column
            
            sys.stdout.write(f"  Session FK column: {session_col}\n")
            sys.stdout.write(f"  Topic FK column: {topic_col}\n")
            sys.stdout.flush()
            
            # Create the table
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
            sys.stdout.write(f"✓ Table '{m2m_table}' created successfully!\n")
            sys.stdout.flush()
            
            # Verify
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{m2m_table}'")
            if cursor.fetchone():
                sys.stdout.write(f"✓ Verified: '{m2m_table}' exists\n")
                sys.stdout.flush()
                
except Exception as e:
    sys.stdout.write(f"ERROR: {e}\n")
    sys.stdout.flush()
    import traceback
    traceback.print_exc()
