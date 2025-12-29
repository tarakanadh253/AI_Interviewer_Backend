#!/usr/bin/env python
"""Ensure interview_sessions table exists"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    django.setup()
    print("Django setup: OK")
    
    from django.core.management import call_command
    from django.db import connection
    
    # Check if table exists
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interview_sessions'")
        exists = cursor.fetchone() is not None
    
    if exists:
        print("✓ interview_sessions table already exists")
        # Show columns
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA table_info(interview_sessions)")
            cols = [row[1] for row in cursor.fetchall()]
            print(f"  Columns: {', '.join(cols)}")
        
        # Check M2M table
        from interview.models import InterviewSession
        m2m_field = InterviewSession._meta.get_field('topics')
        m2m_table = m2m_field.remote_field.through._meta.db_table
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{m2m_table}'")
            m2m_exists = cursor.fetchone() is not None
            if not m2m_exists:
                print(f"✗ M2M table '{m2m_table}' does NOT exist - creating it...")
                # Create M2M table
                through_model = m2m_field.remote_field.through
                session_fk = through_model._meta.get_field('interviewsession')
                topic_fk = through_model._meta.get_field('topic')
                session_col = session_fk.column
                topic_col = topic_fk.column
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
                print(f"✓ M2M table '{m2m_table}' created")
            else:
                print(f"✓ M2M table '{m2m_table}' exists")
    else:
        print("✗ interview_sessions table does NOT exist")
        print("Creating it now...")
        
        # Run syncdb to create all tables
        call_command('migrate', '--run-syncdb', verbosity=1, interactive=False)
        
        # Verify it was created
        with connection.cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interview_sessions'")
            exists = cursor.fetchone() is not None
        
        if exists:
            print("✓ interview_sessions table created successfully!")
        else:
            print("✗ Failed to create interview_sessions table")
            print("Trying to create it manually...")
            
            # Manual creation as last resort - match Django model exactly
            from interview.models import InterviewSession
            
            with connection.cursor() as cursor:
                # Drop table if it exists with wrong schema
                cursor.execute("DROP TABLE IF EXISTS interview_sessions")
                
                # Create main table matching Django model
                cursor.execute("""
                    CREATE TABLE interview_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        started_at DATETIME NOT NULL,
                        ended_at DATETIME NULL,
                        duration_seconds INTEGER NULL,
                        status VARCHAR(20) NOT NULL DEFAULT 'CREATED',
                        communication_score REAL NULL,
                        technology_score REAL NULL,
                        result_summary TEXT NULL,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES user_profiles(id)
                    )
                """)
                
                # Get Django's ManyToMany table name and column names
                m2m_field = InterviewSession._meta.get_field('topics')
                m2m_table = m2m_field.remote_field.through._meta.db_table
                print(f"  Django M2M table name: {m2m_table}")
                
                # Get the actual column names Django uses
                through_model = m2m_field.remote_field.through
                session_fk = through_model._meta.get_field('interviewsession')
                topic_fk = through_model._meta.get_field('topic')
                session_col = session_fk.column
                topic_col = topic_fk.column
                
                # Create ManyToMany join table using Django's naming convention
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {m2m_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        {session_col} INTEGER NOT NULL,
                        {topic_col} INTEGER NOT NULL,
                        FOREIGN KEY ({session_col}) REFERENCES interview_sessions(id) ON DELETE CASCADE,
                        FOREIGN KEY ({topic_col}) REFERENCES topics(id) ON DELETE CASCADE,
                        UNIQUE({session_col}, {topic_col})
                    )
                """)
                
                # Commit the transaction
                connection.commit()
                
                print("✓ Table created manually with correct schema")
                print(f"✓ ManyToMany join table '{m2m_table}' created")
                
                # Verify tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interview_sessions'")
                if cursor.fetchone():
                    print("✓ Verified: interview_sessions table exists")
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{m2m_table}'")
                if cursor.fetchone():
                    print(f"✓ Verified: {m2m_table} table exists")
                
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
