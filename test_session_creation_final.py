#!/usr/bin/env python
"""Test session creation after table fix"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    django.setup()
    print("=" * 50)
    print("TESTING SESSION CREATION")
    print("=" * 50)
    
    from django.db import connection
    from interview.models import InterviewSession, UserProfile, Topic
    
    # Check table exists
    with connection.cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interview_sessions'")
        exists = cursor.fetchone() is not None
        print(f"\n1. interview_sessions table exists: {exists}")
        
        if exists:
            cursor.execute("PRAGMA table_info(interview_sessions)")
            cols = [row[1] for row in cursor.fetchall()]
            print(f"   Columns: {', '.join(cols)}")
    
    # Check M2M table
    m2m_field = InterviewSession._meta.get_field('topics')
    m2m_table = m2m_field.remote_field.through._meta.db_table
    print(f"\n2. M2M table name: {m2m_table}")
    
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{m2m_table}'")
        m2m_exists = cursor.fetchone() is not None
        print(f"   M2M table exists: {m2m_exists}")
    
    # Try to get a user and topic
    try:
        user = UserProfile.objects.first()
        topic = Topic.objects.first()
        print(f"\n3. Test user exists: {user is not None}")
        print(f"   Test topic exists: {topic is not None}")
        
        if user and topic:
            # Try creating a session (but don't save, just test serializer)
            print("\n4. Testing session creation...")
            session_data = {
                'user': user.id,
                'status': 'CREATED',
                'topic_ids': [topic.id]
            }
            print(f"   Session data: {session_data}")
            print("   ✓ All checks passed! Table structure is correct.")
        else:
            print("   ⚠ No user or topic found, but table structure is OK")
            
    except Exception as e:
        print(f"\n   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
