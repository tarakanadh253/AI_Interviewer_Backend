"""
Fix the answers table - create it if it doesn't exist and add missing columns
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

def fix_answers_table():
    """Create answers table if it doesn't exist and add missing columns"""
    with connection.cursor() as cursor:
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='answers'
        """)
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            print("Creating answers table...")
            cursor.execute("""
                CREATE TABLE answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    question_id INTEGER NOT NULL,
                    user_answer TEXT NOT NULL,
                    similarity_score REAL NOT NULL DEFAULT 0.0,
                    accuracy_score REAL,
                    completeness_score REAL,
                    matched_keywords TEXT NOT NULL DEFAULT '',
                    missing_keywords TEXT NOT NULL DEFAULT '',
                    topic_score REAL,
                    communication_subscore REAL,
                    score_breakdown TEXT,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES interview_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
                    UNIQUE(session_id, question_id)
                )
            """)
            print("✓ Answers table created")
        else:
            print("Answers table exists, checking for missing columns...")
            
            # Get existing columns
            cursor.execute("PRAGMA table_info(answers)")
            existing_columns = {row[1] for row in cursor.fetchall()}
            
            # Add missing columns
            columns_to_add = {
                'accuracy_score': 'REAL',
                'completeness_score': 'REAL',
                'score_breakdown': 'TEXT',
            }
            
            for col_name, col_type in columns_to_add.items():
                if col_name not in existing_columns:
                    print(f"Adding column {col_name}...")
                    try:
                        cursor.execute(f"ALTER TABLE answers ADD COLUMN {col_name} {col_type}")
                        print(f"✓ Added column {col_name}")
                    except Exception as e:
                        print(f"✗ Error adding {col_name}: {e}")
        
        # Create index if it doesn't exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name='answers_session_id_idx'
        """)
        if not cursor.fetchone():
            cursor.execute("CREATE INDEX answers_session_id_idx ON answers(session_id)")
            print("✓ Created index on session_id")
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND name='answers_question_id_idx'
        """)
        if not cursor.fetchone():
            cursor.execute("CREATE INDEX answers_question_id_idx ON answers(question_id)")
            print("✓ Created index on question_id")
        
        print("\n✓ Answers table is ready!")

if __name__ == "__main__":
    try:
        fix_answers_table()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

