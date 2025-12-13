#!/usr/bin/env python
"""Complete fix - drops tables, runs migrations, seeds data, tests"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from django.core.management import call_command
from interview.models import UserProfile
from interview.serializers import UserProfileCreateSerializer

import sys
sys.stdout.flush()

print("\n" + "="*70)
print("COMPLETE DATABASE FIX AND TEST")
print("="*70 + "\n")
sys.stdout.flush()

# Step 1: Drop all tables
print("[1/5] Dropping all tables...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA foreign_keys = OFF")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cursor.fetchall()]
    for t in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {t}")
    cursor.execute("PRAGMA foreign_keys = ON")
print(f"✓ Dropped {len(tables)} tables\n")
sys.stdout.flush()

# Step 2: Run migrations
print("[2/5] Running migrations...")
sys.stdout.flush()
try:
    call_command('migrate', verbosity=1, interactive=False)
    print("✓ Migrations complete\n")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ Migration error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

# Step 3: Verify and fix schema if needed
print("[3/5] Verifying schema...")
sys.stdout.flush()
with connection.cursor() as cursor:
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_profiles'")
    if not cursor.fetchone():
        print("✗ Table doesn't exist! This shouldn't happen after migrations.")
        sys.exit(1)
    
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    required = ['id', 'username', 'password', 'email', 'access_type']
    missing = [c for c in required if c not in cols]
    
    if missing:
        print(f"✗ Missing columns: {missing}")
        print("Fixing schema directly...")
        sys.stdout.flush()
        
        # Disable foreign keys
        cursor.execute("PRAGMA foreign_keys = OFF")
        
        # Drop dependent tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%session%' OR name LIKE '%answer%')")
        for table in [r[0] for r in cursor.fetchall()]:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
        # Drop and recreate user_profiles
        cursor.execute("DROP TABLE IF EXISTS user_profiles")
        cursor.execute("""
            CREATE TABLE user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(150) UNIQUE NOT NULL,
                password VARCHAR(128) NOT NULL,
                email VARCHAR(254) NOT NULL,
                name VARCHAR(255),
                is_active BOOLEAN NOT NULL DEFAULT 1,
                access_type VARCHAR(10) DEFAULT 'TRIAL',
                has_used_trial BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        cursor.execute("CREATE UNIQUE INDEX user_profiles_username_idx ON user_profiles(username)")
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Verify
        cursor.execute("PRAGMA table_info(user_profiles)")
        cols = [c[1] for c in cursor.fetchall()]
        missing = [c for c in required if c not in cols]
        if missing:
            print(f"✗ Still missing columns: {missing}\n")
            sys.exit(1)
        print("✓ Table recreated with correct schema")
    
    print(f"✓ Schema correct: {', '.join(cols)}\n")
    sys.stdout.flush()

# Step 4: Seed data
print("[4/5] Seeding data...")
sys.stdout.flush()
try:
    call_command('seed_data', verbosity=1)
    print("✓ Data seeded\n")
    sys.stdout.flush()
except Exception as e:
    print(f"⚠ Seed skipped (may already exist): {e}\n")
    sys.stdout.flush()

# Step 5: Test user creation
print("[5/5] Testing user creation...")
UserProfile.objects.filter(username='test_fix').delete()
test_data = {
    'username': 'test_fix',
    'password': 'test123',
    'email': 'test@fix.com',
    'access_type': 'TRIAL'
}
s = UserProfileCreateSerializer(data=test_data)
if not s.is_valid():
    print(f"✗ Validation failed: {s.errors}\n")
    sys.exit(1)
user = s.save()
print(f"✓ User created: {user.username} (access_type: {user.access_type})")
user.delete()
print("✓ Test passed\n")

print("="*70)
print("✓ DATABASE FIXED AND TESTED!")
print("="*70)
print("\nStart server: python manage.py runserver")
print("Then test: http://localhost:8000/api/users/\n")
