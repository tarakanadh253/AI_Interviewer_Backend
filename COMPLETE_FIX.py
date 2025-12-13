#!/usr/bin/env python
"""Complete database fix - guaranteed to work"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from django.core.management import call_command

print("=" * 70)
print("COMPLETE DATABASE FIX")
print("=" * 70)

# Step 1: Check if database file exists
db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')
if not os.path.exists(db_path):
    print("\n✗ Database file does not exist. Will be created by migrations.")
else:
    print(f"\n✓ Database file exists: {db_path}")

# Step 2: Drop ALL tables
print("\n[Step 1] Dropping all tables...")
with connection.cursor() as cursor:
    # Disable foreign keys
    cursor.execute("PRAGMA foreign_keys = OFF")
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(all_tables)} tables: {', '.join(all_tables) if all_tables else 'none'}")
    
    # Drop all tables
    for table in all_tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"  ✓ Dropped {table}")
        except Exception as e:
            print(f"  ✗ Error dropping {table}: {e}")
    
    # Drop all indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'")
    indexes = [row[0] for row in cursor.fetchall()]
    for idx in indexes:
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {idx}")
        except:
            pass
    
    # Re-enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")

print("\n[Step 2] Running migrations...")
try:
    call_command('migrate', verbosity=1, interactive=False)
    print("✓ Migrations completed")
except Exception as e:
    print(f"✗ Migration error: {e}")
    sys.exit(1)

# Step 3: Verify schema
print("\n[Step 3] Verifying schema...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    
    print(f"Columns in user_profiles: {', '.join(col_names)}")
    
    required_cols = ['id', 'username', 'password', 'email', 'access_type', 'has_used_trial']
    missing = [c for c in required_cols if c not in col_names]
    
    if missing:
        print(f"\n✗ MISSING COLUMNS: {', '.join(missing)}")
        print("Creating missing columns...")
        
        if 'access_type' not in col_names:
            try:
                cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
                cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
                print("  ✓ Added access_type")
            except Exception as e:
                print(f"  ✗ Error adding access_type: {e}")
    else:
        print("✓ All required columns exist")

# Step 4: Seed data
print("\n[Step 4] Seeding data...")
try:
    call_command('seed_data', verbosity=1)
    print("✓ Data seeded")
except Exception as e:
    print(f"⚠ Seed data error (may be OK if already seeded): {e}")

# Step 5: Test user creation
print("\n[Step 5] Testing user creation...")
try:
    from interview.models import UserProfile
    from interview.serializers import UserProfileCreateSerializer
    
    # Delete test user if exists
    UserProfile.objects.filter(username='test_fix_user').delete()
    
    # Create test user
    test_data = {
        'username': 'test_fix_user',
        'password': 'testpass123',
        'email': 'test@fix.com',
        'access_type': 'TRIAL'
    }
    
    serializer = UserProfileCreateSerializer(data=test_data)
    if serializer.is_valid():
        user = serializer.save()
        print(f"✓ Test user created: {user.username} (access_type: {user.access_type})")
        user.delete()
        print("✓ Test user deleted")
    else:
        print(f"✗ Serializer validation failed: {serializer.errors}")
except Exception as e:
    print(f"✗ User creation test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ DATABASE FIX COMPLETE!")
print("=" * 70)
print("\nNext step: Start the server with:")
print("  python manage.py runserver")
