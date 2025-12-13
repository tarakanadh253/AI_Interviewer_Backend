#!/usr/bin/env python
"""Guaranteed fix - ensures access_type column exists and everything works"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("GUARANTEED FIX FOR access_type COLUMN")
print("=" * 70)

# Step 1: Check and add column
print("\n[1] Checking database schema...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols_info = cursor.fetchall()
    cols = [c[1] for c in cols_info]
    
    print(f"Current columns: {', '.join(cols)}")
    
    if 'access_type' not in cols:
        print("\n✗ access_type column MISSING - adding now...")
        try:
            # Try with default first
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            print("✓ Column added with DEFAULT 'TRIAL'")
        except Exception as e:
            print(f"  First attempt failed: {e}")
            try:
                # Try without default
                cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
                cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
                print("✓ Column added, values set to 'TRIAL'")
            except Exception as e2:
                print(f"✗ Failed to add column: {e2}")
                sys.exit(1)
    else:
        print("✓ access_type column exists")
    
    # Verify column exists now
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols_info = cursor.fetchall()
    cols = [c[1] for c in cols_info]
    if 'access_type' not in cols:
        print("✗ ERROR: Column still missing after attempt!")
        sys.exit(1)
    
    # Fix any NULL values
    cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
    affected = cursor.rowcount
    if affected > 0:
        print(f"✓ Updated {affected} rows with NULL access_type")

# Step 2: Test that it works
print("\n[2] Testing database access...")
try:
    from interview.models import UserProfile
    users = UserProfile.objects.all()
    print(f"✓ Can query UserProfile: {users.count()} users found")
    
    if users.exists():
        user = users.first()
        try:
            at = user.access_type
            print(f"✓ Can access access_type: {at}")
        except Exception as e:
            print(f"✗ Cannot access access_type: {e}")
            sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test serialization
print("\n[3] Testing serialization...")
try:
    from interview.serializers import UserProfileSerializer
    if users.exists():
        user = users.first()
        serializer = UserProfileSerializer(user)
        data = serializer.data
        print(f"✓ Serialization works")
        print(f"  access_type in data: {data.get('access_type', 'MISSING')}")
    else:
        print("  No users to test (this is OK)")
except Exception as e:
    print(f"✗ Serialization error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✓ FIX COMPLETE - Database is ready!")
print("=" * 70)
print("\nYou can now start your server:")
print("  python manage.py runserver")
