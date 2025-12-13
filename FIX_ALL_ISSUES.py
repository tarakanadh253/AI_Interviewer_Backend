#!/usr/bin/env python
"""Comprehensive fix for all user-related issues"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from interview.models import UserProfile

print("=" * 70)
print("COMPREHENSIVE FIX FOR USER CREATION/LISTING ERRORS")
print("=" * 70)

# Step 1: Add access_type column if missing
print("\n[1] Checking and fixing database schema...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"   Current columns: {', '.join(cols)}")
    
    if 'access_type' not in cols:
        print("   ✗ access_type column missing - adding...")
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
            print("   ✓ Column added")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            sys.exit(1)
    else:
        print("   ✓ access_type column exists")

# Step 2: Fix any NULL values
print("\n[2] Fixing NULL access_type values...")
with connection.cursor() as cursor:
    cursor.execute("SELECT COUNT(*) FROM user_profiles WHERE access_type IS NULL;")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        print(f"   Found {null_count} users with NULL access_type")
        cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
        print("   ✓ Fixed NULL values")
    else:
        print("   ✓ No NULL values found")

# Step 3: Verify all users have access_type
print("\n[3] Verifying all users...")
users = UserProfile.objects.all()
print(f"   Total users: {users.count()}")
for user in users:
    try:
        # Force refresh from database
        user.refresh_from_db()
        access_type = getattr(user, 'access_type', None)
        if not access_type:
            print(f"   Fixing user {user.username}...")
            user.access_type = 'TRIAL'
            user.save()
            print(f"   ✓ Fixed {user.username}")
    except Exception as e:
        print(f"   ✗ Error with user {user.username}: {e}")

# Step 4: Test serialization
print("\n[4] Testing serialization...")
from interview.serializers import UserProfileSerializer
try:
    if users.exists():
        user = users.first()
        serializer = UserProfileSerializer(user)
        data = serializer.data
        print(f"   ✓ Serialization works")
        print(f"   Keys: {list(data.keys())}")
        if 'access_type' in data:
            print(f"   ✓ access_type in data: {data['access_type']}")
    else:
        print("   No users to test")
except Exception as e:
    print(f"   ✗ Serialization error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("FIX COMPLETE!")
print("=" * 70)
print("\nPlease restart your Django server:")
print("  python manage.py runserver")
