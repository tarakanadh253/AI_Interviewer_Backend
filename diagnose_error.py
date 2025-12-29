#!/usr/bin/env python
"""Diagnose the 500 error by checking database and testing API"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from interview.models import UserProfile
from interview.serializers import UserProfileSerializer, UserProfileCreateSerializer

print("=" * 70)
print("DIAGNOSING 500 ERROR")
print("=" * 70)

# 1. Check database schema
print("\n[1] Checking database schema...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"   Columns: {', '.join(cols)}")
    has_access_type = 'access_type' in cols
    print(f"   Has access_type: {has_access_type}")

# 2. Check existing users
print("\n[2] Checking existing users...")
users = UserProfile.objects.all()
print(f"   Total users: {users.count()}")
for user in users[:5]:  # Check first 5
    try:
        print(f"   - {user.username}: access_type = {getattr(user, 'access_type', 'MISSING')}")
    except Exception as e:
        print(f"   - {user.username}: ERROR accessing access_type - {e}")

# 3. Test serializer on existing users
print("\n[3] Testing serializer on existing users...")
if users.exists():
    user = users.first()
    try:
        serializer = UserProfileSerializer(user)
        data = serializer.data
        print(f"   ✓ Serializer works: {list(data.keys())}")
        if 'access_type' in data:
            print(f"   ✓ access_type in data: {data['access_type']}")
        else:
            print(f"   ✗ access_type NOT in serializer data!")
    except Exception as e:
        print(f"   ✗ Serializer error: {e}")
        import traceback
        traceback.print_exc()

# 4. Test creating a new user
print("\n[4] Testing user creation...")
test_data = {
    'username': 'diagnose_test_user',
    'password': 'testpass123',
    'email': 'diagnose@test.com',
    'name': 'Diagnose Test',
    'is_active': True,
    'access_type': 'TRIAL'
}

# Clean up
UserProfile.objects.filter(username='diagnose_test_user').delete()

try:
    serializer = UserProfileCreateSerializer(data=test_data)
    print(f"   Serializer valid: {serializer.is_valid()}")
    if not serializer.is_valid():
        print(f"   Errors: {serializer.errors}")
    else:
        user = serializer.save()
        print(f"   ✓ User created: {user.username}, access_type: {user.access_type}")
        user.delete()
except Exception as e:
    print(f"   ✗ Error creating user: {e}")
    import traceback
    traceback.print_exc()

# 5. Fix if needed
print("\n[5] Applying fixes if needed...")
if not has_access_type:
    print("   Adding access_type column...")
    with connection.cursor() as cursor:
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("   ✓ Column added!")
        except Exception as e:
            print(f"   ✗ Error: {e}")

# 6. Check for users with NULL access_type
print("\n[6] Checking for users with NULL access_type...")
with connection.cursor() as cursor:
    cursor.execute("SELECT COUNT(*) FROM user_profiles WHERE access_type IS NULL;")
    null_count = cursor.fetchone()[0]
    if null_count > 0:
        print(f"   Found {null_count} users with NULL access_type, fixing...")
        cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
        print("   ✓ Fixed!")
    else:
        print("   ✓ No users with NULL access_type")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
