#!/usr/bin/env python
"""Verify database schema and fix if needed"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from interview.models import UserProfile
from interview.serializers import UserProfileCreateSerializer

print("=" * 70)
print("VERIFYING DATABASE AND TESTING USER CREATION")
print("=" * 70)

# Check database schema
print("\n1. Checking database schema...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"   Columns found: {', '.join(cols)}")
    
    if 'access_type' not in cols:
        print("   ✗ access_type column is MISSING!")
        print("   Adding column...")
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("   ✓ Column added successfully!")
        except Exception as e:
            print(f"   ✗ Error adding column: {e}")
    else:
        print("   ✓ access_type column exists!")

# Test serializer
print("\n2. Testing user creation serializer...")
test_data = {
    'username': 'test_verify_user',
    'password': 'testpass123',
    'email': 'testverify@example.com',
    'name': 'Test Verify',
    'is_active': True,
    'access_type': 'TRIAL'
}

# Clean up any existing test user
UserProfile.objects.filter(username='test_verify_user').delete()

try:
    serializer = UserProfileCreateSerializer(data=test_data)
    if serializer.is_valid():
        print("   ✓ Serializer validation passed!")
        user = serializer.save()
        print(f"   ✓ User created: {user.username}, access_type: {user.access_type}")
        user.delete()
        print("   ✓ Test user cleaned up")
    else:
        print(f"   ✗ Serializer validation failed:")
        for field, errors in serializer.errors.items():
            print(f"      {field}: {errors}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test without access_type (should use default)
print("\n3. Testing user creation without access_type (should use default)...")
test_data_no_access = {
    'username': 'test_default_user',
    'password': 'testpass123',
    'email': 'testdefault@example.com',
    'name': 'Test Default',
    'is_active': True
}

UserProfile.objects.filter(username='test_default_user').delete()

try:
    serializer = UserProfileCreateSerializer(data=test_data_no_access)
    if serializer.is_valid():
        print("   ✓ Serializer validation passed (without access_type)!")
        user = serializer.save()
        print(f"   ✓ User created: {user.username}, access_type: {user.access_type}")
        user.delete()
        print("   ✓ Test user cleaned up")
    else:
        print(f"   ✗ Serializer validation failed:")
        for field, errors in serializer.errors.items():
            print(f"      {field}: {errors}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
