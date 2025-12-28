#!/usr/bin/env python
"""Detailed test of user creation to find the exact error"""
import os
import sys
import django
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from interview.models import UserProfile
from interview.serializers import UserProfileCreateSerializer

print("=" * 70)
print("DETAILED USER CREATION TEST")
print("=" * 70)

# Step 1: Check database
print("\n[1] Database Check")
print("-" * 70)
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [(c[1], c[2], c[3]) for c in cursor.fetchall()]
    print("Columns in user_profiles:")
    for name, type, notnull in cols:
        print(f"  - {name}: {type} (NOT NULL: {notnull})")
    
    has_access_type = any('access_type' in name for name, _, _ in cols)
    print(f"\nHas access_type column: {has_access_type}")
    
    if not has_access_type:
        print("\nAdding access_type column...")
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("✓ Column added!")
        except Exception as e:
            print(f"✗ Error: {e}")
            traceback.print_exc()

# Step 2: Test serializer validation
print("\n[2] Serializer Validation Test")
print("-" * 70)
test_data = {
    'username': 'test_create_user_123',
    'password': 'testpass123',
    'email': 'testcreate@example.com',
    'name': 'Test Create User',
    'is_active': True,
    'access_type': 'TRIAL'
}

# Clean up
UserProfile.objects.filter(username='test_create_user_123').delete()

print(f"Test data: {test_data}")
serializer = UserProfileCreateSerializer(data=test_data)
print(f"\nIs valid: {serializer.is_valid()}")

if not serializer.is_valid():
    print(f"\nValidation errors:")
    for field, errors in serializer.errors.items():
        print(f"  {field}: {errors}")
else:
    print("✓ Validation passed!")

# Step 3: Test actual creation
print("\n[3] User Creation Test")
print("-" * 70)
try:
    if serializer.is_valid():
        print("Attempting to save user...")
        user = serializer.save()
        print(f"✓ User created successfully!")
        print(f"  ID: {user.id}")
        print(f"  Username: {user.username}")
        print(f"  Email: {user.email}")
        print(f"  Access Type: {user.access_type}")
        print(f"  Is Active: {user.is_active}")
        
        # Verify in database
        db_user = UserProfile.objects.get(id=user.id)
        print(f"\n✓ Verified in database:")
        print(f"  Username: {db_user.username}")
        print(f"  Access Type: {db_user.access_type}")
        
        # Clean up
        user.delete()
        print("\n✓ Test user deleted")
    else:
        print("✗ Cannot create user - validation failed")
except Exception as e:
    print(f"✗ ERROR creating user: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
    # Check if user was partially created
    try:
        partial_user = UserProfile.objects.get(username='test_create_user_123')
        print(f"\n⚠ User was partially created! ID: {partial_user.id}")
        partial_user.delete()
        print("✓ Cleaned up partial user")
    except UserProfile.DoesNotExist:
        pass

# Step 4: Test with minimal data
print("\n[4] Test with Minimal Data (no access_type)")
print("-" * 70)
minimal_data = {
    'username': 'test_minimal_user',
    'password': 'testpass123',
    'email': 'minimal@example.com',
    'is_active': True
}

UserProfile.objects.filter(username='test_minimal_user').delete()

try:
    serializer2 = UserProfileCreateSerializer(data=minimal_data)
    print(f"Data: {minimal_data}")
    print(f"Is valid: {serializer2.is_valid()}")
    
    if serializer2.is_valid():
        user2 = serializer2.save()
        print(f"✓ User created: {user2.username}, access_type: {user2.access_type}")
        user2.delete()
        print("✓ Test user deleted")
    else:
        print(f"✗ Validation errors: {serializer2.errors}")
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
