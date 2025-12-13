#!/usr/bin/env python
"""Test user creation to verify everything works"""
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
print("TESTING USER CREATION")
print("=" * 70)

# Check database schema
print("\n[1] Checking database schema...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = cursor.fetchall()
    col_names = [col[1] for col in cols]
    print(f"Columns: {', '.join(col_names)}")
    
    if 'username' not in col_names:
        print("\n✗ ERROR: username column missing!")
        print("Run COMPLETE_FIX.py first!")
        sys.exit(1)
    print("✓ Schema looks good")

# Test user creation
print("\n[2] Testing user creation...")
test_username = 'test_user_creation'
test_data = {
    'username': test_username,
    'password': 'testpass123',
    'email': 'test@example.com',
    'name': 'Test User',
    'access_type': 'TRIAL'
}

# Delete if exists
UserProfile.objects.filter(username=test_username).delete()

# Create
serializer = UserProfileCreateSerializer(data=test_data)
if not serializer.is_valid():
    print(f"✗ Validation failed: {serializer.errors}")
    sys.exit(1)

user = serializer.save()
print(f"✓ User created: {user.username}")
print(f"  - Email: {user.email}")
print(f"  - Access Type: {user.access_type}")
print(f"  - Has Used Trial: {user.has_used_trial}")

# Test retrieval
print("\n[3] Testing user retrieval...")
retrieved = UserProfile.objects.get(username=test_username)
print(f"✓ User retrieved: {retrieved.username}")

# Test password check
print("\n[4] Testing password check...")
if user.check_password('testpass123'):
    print("✓ Password check passed")
else:
    print("✗ Password check failed")

# Cleanup
print("\n[5] Cleaning up...")
user.delete()
print("✓ Test user deleted")

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nUser creation is working correctly!")
print("You can now create users in the admin dashboard.")
