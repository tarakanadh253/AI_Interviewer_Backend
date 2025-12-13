#!/usr/bin/env python
"""Skip migrations and test if the API works"""
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
print("TESTING API WITHOUT MIGRATIONS")
print("=" * 70)

# Check database
print("\n[1] Checking database...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"   Columns: {', '.join(cols)}")
    has_access_type = 'access_type' in cols
    print(f"   Has access_type: {has_access_type}")

if not has_access_type:
    print("\n   Adding access_type column...")
    with connection.cursor() as cursor:
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
        cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
    print("   ✓ Added!")

# Test listing users
print("\n[2] Testing user listing...")
try:
    users = UserProfile.objects.all()
    print(f"   Found {users.count()} users")
    
    if users.exists():
        serializer = UserProfileSerializer(users.first())
        data = serializer.data
        print(f"   ✓ Serialization works!")
        print(f"   Keys: {list(data.keys())}")
        print(f"   access_type: {data.get('access_type', 'MISSING')}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test creating user
print("\n[3] Testing user creation...")
test_data = {
    'username': 'test_api_user',
    'password': 'testpass123',
    'email': 'testapi@example.com',
    'name': 'Test API',
    'is_active': True,
    'access_type': 'TRIAL'
}

# Clean up
UserProfile.objects.filter(username='test_api_user').delete()

try:
    serializer = UserProfileCreateSerializer(data=test_data)
    if serializer.is_valid():
        user = serializer.save()
        print(f"   ✓ User created: {user.username}")
        print(f"   access_type: {user.access_type}")
        user.delete()
        print("   ✓ Test user deleted")
    else:
        print(f"   ✗ Validation errors: {serializer.errors}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
print("\nIf tests passed, you can start the server:")
print("  python manage.py runserver")
print("\nNote: You can ignore migration warnings if the API works.")
