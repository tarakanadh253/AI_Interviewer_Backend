#!/usr/bin/env python
"""Capture the actual error from Django when creating/fetching users"""
import os
import sys
import django
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

# Enable Django logging
import logging
logging.basicConfig(level=logging.DEBUG)

django.setup()

from django.db import connection
from interview.models import UserProfile
from interview.serializers import UserProfileSerializer, UserProfileCreateSerializer
from rest_framework.test import APIRequestFactory
from interview.views import UserProfileViewSet

print("=" * 70)
print("CAPTURING ACTUAL ERROR")
print("=" * 70)

# Step 1: Check database
print("\n[1] Database Check")
print("-" * 70)
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [(c[1], c[2]) for c in cursor.fetchall()]
    print("Columns:")
    for name, type in cols:
        print(f"  - {name}: {type}")
    
    has_access_type = any('access_type' in name for name, _ in cols)
    print(f"\nHas access_type: {has_access_type}")
    
    if not has_access_type:
        print("\nAdding column...")
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            print("✓ Added")
        except Exception as e:
            print(f"Error: {e}")
            try:
                cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
                cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
                print("✓ Added (fallback)")
            except Exception as e2:
                print(f"Error: {e2}")

# Step 2: Test listing users (GET /api/users/)
print("\n[2] Testing GET /api/users/ (listing users)")
print("-" * 70)
try:
    users = UserProfile.objects.all()
    print(f"Found {users.count()} users in database")
    
    if users.exists():
        # Test serializer on first user
        user = users.first()
        print(f"\nTesting serializer on user: {user.username}")
        try:
            serializer = UserProfileSerializer(user)
            data = serializer.data
            print(f"✓ Serialization successful")
            print(f"  Keys: {list(data.keys())}")
            print(f"  access_type: {data.get('access_type', 'MISSING')}")
        except Exception as e:
            print(f"✗ Serialization error: {e}")
            traceback.print_exc()
            
            # Try to access access_type directly
            try:
                print(f"\nTrying to access access_type directly...")
                at = user.access_type
                print(f"  Direct access: {at}")
            except Exception as e2:
                print(f"  Direct access error: {e2}")
    else:
        print("No users to test")
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Step 3: Test creating user (POST /api/users/)
print("\n[3] Testing POST /api/users/ (creating user)")
print("-" * 70)
test_data = {
    'username': 'error_test_user',
    'password': 'testpass123',
    'email': 'errortest@example.com',
    'name': 'Error Test',
    'is_active': True,
    'access_type': 'TRIAL'
}

UserProfile.objects.filter(username='error_test_user').delete()

try:
    serializer = UserProfileCreateSerializer(data=test_data)
    print(f"Test data: {test_data}")
    print(f"Serializer valid: {serializer.is_valid()}")
    
    if not serializer.is_valid():
        print(f"✗ Validation errors: {serializer.errors}")
    else:
        print("✓ Validation passed, attempting to save...")
        try:
            user = serializer.save()
            print(f"✓ User created: {user.username}")
            print(f"  ID: {user.id}")
            print(f"  Access type: {user.access_type}")
            user.delete()
            print("✓ Test user deleted")
        except Exception as e:
            print(f"✗ Save error: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
except Exception as e:
    print(f"✗ Error: {e}")
    traceback.print_exc()

# Step 4: Test via ViewSet (simulating API call)
print("\n[4] Testing via ViewSet (simulating API)")
print("-" * 70)
factory = APIRequestFactory()
viewset = UserProfileViewSet()

# Test GET
try:
    request = factory.get('/api/users/')
    viewset.request = request
    viewset.action = 'list'
    
    queryset = viewset.get_queryset()
    serializer = viewset.get_serializer(queryset, many=True)
    data = serializer.data
    print(f"✓ GET /api/users/ works")
    print(f"  Returned {len(data)} users")
except Exception as e:
    print(f"✗ GET error: {e}")
    traceback.print_exc()

# Test POST
test_data2 = {
    'username': 'viewset_test_user',
    'password': 'testpass123',
    'email': 'viewset@test.com',
    'access_type': 'TRIAL'
}
UserProfile.objects.filter(username='viewset_test_user').delete()

try:
    request = factory.post('/api/users/', test_data2, format='json')
    viewset.request = request
    viewset.action = 'create'
    
    serializer = viewset.get_serializer(data=test_data2)
    if serializer.is_valid():
        user = serializer.save()
        print(f"✓ POST /api/users/ works")
        print(f"  Created: {user.username}")
        user.delete()
    else:
        print(f"✗ POST validation error: {serializer.errors}")
except Exception as e:
    print(f"✗ POST error: {e}")
    traceback.print_exc()

print("\n" + "=" * 70)
print("ERROR CAPTURE COMPLETE")
print("=" * 70)
