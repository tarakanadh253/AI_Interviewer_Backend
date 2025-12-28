#!/usr/bin/env python
"""Complete diagnostic - check everything"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from interview.models import UserProfile
from interview.serializers import UserProfileSerializer, UserProfileCreateSerializer
from rest_framework.test import APIRequestFactory
from interview.views import UserProfileViewSet

print("=" * 70)
print("COMPLETE DIAGNOSTIC - React/Django Connection Check")
print("=" * 70)

# 1. Database Check
print("\n[1] DATABASE CHECK")
print("-" * 70)
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"Columns: {', '.join(cols)}")
    has_access_type = 'access_type' in cols
    print(f"Has access_type: {has_access_type}")
    
    if not has_access_type:
        print("\n⚠️  FIXING: Adding access_type column...")
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("✓ Column added!")
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)

# 2. Model Check
print("\n[2] MODEL CHECK")
print("-" * 70)
try:
    users = UserProfile.objects.all()
    print(f"✓ Can query UserProfile: {users.count()} users")
    
    if users.exists():
        user = users.first()
        print(f"  Sample user: {user.username}")
        print(f"  access_type: {getattr(user, 'access_type', 'MISSING')}")
except Exception as e:
    print(f"✗ Model error: {e}")
    import traceback
    traceback.print_exc()

# 3. Serializer Check - GET
print("\n[3] SERIALIZER CHECK - GET (Listing Users)")
print("-" * 70)
try:
    if users.exists():
        user = users.first()
        serializer = UserProfileSerializer(user)
        data = serializer.data
        print(f"✓ Serialization works")
        print(f"  Keys: {list(data.keys())}")
        print(f"  access_type: {data.get('access_type', 'MISSING')}")
    else:
        print("  No users to test (OK)")
except Exception as e:
    print(f"✗ Serialization error: {e}")
    import traceback
    traceback.print_exc()

# 4. Serializer Check - POST
print("\n[4] SERIALIZER CHECK - POST (Creating User)")
print("-" * 70)
test_data = {
    'username': 'diag_test_user',
    'password': 'testpass123',
    'email': 'diag@test.com',
    'name': 'Diagnostic Test',
    'is_active': True,
    'access_type': 'TRIAL'
}

UserProfile.objects.filter(username='diag_test_user').delete()

try:
    serializer = UserProfileCreateSerializer(data=test_data)
    print(f"Test data: {test_data}")
    print(f"Valid: {serializer.is_valid()}")
    
    if not serializer.is_valid():
        print(f"✗ Validation errors: {serializer.errors}")
    else:
        user = serializer.save()
        print(f"✓ User created: {user.username}")
        print(f"  ID: {user.id}")
        print(f"  access_type: {user.access_type}")
        user.delete()
        print("✓ Test user deleted")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 5. API Endpoint Check
print("\n[5] API ENDPOINT CHECK (Simulating React Request)")
print("-" * 70)
factory = APIRequestFactory()

# Test GET /api/users/
try:
    request = factory.get('/api/users/')
    viewset = UserProfileViewSet()
    viewset.request = request
    viewset.action = 'list'
    
    queryset = viewset.get_queryset()
    serializer = viewset.get_serializer(queryset, many=True)
    data = serializer.data
    print(f"✓ GET /api/users/ works")
    print(f"  Returned {len(data)} users")
    if data:
        print(f"  First user keys: {list(data[0].keys())}")
except Exception as e:
    print(f"✗ GET error: {e}")
    import traceback
    traceback.print_exc()

# Test POST /api/users/
test_data2 = {
    'username': 'api_test_user',
    'password': 'testpass123',
    'email': 'api@test.com',
    'access_type': 'TRIAL'
}
UserProfile.objects.filter(username='api_test_user').delete()

try:
    request = factory.post('/api/users/', test_data2, format='json')
    viewset = UserProfileViewSet()
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
    import traceback
    traceback.print_exc()

# 6. Connection Info
print("\n[6] CONNECTION INFO")
print("-" * 70)
print("Frontend API URL: http://localhost:8000/api")
print("Backend should run on: http://localhost:8000")
print("CORS is configured in settings.py")
print("\nTo test connection:")
print("  1. Start Django: python manage.py runserver")
print("  2. Check: http://localhost:8000/api/users/ in browser")
print("  3. Should return JSON (or empty array if no users)")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
