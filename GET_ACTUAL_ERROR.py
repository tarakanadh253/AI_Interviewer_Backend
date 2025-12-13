#!/usr/bin/env python
"""Get the actual error from Django when accessing /api/users/"""
import os
import sys
import django
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

# Enable detailed error logging
import logging
logging.basicConfig(level=logging.DEBUG)

django.setup()

print("=" * 70)
print("GETTING ACTUAL ERROR FROM DJANGO")
print("=" * 70)

# First, ensure database is fixed
print("\n[1] Fixing database...")
from django.db import connection
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"Columns: {', '.join(cols)}")
    
    if 'access_type' not in cols:
        print("Adding access_type column...")
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("✓ Column added")
        except Exception as e:
            print(f"Error: {e}")

# Now test what happens when we try to list users
print("\n[2] Testing UserProfile.objects.all()...")
try:
    from interview.models import UserProfile
    users = UserProfile.objects.all()
    print(f"✓ Query works: {users.count()} users")
    
    # Try to access each user
    for user in users[:3]:
        try:
            print(f"  User: {user.username}, access_type: {getattr(user, 'access_type', 'MISSING')}")
        except Exception as e:
            print(f"  User: {user.username}, ERROR: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"✗ Query error: {e}")
    traceback.print_exc()

# Test serialization
print("\n[3] Testing serialization...")
try:
    from interview.serializers import UserProfileSerializer
    if users.exists():
        user = users.first()
        print(f"Serializing user: {user.username}")
        serializer = UserProfileSerializer(user)
        data = serializer.data
        print(f"✓ Serialization successful!")
        print(f"  Response keys: {list(data.keys())}")
    else:
        print("No users to serialize (this is OK - empty list should work)")
        
        # Test with empty queryset
        serializer = UserProfileSerializer([], many=True)
        data = serializer.data
        print(f"✓ Empty list serialization works: {data}")
except Exception as e:
    print(f"✗ Serialization error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test ViewSet list action
print("\n[4] Testing ViewSet.list()...")
try:
    from rest_framework.test import APIRequestFactory
    from interview.views import UserProfileViewSet
    
    factory = APIRequestFactory()
    request = factory.get('/api/users/')
    
    viewset = UserProfileViewSet()
    viewset.request = request
    viewset.action = 'list'
    viewset.format_kwarg = None
    
    # Get queryset
    queryset = viewset.get_queryset()
    print(f"✓ Queryset: {queryset.count()} users")
    
    # Get serializer
    serializer = viewset.get_serializer(queryset, many=True)
    data = serializer.data
    print(f"✓ Serialization: {len(data)} users in response")
    
    # Try to get response (this is what the API actually returns)
    from rest_framework.response import Response
    response = Response(serializer.data)
    print(f"✓ Response created successfully")
    
except Exception as e:
    print(f"✗ ViewSet error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("ERROR CHECK COMPLETE")
print("=" * 70)
print("\nIf you see errors above, those are the issues causing 500 errors.")
print("If everything shows ✓, the backend should work.")
