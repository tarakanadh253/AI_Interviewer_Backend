#!/usr/bin/env python
"""Check what error the server is actually returning"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.test import RequestFactory
from interview.views import UserProfileViewSet
from rest_framework.test import APIRequestFactory

print("=" * 70)
print("TESTING API ENDPOINT DIRECTLY")
print("=" * 70)

# Create test request
factory = APIRequestFactory()
viewset = UserProfileViewSet()

# Test data
test_data = {
    'username': 'apitest_user',
    'password': 'testpass123',
    'email': 'apitest@example.com',
    'name': 'API Test User',
    'is_active': True,
    'access_type': 'TRIAL'
}

# Clean up
from interview.models import UserProfile
UserProfile.objects.filter(username='apitest_user').delete()

print(f"\nTest data: {test_data}")

# Create request
request = factory.post('/api/users/', test_data, format='json')
request.user = None  # Anonymous user

# Get serializer class
viewset.action = 'create'
serializer_class = viewset.get_serializer_class()
print(f"\nUsing serializer: {serializer_class.__name__}")

# Test serializer
serializer = serializer_class(data=test_data)
print(f"Serializer valid: {serializer.is_valid()}")
if not serializer.is_valid():
    print(f"Errors: {serializer.errors}")
else:
    print("✓ Serializer validation passed")
    
    # Try to create
    try:
        user = serializer.save()
        print(f"✓ User created: {user.username}")
        print(f"  Access type: {user.access_type}")
        user.delete()
        print("✓ Test user deleted")
    except Exception as e:
        print(f"✗ Error saving: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
