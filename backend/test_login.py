
import os
import django
import sys
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import UserProfile
from interview.serializers import UserProfileSerializer

try:
    user = UserProfile.objects.get(username='admin@ohg')
    print(f"DB User: {user.username}") # Removed Access Type from this line as it will be printed from serialized data
    
    serializer = UserProfileSerializer(user)
    data = serializer.data
    
    print(f"Role: {data.get('role')}")
    print(f"Access Type: {data.get('access_type')}")
    print("-" * 30)
    
    print(f"Serialized Data: {json.dumps(data, indent=2)}")
    
    if data.get('access_type') == 'ADMIN':
        print("SUCCESS: Serializer returns ADMIN")
    else:
        print(f"FAILURE: Serializer returns {data.get('access_type')}")

except UserProfile.DoesNotExist:
    print("User admin@ohg not found")
except Exception as e:
    print(f"Error: {e}")
