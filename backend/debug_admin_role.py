import os
import django
import sys
import json

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import UserProfile
from interview.serializers import UserProfileSerializer

def check_admin_serialization():
    try:
        user = UserProfile.objects.get(username='admin@ohg')
        print(f"User found: {user.username}, Role: '{user.role}'")
        
        serializer = UserProfileSerializer(user)
        data = serializer.data
        print("\nSerialized Data:")
        print(json.dumps(data, indent=2))
        
        if data.get('role') == 'ADMIN':
            print("\nSUCCESS: Role is correctly serialized as 'ADMIN'")
        else:
            print(f"\nFAILURE: Role is serialized as '{data.get('role')}'")
            
    except UserProfile.DoesNotExist:
        print("Admin user not found!")

if __name__ == "__main__":
    check_admin_serialization()
