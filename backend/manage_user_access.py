
import os
import django
import sys

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import UserProfile

def update_user_access(username, access_type):
    """Update user access type"""
    try:
        user = UserProfile.objects.get(username=username)
        print(f"User found: {user.username} (Current Access: {user.access_type})")
        
        if access_type.upper() not in ['TRIAL', 'FULL', 'ADMIN']:
            print("Error: Invalid access type. Choose from: TRIAL, FULL, ADMIN")
            return False
            
        user.access_type = access_type.upper()
        
        # If Admin, ensure is_active is True
        if user.access_type == 'ADMIN':
            user.is_active = True
            
        user.save()
        print(f"SUCCESS: User {username} updated to {access_type.upper()}")
        return True
    except UserProfile.DoesNotExist:
        print(f"Error: User {username} not found")
        return False
    except Exception as e:
        print(f"Error updating user: {e}")
        return False

def list_users():
    print("\n--- Current Users ---")
    users = UserProfile.objects.all().order_by('-created_at')
    for u in users:
        print(f"- {u.username}: {u.access_type} (Active: {u.is_active})")
    print("---------------------\n")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        list_users()
        print("Usage: python manage_user_access.py <username> <access_type>")
        print("Example: python manage_user_access.py myuser FULL")
        print("Example: python manage_user_access.py adminuser ADMIN")
    else:
        username = sys.argv[1]
        access_type = sys.argv[2]
        update_user_access(username, access_type)
        list_users()
