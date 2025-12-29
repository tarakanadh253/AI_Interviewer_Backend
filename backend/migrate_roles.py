
import os
import django
import sys

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import UserProfile

def migrate_roles():
    """Migrate ACCESS_TYPE 'ADMIN' to ROLE 'ADMIN'"""
    # Use filter access_type='ADMIN' even though choice is technically removed, database still holds string 'ADMIN'
    users = UserProfile.objects.all()
    print(f"Checking {users.count()} users...")
    
    for user in users:
        print(f"User: {user.username}, Access: {user.access_type}, Role: {user.role}")
        
        # If user was previously marked as ADMIN via access_type
        if user.access_type == 'ADMIN' or user.username == 'admin@ohg':
            user.role = 'ADMIN'
            user.access_type = 'FULL' # Admins can have full access implicitly
            user.save()
            print(f"-> Migrated {user.username} to ROLE: ADMIN")
        
        # Ensure default users are role USER
        elif user.role != 'ADMIN':
            # Default is USER, access_type remains (TRIAL/FULL)
            pass

if __name__ == '__main__':
    migrate_roles()
