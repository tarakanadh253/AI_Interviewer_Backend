import os
import django
import sys

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import UserProfile

def fix_admin_role():
    try:
        user = UserProfile.objects.get(username='admin@ohg')
        print(f"Current Role: '{user.role}'")
        
        if user.role != 'ADMIN':
            print("Updating role to 'ADMIN'...")
            user.role = 'ADMIN'
            user.access_type = 'FULL' # Ensure full access too
            user.save()
            print("Successfully updated admin user role.")
        else:
            print("Role is already 'ADMIN'.")
            
    except UserProfile.DoesNotExist:
        print("Admin user 'admin@ohg' not found. Creating it...")
        # If it doesn't exist, Create it
        user = UserProfile(
            username='admin@ohg',
            email='admin@ohg.com',
            name='Admin User',
            role='ADMIN',
            access_type='FULL',
            is_active=True
        )
        user.set_password('ohg@365')
        user.save()
        print("Created new admin user.")

if __name__ == "__main__":
    fix_admin_role()
