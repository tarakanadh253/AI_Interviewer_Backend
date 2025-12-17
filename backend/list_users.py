
import os
import django
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import UserProfile

print("USERS:")
for u in UserProfile.objects.all():
    print(f"User: {u.username} | Type: {u.access_type}")
