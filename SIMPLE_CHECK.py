import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

# Check database
cursor = connection.cursor()
cursor.execute("PRAGMA table_info(user_profiles)")
cols = [c[1] for c in cursor.fetchall()]
print("Database columns:", ', '.join(cols))
print("Has access_type:", 'access_type' in cols)

if 'access_type' not in cols:
    print("\nAdding access_type column...")
    cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
    cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
    print("✓ Column added!")

# Test model
from interview.models import UserProfile
users = UserProfile.objects.all()
print(f"\nUsers in database: {users.count()}")

if users.exists():
    user = users.first()
    print(f"Sample user: {user.username}")
    try:
        print(f"access_type: {user.access_type}")
    except:
        print("access_type: ERROR accessing")

# Test serializer
from interview.serializers import UserProfileSerializer
if users.exists():
    try:
        serializer = UserProfileSerializer(users.first())
        data = serializer.data
        print(f"\n✓ Serialization works!")
        print(f"access_type in response: {data.get('access_type', 'MISSING')}")
    except Exception as e:
        print(f"\n✗ Serialization error: {e}")

print("\n✓ Database check complete!")
print("\nNext: Start Django server with: python manage.py runserver")
print("Then test: http://localhost:8000/api/users/")
