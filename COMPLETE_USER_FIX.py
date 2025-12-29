#!/usr/bin/env python
"""Complete fix for user creation - ensures everything is correct"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection
from interview.models import UserProfile
from interview.serializers import UserProfileCreateSerializer

print("=" * 70)
print("COMPLETE USER CREATION FIX")
print("=" * 70)

# Step 1: Ensure database column exists
print("\n[1] Ensuring access_type column exists...")
with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = [c[1] for c in cursor.fetchall()]
    
    if 'access_type' not in cols:
        print("   Adding access_type column...")
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
            print("   ✓ Column added with default")
        except:
            try:
                cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
                cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
                print("   ✓ Column added, values updated")
            except Exception as e:
                print(f"   ✗ Error: {e}")
                sys.exit(1)
    else:
        print("   ✓ Column exists")
    
    # Ensure no NULL values
    cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
    print("   ✓ NULL values fixed")

# Step 2: Test user creation
print("\n[2] Testing user creation...")
test_data = {
    'username': 'final_test_user',
    'password': 'testpass123',
    'email': 'finaltest@example.com',
    'name': 'Final Test',
    'is_active': True,
    'access_type': 'TRIAL'
}

UserProfile.objects.filter(username='final_test_user').delete()

try:
    serializer = UserProfileCreateSerializer(data=test_data)
    if serializer.is_valid():
        user = serializer.save()
        print(f"   ✓ User created: {user.username}")
        print(f"   ✓ Access type: {user.access_type}")
        
        # Verify
        db_user = UserProfile.objects.get(id=user.id)
        print(f"   ✓ Verified in DB: {db_user.access_type}")
        
        user.delete()
        print("   ✓ Test user deleted")
    else:
        print(f"   ✗ Validation failed: {serializer.errors}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test without access_type
print("\n[3] Testing without access_type (should default to TRIAL)...")
min_data = {
    'username': 'min_test_user',
    'password': 'testpass123',
    'email': 'min@test.com',
    'is_active': True
}

UserProfile.objects.filter(username='min_test_user').delete()

try:
    serializer2 = UserProfileCreateSerializer(data=min_data)
    if serializer2.is_valid():
        user2 = serializer2.save()
        print(f"   ✓ User created: {user2.username}")
        print(f"   ✓ Access type (default): {user2.access_type}")
        user2.delete()
        print("   ✓ Test user deleted")
    else:
        print(f"   ✗ Validation failed: {serializer2.errors}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("FIX COMPLETE!")
print("=" * 70)
print("\nIf tests passed, start your server:")
print("  python manage.py runserver")
print("\nThen try creating a user from the admin dashboard.")
