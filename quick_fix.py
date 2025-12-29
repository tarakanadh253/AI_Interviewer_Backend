import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("Checking database schema...")
cursor = connection.cursor()
cursor.execute("PRAGMA table_info(user_profiles)")
cols = [c[1] for c in cursor.fetchall()]

print(f"Current columns: {', '.join(cols)}")

if 'access_type' not in cols:
    print("✗ access_type column is MISSING!")
    print("Adding access_type column...")
    try:
        # Add column with default value
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10) DEFAULT 'TRIAL';")
        print("✓ Column added successfully!")
    except Exception as e:
        print(f"✗ Error adding column: {e}")
        # Try without default
        try:
            cursor.execute("ALTER TABLE user_profiles ADD COLUMN access_type VARCHAR(10);")
            cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
            print("✓ Column added (without default, values updated)")
        except Exception as e2:
            print(f"✗ Error on retry: {e2}")
            sys.exit(1)
else:
    print("✓ access_type column exists!")

# Fix any NULL values
print("Fixing NULL access_type values...")
cursor.execute("SELECT COUNT(*) FROM user_profiles WHERE access_type IS NULL;")
null_count = cursor.fetchone()[0]
if null_count > 0:
    print(f"Found {null_count} users with NULL access_type, fixing...")
    cursor.execute("UPDATE user_profiles SET access_type = 'TRIAL' WHERE access_type IS NULL;")
    print("✓ NULL values fixed!")
else:
    print("✓ No NULL values found")

# Verify
cursor.execute("PRAGMA table_info(user_profiles)")
cols = [c[1] for c in cursor.fetchall()]
print(f"\nFinal columns: {', '.join(cols)}")
print("✓ Database is ready!")
