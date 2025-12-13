# Fix 500 Internal Server Error

## Problem
The backend is returning 500 errors because the database schema doesn't match the code. The database still has `google_id` but the code expects `username` and `password`.

## Solution

### Option 1: Run Migration (Recommended)

1. **Stop the backend server** if it's running (Ctrl+C)

2. **Run the migration**:
   ```bash
   cd e:\ai-interview-buddy-main\backend
   python manage.py migrate interview
   ```

3. **If migration fails**, you may need to reset the database:
   ```bash
   # Delete the database file (SQLite)
   del db.sqlite3
   
   # Or if using a different database, drop and recreate
   
   # Then run migrations from scratch
   python manage.py migrate
   ```

4. **Restart the server**:
   ```bash
   python manage.py runserver
   ```

### Option 2: Manual Database Fix (SQLite)

If you're using SQLite and migrations don't work:

1. **Backup your database** (if you have important data):
   ```bash
   copy db.sqlite3 db.sqlite3.backup
   ```

2. **Delete the database**:
   ```bash
   del db.sqlite3
   ```

3. **Recreate everything**:
   ```bash
   python manage.py migrate
   python manage.py seed_data  # If you have seed data
   ```

### Option 3: Use the Fix Script

Run the fix script:
```bash
cd e:\ai-interview-buddy-main\backend
python fix_migration.py
```

## After Fixing

1. **Create a test user** in Django Admin:
   - Go to `http://localhost:8000/admin/`
   - Navigate to "User Profiles"
   - Click "Add User Profile"
   - Fill in:
     - Username: `testuser`
     - Password: `testpass123`
     - Email: `test@example.com`
     - Name: `Test User`
     - Is Active: ✓
   - Save

2. **Test login** in the frontend with the credentials above

## Verify It's Fixed

Check the API endpoint:
```bash
curl http://localhost:8000/api/users/
```

Should return an empty array `[]` or a list of users, not a 500 error.
