# Manual Fix Instructions for 500 Errors

## The Problem
Your database has the old `google_id` column but the code expects `username` and `password` columns.

## Quick Fix (3 Steps)

### Step 1: Stop Your Django Server
Press `Ctrl+C` in the terminal where `python manage.py runserver` is running.

### Step 2: Run the Fix Script

**Option A - Double-click:**
- Go to `e:\ai-interview-buddy-main\backend\`
- Double-click `FIX_NOW.py`
- Wait for it to finish (it will say "Press Enter to exit")

**Option B - Command Line:**
```bash
cd e:\ai-interview-buddy-main\backend
python FIX_NOW.py
```

**Option C - Simple Script:**
```bash
cd e:\ai-interview-buddy-main\backend
python fix_db_simple.py
```

### Step 3: Restart Your Server
```bash
python manage.py runserver
```

## Verify It Worked

1. **Test the API:**
   - Open: `http://localhost:8000/api/users/`
   - Should return `[]` (empty array), NOT a 500 error

2. **Create a Test User:**
   - Go to: `http://localhost:8000/admin/`
   - Click "User Profiles" → "Add User Profile"
   - Fill in:
     - Username: `testuser`
     - Password: `testpass123`
     - Email: `test@example.com`
     - Name: `Test User`
     - Is Active: ✓
   - Click "Save"

3. **Test Login:**
   - Go to your frontend login page
   - Use username: `testuser` and password: `testpass123`
   - Should login successfully!

## If Scripts Don't Work

**Nuclear Option - Delete and Recreate Database:**

```bash
cd e:\ai-interview-buddy-main\backend

# Delete database
del db.sqlite3

# Recreate everything
python manage.py migrate

# Create admin user (if needed)
python manage.py createsuperuser

# Seed data (if you have topics/questions)
python manage.py seed_data

# Start server
python manage.py runserver
```

This will delete all existing data but will definitely fix the schema issue.
