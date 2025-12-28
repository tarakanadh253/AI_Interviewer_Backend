# Fix "No Topics Available" Issue

## Quick Diagnosis

### Step 1: Check if Backend Server is Running

Open a browser and go to:
```
http://localhost:8000/api/topics/
```

**Expected:** Should return a JSON array like `[]` or `[{...}, {...}]`
**If you get:** Connection error or page not found → Server is not running

### Step 2: Check if Topics are Seeded

Run this command:
```bash
cd e:\ai-interview-buddy-main\backend
python check_topics.py
```

**Expected:** Should show list of topics
**If you get:** "No topics found" → Need to seed data

## Solutions

### Solution 1: Start Backend Server

```bash
cd e:\ai-interview-buddy-main\backend
python manage.py runserver
```

Keep this terminal open. The server should be running on `http://localhost:8000`

### Solution 2: Seed Topics Data

**In a NEW terminal** (keep server running in the first one):

```bash
cd e:\ai-interview-buddy-main\backend
python manage.py seed_data
```

You should see output like:
```
Seeding initial data...
Created topic: Python
Created topic: SQL
...
Seeding complete! Created X new questions.
```

### Solution 3: Verify Everything Works

1. **Check API directly:**
   - Go to: `http://localhost:8000/api/topics/`
   - Should see JSON with topics

2. **Refresh your frontend:**
   - Go to your frontend app
   - Refresh the page (F5)
   - Topics should appear

## Common Issues

### Issue: "Failed to fetch" error
**Cause:** Backend server is not running
**Fix:** Start server with `python manage.py runserver`

### Issue: API returns `[]` (empty array)
**Cause:** Topics not seeded
**Fix:** Run `python manage.py seed_data`

### Issue: CORS error
**Cause:** Backend CORS settings
**Fix:** Check `settings.py` has `CORS_ALLOWED_ORIGINS` configured

### Issue: Database errors
**Cause:** Database schema issues
**Fix:** Run `python manage.py migrate` first

## Quick Fix Script

Run this to do everything at once:

```bash
cd e:\ai-interview-buddy-main\backend

# Make sure server is stopped first (Ctrl+C)
# Then run migrations
python manage.py migrate

# Seed data
python manage.py seed_data

# Start server
python manage.py runserver
```

Then refresh your frontend!
