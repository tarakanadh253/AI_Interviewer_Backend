# Quick Fix for "Subjects Not Showing"

## The Problem
The frontend can't fetch topics because either:
1. Backend server is not running
2. Topics are not seeded in the database

## Solution (Step by Step)

### Step 1: Open a NEW terminal/PowerShell window

### Step 2: Navigate to backend folder
```powershell
cd e:\ai-interview-buddy-main\backend
```

### Step 3: Activate virtual environment (if you have one)
```powershell
venv\Scripts\Activate.ps1
```

### Step 4: Run migrations
```powershell
python manage.py migrate
```

### Step 5: Seed topics
```powershell
python manage.py seed_data
```

You should see output like:
```
Seeding initial data...
Created topic: Python
Created topic: SQL
...
```

### Step 6: Start the backend server
```powershell
python manage.py runserver
```

You should see:
```
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

### Step 7: Test the API
Open your browser and go to:
```
http://localhost:8000/api/topics/
```

You should see JSON with topics like:
```json
[
  {
    "id": 1,
    "name": "Python",
    "description": "Python programming language",
    "question_count": 3,
    ...
  },
  ...
]
```

### Step 8: Refresh your frontend
Go back to your frontend app and refresh the page. Topics should now appear!

## Troubleshooting

### If you see "Connection refused" error:
- Make sure the backend server is running (Step 6)
- Check that it's running on port 8000

### If you see empty array []:
- Make sure you ran `python manage.py seed_data` (Step 5)
- Check the database has topics by running:
  ```powershell
  python verify_setup.py
  ```

### If topics still don't show:
1. Check browser console (F12) for errors
2. Verify API URL in frontend `.env` file is: `VITE_API_URL=http://localhost:8000/api`
3. Make sure CORS is configured (should be already set in settings.py)

## Quick Commands Summary
```powershell
cd e:\ai-interview-buddy-main\backend
python manage.py migrate
python manage.py seed_data
python manage.py runserver
```
