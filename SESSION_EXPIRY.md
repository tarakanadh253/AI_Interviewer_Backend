# Session Auto-Expiry Feature

## Overview

Interview sessions now automatically expire after **30 minutes** of inactivity. This ensures that if a user closes their browser without completing an interview, they won't be blocked from starting a new session indefinitely.

## How It Works

### 1. Session Expiry Time
- Sessions expire **30 minutes** after they are created (`started_at` timestamp)
- This applies to sessions with status `CREATED` or `IN_PROGRESS`
- Expired sessions are automatically marked as `CANCELLED`

### 2. Automatic Expiry Checks

Sessions are checked and expired automatically in the following scenarios:

#### When Creating a New Session
- Before creating a new session, the system checks for existing active sessions
- If an active session exists and is older than 30 minutes, it's automatically expired
- If an active session exists and is less than 30 minutes old, the user gets an error message

#### When Retrieving a Session
- When a session is retrieved (e.g., loading the interview page), it's checked for expiry
- If expired, the session is marked as `CANCELLED` before being returned

#### When Listing Sessions
- When listing sessions (e.g., admin dashboard), active sessions are checked for expiry
- Expired sessions are automatically marked as `CANCELLED`

### 3. Model Methods

The `InterviewSession` model includes two new methods:

#### `is_expired(timeout_minutes=30)`
Checks if a session has exceeded the timeout period.

```python
if session.is_expired():
    # Session is older than 30 minutes
```

#### `auto_expire_if_needed(timeout_minutes=30)`
Automatically expires a session if it has exceeded the timeout. Returns `True` if the session was expired, `False` otherwise.

```python
if session.auto_expire_if_needed():
    # Session was just expired
```

## User Experience

### Frontend Behavior

1. **Starting a New Interview**
   - If user has an active session less than 30 minutes old:
     - Error: "You have an active interview session. Please complete it or wait 30 minutes for it to expire automatically."
   - If user has an active session older than 30 minutes:
     - Session is automatically expired
     - New session can be created

2. **Loading an Expired Session**
   - If user tries to load an expired session:
     - Session is marked as `CANCELLED`
     - User is redirected to topic selection
     - Message: "Your interview session has expired (30 minutes). Please start a new interview."

## Technical Details

### Backend Changes

**File: `backend/interview/models.py`**
- Added `is_expired()` method
- Added `auto_expire_if_needed()` method

**File: `backend/interview/views.py`**
- Updated `create()` method to check and expire old sessions
- Updated `get_queryset()` to auto-expire sessions when listing
- Added `retrieve()` override to check expiry when fetching a session

### Frontend Changes

**File: `frontend/src/pages/Interview.tsx`**
- Added check for `CANCELLED` status when loading session
- Redirects to topic selection if session is expired

**File: `frontend/src/pages/TopicSelection.tsx`**
- Updated error message to mention 30-minute auto-expiry

## Configuration

The timeout can be adjusted by changing the `timeout_minutes` parameter:

```python
# Default: 30 minutes
session.auto_expire_if_needed(timeout_minutes=30)

# Custom: 60 minutes
session.auto_expire_if_needed(timeout_minutes=60)
```

## Testing

To test the expiry feature:

1. **Create a session** and note the `started_at` timestamp
2. **Manually set `started_at`** to 31 minutes ago:
   ```python
   from django.utils import timezone
   from datetime import timedelta
   session.started_at = timezone.now() - timedelta(minutes=31)
   session.save()
   ```
3. **Try to create a new session** - the old one should be auto-expired
4. **Try to load the old session** - it should be marked as `CANCELLED`

## Benefits

1. **Prevents Blocking**: Users aren't permanently blocked by abandoned sessions
2. **Automatic Cleanup**: No manual intervention needed
3. **Fair Time Limit**: 30 minutes is reasonable for an interview session
4. **Seamless Experience**: Expiry happens automatically without user action

## Migration Notes

- No database migrations required
- Existing sessions will be checked for expiry on next access
- Sessions older than 30 minutes will be automatically expired when accessed

