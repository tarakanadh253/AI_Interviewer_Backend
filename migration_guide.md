# Migration Guide: Google Sign-In to Username/Password

## Changes Made

### Backend Changes

1. **UserProfile Model** (`models.py`):
   - Removed: `google_id` field
   - Added: `username` (unique), `password` (hashed), `is_active` fields
   - Added: `set_password()` and `check_password()` methods

2. **API Endpoints** (`views.py`):
   - Removed: `/users/get-or-create/` endpoint
   - Added: `/users/login/` endpoint (POST with username/password)
   - Updated: `/users/{username}/check-trial/` (changed from google_id to username)
   - Updated: Session creation now uses `username` instead of `google_id`

3. **Serializers** (`serializers.py`):
   - Added: `UserProfileCreateSerializer` for admin user creation
   - Added: `UserLoginSerializer` for login
   - Updated: `UserProfileSerializer` to use username

4. **Admin Interface** (`admin.py`):
   - Updated to show username instead of google_id
   - Added password field with automatic hashing
   - Added fieldsets for better organization

### Frontend Changes

1. **Login Page** (`Login.tsx`):
   - Removed Google Sign-In button
   - Added username/password form
   - Updated to use new login API

2. **API Service** (`api.ts`):
   - Removed: `getOrCreateUser()`
   - Added: `login(username, password)`
   - Updated: `createSession()` to use username
   - Updated: `checkTrial()` to use username

3. **TopicSelection** (`TopicSelection.tsx`):
   - Changed from `google_id` to `username` in localStorage

## Database Migration Required

You need to create and run a migration:

```bash
cd backend
python manage.py makemigrations
python manage.py migrate
```

**Important**: This migration will:
- Remove the `google_id` column
- Add `username`, `password`, and `is_active` columns
- **All existing user data will be lost** (since we're changing the authentication method)

## How to Use

### For Admin:

1. **Create User Accounts**:
   - Go to Django Admin: `http://localhost:8000/admin/`
   - Navigate to "User Profiles"
   - Click "Add User Profile"
   - Fill in:
     - Username (unique)
     - Password (will be automatically hashed)
     - Email
     - Name (optional)
     - Is Active (checked by default)
   - Save

2. **Give Credentials to Users**:
   - Share the username and password with the user
   - Users can now login with these credentials

### For Users:

1. **Login**:
   - Go to login page
   - Enter username and password provided by admin
   - Click "Sign In"

2. **If Account Issues**:
   - Contact administrator if credentials don't work
   - Administrator can reset password or activate/deactivate account in admin panel

## Testing

1. **Create a test user in admin**:
   - Username: `testuser`
   - Password: `testpass123`
   - Email: `test@example.com`

2. **Test login**:
   - Use the credentials above
   - Should successfully login and navigate to topic selection

## Notes

- Passwords are automatically hashed using Django's password hashing
- Admin can see username, email, and trial status
- Admin can activate/deactivate accounts
- One trial interview per user account (enforced)
