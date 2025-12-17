# Environment Variables Setup

## Required for Supabase (Optional)

Create a `.env` file in the `frontend` folder with:

```env
# Supabase Configuration
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-anon-key-here

# Django Backend API URL (optional, defaults to http://localhost:8000/api)
VITE_API_URL=http://localhost:8000/api
```

## How to Get Supabase Credentials

1. Go to [https://supabase.com](https://supabase.com)
2. Create a new project or use existing one
3. Go to **Settings** → **API**
4. Copy:
   - **Project URL** → `VITE_SUPABASE_URL`
   - **anon/public key** → `VITE_SUPABASE_PUBLISHABLE_KEY`

## Note

If you don't set up Supabase, the app will still work but audio transcription will be disabled. Users can manually type their answers instead.

