# Supabase Setup Guide

## What is Supabase Used For?

In this project, Supabase is used for **audio transcription** (speech-to-text) in the Interview page. The frontend records audio and sends it to Supabase Edge Functions for transcription.

## Option 1: Set Up Supabase (Recommended for Production)

### Step 1: Create a Supabase Project

1. Go to [https://supabase.com](https://supabase.com)
2. Sign up or log in
3. Click "New Project"
4. Fill in:
   - Project name: `ai-interview-buddy` (or your choice)
   - Database password: (save this securely)
   - Region: Choose closest to you
5. Click "Create new project"

### Step 2: Get Your Credentials

1. In your Supabase project dashboard, go to **Settings** → **API**
2. Copy:
   - **Project URL** → This is your `VITE_SUPABASE_URL`
   - **anon/public key** → This is your `VITE_SUPABASE_PUBLISHABLE_KEY`

### Step 3: Create Environment File

1. In the `frontend` folder, create a `.env` file:
   ```bash
   cd frontend
   cp .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```env
   VITE_SUPABASE_URL=https://your-project-id.supabase.co
   VITE_SUPABASE_PUBLISHABLE_KEY=your-anon-key-here
   ```

### Step 4: Deploy Transcription Function

The transcription function is in `frontend/supabase/functions/transcribe-audio/`. You'll need to:

1. Install Supabase CLI:
   ```bash
   npm install -g supabase
   ```

2. Login to Supabase:
   ```bash
   supabase login
   ```

3. Link your project:
   ```bash
   supabase link --project-ref your-project-id
   ```

4. Deploy the function:
   ```bash
   supabase functions deploy transcribe-audio
   ```

### Step 5: Restart Frontend

After setting up `.env`, restart your frontend dev server:
```bash
pnpm dev
```

## Option 2: Use Django Backend for Transcription

If you prefer not to use Supabase, you can:

1. **Create a Django endpoint** for audio transcription
2. **Use a transcription service** like:
   - Google Speech-to-Text API
   - Azure Speech Services
   - AssemblyAI
   - Or any other transcription service

3. **Update the frontend** to call your Django endpoint instead of Supabase

## Option 3: Use Browser's Built-in Speech Recognition (Fallback)

The code has been updated to handle missing Supabase gracefully. You can also implement browser's Web Speech API as a fallback:

```typescript
// Example using Web Speech API
const recognition = new (window as any).webkitSpeechRecognition();
recognition.continuous = true;
recognition.interimResults = true;

recognition.onresult = (event: any) => {
  const transcript = Array.from(event.results)
    .map((result: any) => result[0].transcript)
    .join('');
  setTranscript(transcript);
};

recognition.start();
```

**Note:** Web Speech API has limited browser support and may not be as accurate as cloud-based solutions.

## Current Status

The application will work without Supabase, but audio transcription will be disabled. Users can still:
- View questions
- Manually type answers
- Complete interviews

To enable full audio transcription functionality, set up Supabase as described above.

## Troubleshooting

### Error: "supabaseUrl is required"
- Make sure `.env` file exists in the `frontend` folder
- Check that `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY` are set
- Restart the dev server after creating/updating `.env`

### Transcription not working
- Check Supabase function is deployed
- Verify your Supabase project is active
- Check browser console for errors
- Ensure microphone permissions are granted

