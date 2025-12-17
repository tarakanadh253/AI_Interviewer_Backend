# Deploy Supabase Transcription Function

## Current Status

✅ Supabase credentials configured in `.env`
❌ Transcription function not deployed yet
❌ OpenAI API key not set in Supabase

## Step-by-Step Deployment

### Prerequisites

1. **Supabase CLI** - Install if you don't have it:
   ```bash
   npm install -g supabase
   ```

2. **OpenAI API Key** - You'll need this for transcription:
   - Get one from: https://platform.openai.com/api-keys
   - Keep it secure!

### Deployment Steps

1. **Login to Supabase CLI**:
   ```bash
   supabase login
   ```
   This will open your browser to authenticate.

2. **Link Your Project**:
   ```bash
   cd frontend
   supabase link --project-ref msgbiyznjoioqkpgycnp
   ```
   Your project ref is: `msgbiyznjoioqkpgycnp`

3. **Set OpenAI API Key Secret**:
   ```bash
   supabase secrets set OPENAI_API_KEY=your_openai_api_key_here
   ```
   Replace `your_openai_api_key_here` with your actual OpenAI API key.

4. **Deploy the Function**:
   ```bash
   supabase functions deploy transcribe-audio
   ```

5. **Verify Deployment**:
   - Go to your Supabase dashboard: https://supabase.com/dashboard/project/msgbiyznjoioqkpgycnp
   - Navigate to **Edge Functions** section
   - You should see `transcribe-audio` function listed

## Alternative: Use Browser Speech Recognition (No Supabase Needed)

If you don't want to use Supabase/OpenAI, the app already has browser-based speech recognition built in! It works automatically when you start recording.

## Testing

After deployment, test the function:
1. Start the frontend: `pnpm dev`
2. Go to interview page
3. Click record and speak
4. The transcription should work via Supabase function

## Troubleshooting

- **Function not found**: Make sure you're in the `frontend` directory
- **Permission denied**: Check you're logged in with `supabase login`
- **OpenAI errors**: Verify your API key is correct and has credits
- **CORS errors**: The function already has CORS headers configured

