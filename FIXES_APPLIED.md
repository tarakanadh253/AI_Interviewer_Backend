# Fixes Applied

## Issue 1: Active Session Blocking New Sessions ✅

### Problem
Old interview sessions that were never completed were blocking new session creation.

### Solution
1. **Auto-cancel old sessions**: Sessions older than 24 hours are automatically cancelled when trying to create a new session.
2. **Manual cancel endpoint**: Added `/api/sessions/cancel-active/` endpoint to manually cancel active sessions.

### Usage

**Auto-cancel**: Happens automatically when you try to start a new interview.

**Manual cancel** (if needed):
```bash
POST /api/sessions/cancel-active/
{
  "username": "your_username"
}
```

## Issue 2: Question Extraction from External Links ✅

### Problem
Questions with `source_type='LINK'` were not extracting questions from the provided URLs.

### Solution
1. Created question extraction utility (`interview/utils/question_extractor.py`)
2. Questions are automatically extracted when a LINK-type question is created
3. Added manual extraction endpoint for existing LINK-type questions

### Installation

Install required packages:
```bash
pip install requests beautifulsoup4
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Usage

**Automatic extraction**: When you create a new question with `source_type='LINK'` and provide `reference_links`, questions are automatically extracted.

**Manual extraction** (for existing LINK-type questions):
```bash
POST /api/admin/questions/{question_id}/extract-from-links/
```

This will:
- Extract questions and answers from all URLs in `reference_links`
- Create new Question objects with `source_type='MANUAL'` for each extracted Q&A
- Use the same topic and difficulty as the original LINK question

### How It Works

The extractor:
1. Fetches content from each URL
2. Parses HTML content
3. Looks for Q&A patterns (Q: ... A: ..., numbered lists, question marks, etc.)
4. Creates individual Question objects for each extracted Q&A pair
5. Stores them in the database for use in interviews

### Notes

- Extraction may not work perfectly for all websites (depends on HTML structure)
- Some websites may block automated access
- Extracted questions are stored as separate Question objects, so they can be edited individually
- The original LINK-type question remains as a placeholder

## Next Steps

1. **Restart your Django server** to apply the changes
2. **Install new packages**: `pip install requests beautifulsoup4`
3. **Try creating a new interview** - old sessions should be auto-cancelled
4. **For link extraction**: Create a LINK-type question with URLs, or use the manual extraction endpoint
