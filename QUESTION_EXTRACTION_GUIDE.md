# Question Extraction from External Links - Guide

## Overview

When you create a question with `source_type='LINK'` and provide `reference_links`, the system will automatically extract questions from those URLs and create individual Question objects for use in interviews.

## How It Works

1. **Create a LINK-type question**: In the Admin Dashboard, create a new question and:
   - Set `source_type` to "Use External Links"
   - Enter one or more URLs in the `reference_links` field (one URL per line)
   - Select the topic and difficulty
   - Save the question

2. **Automatic extraction**: When you save the LINK-type question:
   - The system fetches content from each URL
   - Extracts questions and answers using pattern matching
   - Creates individual Question objects with `source_type='MANUAL'`
   - Each extracted question uses the same topic and difficulty as the LINK question

3. **Questions appear in interviews**: 
   - Only MANUAL-type questions (including extracted ones) appear in interviews
   - LINK-type questions are placeholders and are hidden from interviews
   - Questions are automatically shuffled when selected for interviews

## Manual Extraction

If you have existing LINK-type questions that haven't been processed:

```powershell
cd E:\ai-interview-buddy-main\backend
python extract_existing_link_questions.py
```

Or use the batch file:
```powershell
.\EXTRACT_QUESTIONS_FROM_LINKS.bat
```

## API Endpoint

You can also manually trigger extraction for a specific LINK-type question:

```bash
POST /api/admin/questions/{question_id}/extract-from-links/
```

## Troubleshooting

### Questions not appearing?

1. **Check if extraction happened**: Look in the Django server logs for extraction messages
2. **Check the database**: Verify that MANUAL-type questions exist for your topic
3. **Manual extraction**: Run `extract_existing_link_questions.py` to process existing LINK questions

### Extraction not working?

1. **Check URLs**: Make sure the URLs are accessible and don't require authentication
2. **Check logs**: Look for errors in the Django server console
3. **Website structure**: Some websites may block automated access or have complex HTML that's hard to parse
4. **Package installation**: Make sure `requests` and `beautifulsoup4` are installed:
   ```bash
   pip install requests beautifulsoup4
   ```

### How extraction works

The extractor looks for:
- Pattern: `Q: ... A: ...` or `Question: ... Answer: ...`
- Numbered lists: `1. Question 2. Answer`
- Questions ending with `?`

The extraction quality depends on the website's HTML structure. Some sites may not work well.

## Viewing Extracted Questions

- **In Admin Dashboard**: Go to Questions tab - you'll see both LINK (placeholders) and MANUAL (extracted) questions
- **In Interviews**: Only MANUAL questions are shown (extracted questions will appear here)
