# Fix Answers Table

## Issue
The `answers` table was missing from the database, causing a 500 error when submitting answers:
```
Error: Failed to submit answer: no such table: answers
```

## Solution
Created a script `fix_answers_table.py` that:
1. Checks if the `answers` table exists
2. Creates it if missing with all required columns
3. Adds missing columns if table exists but is incomplete
4. Creates necessary indexes

## Table Structure

The `answers` table includes:
- `id` - Primary key
- `session_id` - Foreign key to interview_sessions
- `question_id` - Foreign key to questions
- `user_answer` - Text field for transcribed answer
- `similarity_score` - Semantic similarity (0-1)
- `accuracy_score` - Overall accuracy (0-1)
- `completeness_score` - Completeness score (0-1)
- `matched_keywords` - JSON array of matched keywords
- `missing_keywords` - JSON array of missing keywords
- `topic_score` - Topic score (0-1)
- `communication_subscore` - Communication quality (0-1)
- `score_breakdown` - JSON with detailed breakdown
- `created_at` - Timestamp

## How to Run

If you encounter this error again, run:
```bash
cd backend
python fix_answers_table.py
```

## Status
✅ Table created successfully
✅ All columns added
✅ Indexes created
✅ Ready to accept answer submissions

