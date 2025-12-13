# Database Fields for User Answers

## Answer Model Structure

The `Answer` model in `backend/interview/models.py` has the following fields to store user answers and evaluation results:

### Core Fields

1. **`user_answer`** (TextField)
   - **Purpose**: Stores the user's transcribed voice answer text
   - **Type**: TextField (unlimited length)
   - **Required**: Yes
   - **Example**: "Python is a high-level programming language known for its simplicity and readability."

2. **`session`** (ForeignKey)
   - **Purpose**: Links the answer to the interview session
   - **Type**: ForeignKey to InterviewSession
   - **Required**: Yes

3. **`question`** (ForeignKey)
   - **Purpose**: Links the answer to the specific question
   - **Type**: ForeignKey to Question
   - **Required**: Yes

### Scoring Fields

4. **`similarity_score`** (FloatField)
   - **Purpose**: Semantic similarity score (0-1)
   - **Type**: FloatField
   - **Default**: 0.0
   - **Description**: Backward compatible score

5. **`accuracy_score`** (FloatField)
   - **Purpose**: Overall accuracy score (0-1)
   - **Type**: FloatField, nullable
   - **Description**: Primary score combining semantic similarity (70%) and completeness (30%)

6. **`completeness_score`** (FloatField)
   - **Purpose**: How complete the answer is (0-1)
   - **Type**: FloatField, nullable
   - **Description**: Measures concept coverage

7. **`communication_subscore`** (FloatField)
   - **Purpose**: Communication quality score (0-1)
   - **Type**: FloatField, nullable
   - **Description**: Evaluates answer structure, vocabulary, etc.

8. **`topic_score`** (FloatField)
   - **Purpose**: Per-question contribution for Technologies score
   - **Type**: FloatField, nullable
   - **Description**: Used for calculating overall technology score

### Keyword Analysis Fields

9. **`matched_keywords`** (TextField)
   - **Purpose**: Keywords found in user answer
   - **Type**: TextField (stores JSON)
   - **Description**: List of keywords that matched the ideal answer

10. **`missing_keywords`** (TextField)
    - **Purpose**: Important keywords missing from user answer
    - **Type**: TextField (stores JSON)
    - **Description**: List of keywords that should have been included

### Additional Fields

11. **`score_breakdown`** (TextField)
    - **Purpose**: Detailed breakdown of all scores
    - **Type**: TextField (stores JSON)
    - **Description**: Contains semantic_similarity, keyword_coverage, completeness, communication_quality, accuracy

12. **`created_at`** (DateTimeField)
    - **Purpose**: Timestamp when answer was created
    - **Type**: DateTimeField
    - **Auto**: Set automatically on creation

## Database Table

The model maps to the `answers` table in the database with:
- **Table name**: `answers`
- **Unique constraint**: One answer per question per session (`unique_together = ['session', 'question']`)
- **Ordering**: By `created_at` (oldest first)

## How Answers Are Stored

When a user submits an answer:

1. **Frontend sends**: `{ "session": id, "question": id, "user_answer": "transcribed text" }`
2. **Backend receives**: Via `AnswerCreateSerializer`
3. **Evaluation happens**: `evaluate_answer()` function processes the answer
4. **Answer created**: All fields populated including scores
5. **Database stores**: Complete answer record with evaluation results

## Example Answer Record

```python
{
    "id": 1,
    "session": 5,
    "question": 12,
    "user_answer": "Python is a programming language that is easy to learn and read.",
    "similarity_score": 0.85,
    "accuracy_score": 0.82,
    "completeness_score": 0.75,
    "communication_subscore": 0.88,
    "topic_score": 0.82,
    "matched_keywords": '["python", "programming", "language", "easy", "learn"]',
    "missing_keywords": '["high-level", "simplicity", "readability"]',
    "score_breakdown": '{"semantic_similarity": 0.85, "keyword_coverage": 0.67, ...}',
    "created_at": "2024-01-15T10:30:00Z"
}
```

## API Endpoint

**POST** `/api/answers/`

**Request Body**:
```json
{
    "session": 5,
    "question": 12,
    "user_answer": "User's transcribed answer text"
}
```

**Response**:
```json
{
    "id": 1,
    "session": 5,
    "question": 12,
    "user_answer": "User's transcribed answer text",
    "similarity_score": 0.85,
    "accuracy_score": 0.82,
    "completeness_score": 0.75,
    "communication_subscore": 0.88,
    "topic_score": 0.82,
    "matched_keywords": ["python", "programming", "language"],
    "missing_keywords": ["high-level", "simplicity"],
    "score_breakdown": {...},
    "created_at": "2024-01-15T10:30:00Z"
}
```

## Summary

✅ **Yes, the database has a `user_answer` field** (TextField) that stores the user's transcribed voice answer text.

The field is:
- Required (cannot be null)
- Unlimited length (TextField)
- Stores the exact text from voice transcription
- Used for evaluation and scoring
- Linked to session and question via foreign keys

