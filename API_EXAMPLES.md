# API Usage Examples

Complete examples for using the AI Interview Backend API.

## 🔐 Authentication

All public endpoints use `AllowAny` permission. Admin endpoints require Django superuser authentication.

## 📋 Step-by-Step Interview Flow

### 1. Create or Get User Profile

```bash
POST /api/users/get-or-create/
Content-Type: application/json

{
  "google_id": "12345678901234567890",
  "email": "user@example.com",
  "name": "John Doe"
}
```

**Response:**
```json
{
  "id": 1,
  "google_id": "12345678901234567890",
  "email": "user@example.com",
  "name": "John Doe",
  "has_used_trial": false,
  "created_at": "2025-11-29T22:00:00Z",
  "updated_at": "2025-11-29T22:00:00Z"
}
```

### 2. Check Trial Eligibility

```bash
GET /api/users/12345678901234567890/check-trial/
```

**Response:**
```json
{
  "has_used_trial": false,
  "can_start_interview": true
}
```

### 3. Get Available Topics

```bash
GET /api/topics/
```

**Response:**
```json
[
  {
    "id": 1,
    "name": "Python",
    "description": "Python programming language",
    "question_count": 3,
    "created_at": "2025-11-29T22:00:00Z",
    "updated_at": "2025-11-29T22:00:00Z"
  },
  {
    "id": 2,
    "name": "SQL",
    "description": "Structured Query Language",
    "question_count": 2,
    "created_at": "2025-11-29T22:00:00Z",
    "updated_at": "2025-11-29T22:00:00Z"
  }
]
```

### 4. Get Questions for Selected Topics

```bash
GET /api/questions/?topic_id=1
# or
GET /api/questions/?topic_id=1&difficulty=MEDIUM
```

**Response:**
```json
[
  {
    "id": 1,
    "topic": 1,
    "topic_name": "Python",
    "question_text": "What is a list comprehension in Python?",
    "ideal_answer": "A list comprehension is a concise way...",
    "difficulty": "EASY",
    "is_active": true,
    "created_at": "2025-11-29T22:00:00Z",
    "updated_at": "2025-11-29T22:00:00Z"
  }
]
```

### 5. Create Interview Session

```bash
POST /api/sessions/
Content-Type: application/json

{
  "google_id": "12345678901234567890",
  "topic_ids": [1, 2, 3]
}
```

**Response:**
```json
{
  "id": 1,
  "user": 1,
  "user_email": "user@example.com",
  "user_name": "John Doe",
  "started_at": "2025-11-29T22:05:00Z",
  "ended_at": null,
  "duration_seconds": null,
  "topics": [1, 2, 3],
  "topics_list": [
    {"id": 1, "name": "Python"},
    {"id": 2, "name": "SQL"},
    {"id": 3, "name": "DSA"}
  ],
  "status": "IN_PROGRESS",
  "communication_score": null,
  "technology_score": null,
  "result_summary": null,
  "answers": [],
  "answer_count": 0,
  "created_at": "2025-11-29T22:05:00Z",
  "updated_at": "2025-11-29T22:05:00Z"
}
```

**Note:** This will mark `has_used_trial = true` for the user.

### 6. Submit Answers

For each question, submit an answer:

```bash
POST /api/answers/
Content-Type: application/json

{
  "session": 1,
  "question": 1,
  "user_answer": "A list comprehension is a Python feature that allows you to create lists in a concise way. It's written with square brackets and contains an expression followed by a for clause. For example, [x*2 for x in range(10)] creates a list of even numbers. It's more readable and efficient than using a regular for loop."
}
```

**Response:**
```json
{
  "id": 1,
  "session": 1,
  "question": 1,
  "question_id": 1,
  "question_text": "What is a list comprehension in Python?",
  "user_answer": "A list comprehension is a Python feature...",
  "similarity_score": 0.75,
  "matched_keywords": "[\"list\", \"comprehension\", \"python\", \"create\", \"concise\", \"expression\", \"for\", \"clause\", \"example\", \"readable\", \"efficient\", \"loop\"]",
  "missing_keywords": "[\"brackets\", \"zero\", \"clauses\", \"squares\"]",
  "matched_keywords_list": ["list", "comprehension", "python", "create", "concise"],
  "missing_keywords_list": ["brackets", "zero", "clauses"],
  "topic_score": 0.75,
  "communication_subscore": 0.82,
  "created_at": "2025-11-29T22:10:00Z"
}
```

### 7. Complete Interview Session

After submitting all answers:

```bash
POST /api/sessions/1/complete/
```

**Response:**
```json
{
  "id": 1,
  "user": 1,
  "user_email": "user@example.com",
  "user_name": "John Doe",
  "started_at": "2025-11-29T22:05:00Z",
  "ended_at": "2025-11-29T22:15:00Z",
  "duration_seconds": 600,
  "topics": [1, 2, 3],
  "topics_list": [
    {"id": 1, "name": "Python"},
    {"id": 2, "name": "SQL"},
    {"id": 3, "name": "DSA"}
  ],
  "status": "COMPLETED",
  "communication_score": 0.78,
  "technology_score": 0.72,
  "result_summary": "Communication: 78% | Technical Knowledge: 72% | Improvements: Focus on SQL fundamentals",
  "answers": [
    {
      "id": 1,
      "session": 1,
      "question": 1,
      "question_id": 1,
      "question_text": "What is a list comprehension in Python?",
      "user_answer": "...",
      "similarity_score": 0.75,
      "topic_score": 0.75,
      "communication_subscore": 0.82,
      "created_at": "2025-11-29T22:10:00Z"
    }
  ],
  "answer_count": 5,
  "created_at": "2025-11-29T22:05:00Z",
  "updated_at": "2025-11-29T22:15:00Z"
}
```

### 8. Get Detailed Results

```bash
GET /api/sessions/1/results/
```

Returns the same as step 7, but ensures the session is completed.

## 🔧 Error Responses

### Trial Already Used

```bash
POST /api/sessions/
{
  "google_id": "12345678901234567890",
  "topic_ids": [1, 2]
}
```

**Response (403):**
```json
{
  "error": "You have already used your trial interview"
}
```

### Active Session Exists

**Response (400):**
```json
{
  "error": "You have an active interview session. Please complete it first."
}
```

### User Not Found

**Response (404):**
```json
{
  "error": "User not found. Please create user profile first"
}
```

## 👨‍💼 Admin Endpoints

### Get Statistics

```bash
GET /api/admin/sessions/stats/
# Requires: Admin authentication
```

**Response:**
```json
{
  "total_users": 50,
  "total_sessions": 45,
  "completed_sessions": 40,
  "total_questions": 9,
  "total_topics": 5,
  "avg_communication_score": 0.75,
  "avg_technology_score": 0.68
}
```

### List All Sessions (Admin)

```bash
GET /api/admin/sessions/
GET /api/admin/sessions/?status=COMPLETED
GET /api/admin/sessions/?user_email=user@example.com
```

### Manage Questions (Admin)

```bash
# Create Question
POST /api/admin/questions/
{
  "topic": 1,
  "question_text": "What is a decorator in Python?",
  "ideal_answer": "A decorator is a design pattern...",
  "difficulty": "MEDIUM",
  "is_active": true
}

# Update Question
PUT /api/admin/questions/10/
{
  "is_active": false
}
```

## 🧪 Testing with cURL

```bash
# Get topics
curl http://localhost:8000/api/topics/

# Create user
curl -X POST http://localhost:8000/api/users/get-or-create/ \
  -H "Content-Type: application/json" \
  -d '{"google_id":"test123","email":"test@example.com","name":"Test"}'

# Create session
curl -X POST http://localhost:8000/api/sessions/ \
  -H "Content-Type: application/json" \
  -d '{"google_id":"test123","topic_ids":[1,2]}'

# Submit answer
curl -X POST http://localhost:8000/api/answers/ \
  -H "Content-Type: application/json" \
  -d '{"session":1,"question":1,"user_answer":"My answer here"}'

# Complete session
curl -X POST http://localhost:8000/api/sessions/1/complete/
```

## 📊 Score Calculation Details

- **Similarity Score (0-1)**: Percentage of ideal answer keywords found in user answer
- **Communication Score (0-1)**: Based on:
  - Answer length (optimal: 20-100 words)
  - Punctuation usage
  - Sentence structure
  - Filler word penalty ("um", "uh", etc.)
- **Technology Score**: Average of all similarity scores
- **Topic Score**: Same as similarity score for that question

