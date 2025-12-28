# AI Voice-Based Mock Interview Backend

Django REST Framework backend for the AI Voice-Based Mock Interview module for ohg365.com.

## 🧱 Tech Stack

- **Language**: Python 3.8+
- **Framework**: Django 5.0 + Django REST Framework
- **Database**: SQLite (development), PostgreSQL-compatible models
- **Evaluation**: Pure Python logic (no external AI/LLM APIs)

## 📋 Features

1. **User Management**: Google Sign-In integration with trial enforcement (1 trial per account)
2. **Question Management**: Predefined Q&A created by Admin
3. **Interview Sessions**: Track user interviews with topics and status
4. **Answer Evaluation**: Automatic scoring using keyword matching and communication analysis
5. **Admin APIs**: View results and manage Q&A

## 🚀 Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create superuser** (for admin access):
   ```bash
   python manage.py createsuperuser
   ```

6. **Run development server**:
   ```bash
   python manage.py runserver
   ```

The API will be available at `http://localhost:8000/api/`

## 📚 API Endpoints

### User Endpoints

- `POST /api/users/get-or-create/` - Get or create user profile from Google account
- `GET /api/users/{google_id}/check-trial/` - Check if user has used trial
- `GET /api/users/` - List all users (admin)
- `GET /api/users/{id}/` - Get user details

### Topic Endpoints

- `GET /api/topics/` - List all topics
- `GET /api/topics/{id}/` - Get topic details

### Question Endpoints

- `GET /api/questions/` - List active questions
- `GET /api/questions/?topic_id={id}` - Filter by topic
- `GET /api/questions/?difficulty={EASY|MEDIUM|HARD}` - Filter by difficulty
- `GET /api/questions/{id}/` - Get question details

### Interview Session Endpoints

- `POST /api/sessions/` - Create new interview session (enforces 1 trial per account)
  - Body: `{ "google_id": "...", "topic_ids": [1, 2, 3] }`
- `GET /api/sessions/` - List sessions (filter by `google_id` or `user_id`)
- `GET /api/sessions/{id}/` - Get session details
- `POST /api/sessions/{id}/complete/` - Complete session and calculate scores
- `GET /api/sessions/{id}/results/` - Get detailed results

### Answer Endpoints

- `POST /api/answers/` - Submit answer (auto-evaluates)
  - Body: `{ "session": id, "question": id, "user_answer": "..." }`
- `GET /api/answers/?session_id={id}` - Get answers for a session
- `GET /api/answers/{id}/` - Get answer details

### Admin Endpoints

- `GET /api/admin/questions/` - List all questions (admin only)
- `POST /api/admin/questions/` - Create question (admin only)
- `PUT /api/admin/questions/{id}/` - Update question (admin only)
- `GET /api/admin/sessions/` - List all sessions (admin only)
- `GET /api/admin/sessions/stats/` - Get statistics (admin only)

## 🔐 Authentication

- Public endpoints use `AllowAny` permission
- Admin endpoints require Django superuser authentication
- User identification via Google ID (no JWT/OAuth tokens needed)

## 🧠 Evaluation Logic

The evaluation system uses pure Python logic (no external APIs):

1. **Similarity Score**: Keyword overlap between user answer and ideal answer
2. **Communication Score**: Based on:
   - Answer length (word count)
   - Punctuation usage
   - Sentence structure
   - Filler word detection ("um", "uh", etc.)
3. **Technology Score**: Average similarity scores across all questions

See `interview/utils/evaluation.py` for implementation details.

## 📊 Models

- **UserProfile**: Google account info with trial tracking
- **Topic**: Interview topics (Python, SQL, DSA, etc.)
- **Question**: Questions with ideal answers and difficulty
- **InterviewSession**: User interview sessions with scores
- **Answer**: Individual answers with evaluation results

## 🔧 Configuration

### CORS Settings

CORS is configured for:
- `http://localhost:3000` (frontend dev)
- Add production domain in `settings.py` when ready

### Database

- Development: SQLite (`db.sqlite3`)
- Production: Update `DATABASES` in `settings.py` for PostgreSQL

## 📝 Example API Usage

### 1. Create/Get User
```bash
POST /api/users/get-or-create/
{
  "google_id": "123456789",
  "email": "user@example.com",
  "name": "John Doe"
}
```

### 2. Check Trial Eligibility
```bash
GET /api/users/123456789/check-trial/
```

### 3. Create Interview Session
```bash
POST /api/sessions/
{
  "google_id": "123456789",
  "topic_ids": [1, 2, 3]
}
```

### 4. Submit Answer
```bash
POST /api/answers/
{
  "session": 1,
  "question": 5,
  "user_answer": "Python is a high-level programming language..."
}
```

### 5. Complete Session
```bash
POST /api/sessions/1/complete/
```

### 6. Get Results
```bash
GET /api/sessions/1/results/
```

## 🛠️ Development

### Running Tests
```bash
python manage.py test
```

### Creating Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Accessing Admin Panel
Visit `http://localhost:8000/admin/` and login with superuser credentials.

## 📦 Project Structure

```
backend/
├── ohg365_ai_interviewer/    # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── interview/                 # Main app
│   ├── models.py              # Database models
│   ├── serializers.py        # DRF serializers
│   ├── views.py              # API views
│   ├── urls.py               # URL routing
│   ├── admin.py              # Admin configuration
│   └── utils/
│       └── evaluation.py     # Evaluation logic
├── manage.py
├── requirements.txt
└── README.md
```

## 🚨 Important Notes

1. **Trial Enforcement**: Only 1 interview session per Google account is allowed
2. **No External AI**: All evaluation is done with pure Python (keyword matching, heuristics)
3. **SQLite for Dev**: Easy to run, but models are PostgreSQL-compatible
4. **CORS**: Configured for localhost:3000, update for production

## 📄 License

Part of the ohg365.com project.

