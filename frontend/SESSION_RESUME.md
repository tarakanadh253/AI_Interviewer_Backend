# Session Resume Feature

## Overview

Users can now **resume their interview session** if they leave without clicking "End" and return within 30 minutes. The session continues from where they left off with the same remaining time.

## How It Works

### 1. Active Session Detection

When a user visits the Topic Selection page:
- System checks for active sessions (CREATED or IN_PROGRESS status)
- Validates session is within 30-minute window
- Shows resume dialog if active session found

### 2. Resume Dialog

If an active session is found, a dialog appears showing:
- **Session start time**: When the interview began
- **Time remaining**: Calculated from start time (30 minutes total)
- **Status**: Current session status
- **Options**: 
  - "Resume Interview" - Continue existing session
  - "Start New" - Cancel current and start fresh

### 3. Resuming a Session

When user clicks "Resume Interview":
- Session ID is stored in localStorage
- User is redirected to interview page
- Interview loads with:
  - **Same questions** (using seeded shuffle based on session ID)
  - **Same answers** (previously submitted answers are loaded)
  - **Correct remaining time** (calculated from session start)
  - **Current question** (first unanswered question, or last if all answered)

### 4. Time Calculation

The timer is calculated dynamically:
- **Total time**: 30 minutes (1800 seconds)
- **Elapsed time**: Current time - Session start time
- **Remaining time**: Total time - Elapsed time
- **Updates**: Every second to ensure accuracy

### 5. Question Consistency

Questions are loaded consistently using:
- **Seeded shuffle**: Session ID is used as seed for random shuffle
- **Same order**: Same session ID = same question order
- **Same questions**: First 10 questions from shuffled list

## Features

✅ **Automatic Detection**: Checks for active sessions on page load
✅ **Time Preservation**: Remaining time calculated from start time
✅ **Progress Preservation**: Answers and question progress maintained
✅ **Question Consistency**: Same questions appear in same order
✅ **Smart Navigation**: Jumps to first unanswered question
✅ **30-Minute Window**: Only valid sessions within 30 minutes can be resumed

## User Flow

### Scenario 1: User Leaves and Returns

1. User starts interview → Session created
2. User answers 3 questions
3. User closes browser/tab (without clicking "End")
4. User returns within 30 minutes
5. **Resume dialog appears** on Topic Selection page
6. User clicks "Resume Interview"
7. Interview loads:
   - Same 10 questions
   - 3 answers already submitted
   - Remaining time: ~27 minutes (or whatever is left)
   - Current question: Question 4 (first unanswered)

### Scenario 2: Session Expired

1. User starts interview → Session created
2. User leaves
3. User returns after 35 minutes
4. **No resume dialog** (session auto-expired)
5. User can start new interview

### Scenario 3: User Completes Interview

1. User completes all questions
2. User clicks "Complete Interview"
3. Session marked as COMPLETED
4. **No resume option** (session finished)
5. User can start new interview

## Technical Details

### Frontend Changes

**TopicSelection.tsx**:
- Added `activeSession` state
- Added `showResumeDialog` state
- Checks for active sessions on mount
- Shows resume dialog with session info
- Handles resume and start new actions

**Interview.tsx**:
- Timer recalculates from session start time
- Questions use seeded shuffle (session ID as seed)
- Current question index set to first unanswered
- Answers loaded from session data

### Backend

No changes needed - existing endpoints support this:
- `GET /api/sessions/?username={username}` - Get user sessions
- `GET /api/sessions/{id}/` - Get session details
- Session expiry logic already handles 30-minute timeout

## Code Highlights

### Seeded Shuffle

```typescript
const seededShuffle = (array: Question[], seed: number) => {
  const shuffled = [...array];
  let currentSeed = seed;
  const random = () => {
    currentSeed = (currentSeed * 9301 + 49297) % 233280;
    return currentSeed / 233280;
  };
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};
```

### Time Calculation

```typescript
const startTime = new Date(session.started_at).getTime();
const now = Date.now();
const elapsedSeconds = Math.floor((now - startTime) / 1000);
const totalSeconds = 30 * 60; // 30 minutes
const remaining = Math.max(0, totalSeconds - elapsedSeconds);
```

### Current Question Detection

```typescript
const firstUnansweredIndex = selectedQuestions.findIndex(
  q => !answeredQuestionIds.has(q.id)
);
if (firstUnansweredIndex !== -1) {
  setCurrentQuestionIndex(firstUnansweredIndex);
}
```

## Benefits

1. **User-Friendly**: No lost progress if browser closes accidentally
2. **Time Accurate**: Timer reflects actual elapsed time
3. **Consistent**: Same questions and order when resuming
4. **Smart**: Automatically finds where user left off
5. **Flexible**: User can choose to resume or start new

## Testing

To test the resume feature:

1. Start an interview
2. Answer a few questions
3. Close browser/tab (don't click "End")
4. Wait a few minutes (but less than 30)
5. Reopen browser and go to Topic Selection
6. Resume dialog should appear
7. Click "Resume Interview"
8. Verify:
   - Same questions appear
   - Previous answers are there
   - Timer shows correct remaining time
   - Current question is first unanswered

## Summary

The session resume feature allows users to seamlessly continue their interview if they leave without completing it, as long as they return within 30 minutes. All progress, time, and questions are preserved exactly as they were.

