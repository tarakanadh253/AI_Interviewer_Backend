from django.db import models
from django.utils import timezone
from django.contrib.auth.hashers import make_password, check_password
import json


class UserProfile(models.Model):
    """User profile with username/password authentication"""
    ROLE_CHOICES = [
        ('ADMIN', 'Administrator'),
        ('USER', 'User (Student)'),
    ]

    ACCESS_TYPE_CHOICES = [
        ('TRIAL', 'Trial - One Free Interview'),
        ('FULL', 'Full Access - Unlimited Interviews'),
    ]
    
    username = models.CharField(max_length=150, unique=True, db_index=True)
    password = models.CharField(max_length=128)  # Hashed password
    email = models.EmailField()
    name = models.CharField(max_length=255, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='USER')
    access_type = models.CharField(max_length=10, choices=ACCESS_TYPE_CHOICES, default='TRIAL', null=True, blank=True)
    plain_password = models.CharField(max_length=128, null=True, blank=True, help_text="Stored for admin visibility (Insecure)")
    has_used_trial = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    enrolled_course = models.ForeignKey('Topic', on_delete=models.SET_NULL, null=True, blank=True, related_name='students')
    student_id = models.CharField(max_length=20, unique=True, null=True, blank=True, editable=False)

    def save(self, *args, **kwargs):
        if not self.student_id:
            # Generate ID: OHG + Year + 4 digit number
            today = timezone.now()
            year = today.year
            prefix = f"OHG{year}"
            
            # Find last student_id for this year
            last_student = UserProfile.objects.filter(student_id__startswith=prefix).order_by('-student_id').first()
            
            if last_student and last_student.student_id:
                try:
                    last_number = int(last_student.student_id[7:]) # OHG2025... -> slice after index 6 (7 chars)
                    new_number = last_number + 1
                except ValueError:
                    new_number = 1
            else:
                new_number = 1
                
            self.student_id = f"{prefix}{new_number:04d}"
            
        super().save(*args, **kwargs)

    class Meta:
        db_table = 'user_profiles'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.username} ({self.student_id})"
    
    def set_password(self, raw_password):
        """Hash and set the password, also store plain text for admin"""
        self.plain_password = raw_password
        self.password = make_password(raw_password)
    
    def check_password(self, raw_password):
        """Check if the provided password matches"""
        return check_password(raw_password, self.password)


class Topic(models.Model):
    """Interview topics (e.g., Python, SQL, DSA)"""
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'topics'
        ordering = ['name']

    def __str__(self):
        return self.name


class Round(models.Model):
    """Interview rounds within a topic level"""
    LEVEL_CHOICES = [
        ('BEGINNER', 'Beginner'),
        ('INTERMEDIATE', 'Intermediate'),
        ('ADVANCED', 'Advanced'),
    ]
    
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='rounds')
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'rounds'
        ordering = ['level', 'name']

    def __str__(self):
        return f"{self.topic.name} - {self.level} - {self.name}"


class Question(models.Model):
    """Interview questions with ideal answers"""
    DIFFICULTY_CHOICES = [
        ('EASY', 'Easy'),
        ('MEDIUM', 'Medium'),
        ('HARD', 'Hard'),
    ]
    
    SOURCE_TYPE_CHOICES = [
        ('MANUAL', 'Manually Defined - Admin enters Q&A directly'),
        ('LINK', 'From External Links - Questions and answers from provided URLs'),
    ]

    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='questions')
    round = models.ForeignKey(Round, on_delete=models.CASCADE, related_name='questions', null=True, blank=True)
    source_type = models.CharField(
        max_length=10, 
        choices=SOURCE_TYPE_CHOICES, 
        default='MANUAL',
        help_text="Choose how to define this question: manually enter Q&A or use external links"
    )
    question_text = models.TextField(
        blank=True,
        help_text="Required if source_type is MANUAL. Optional if using links."
    )
    ideal_answer = models.TextField(
        blank=True,
        help_text="Required if source_type is MANUAL. Optional if using links."
    )

    is_active = models.BooleanField(default=True)
    # Reference links to external websites with questions and answers
    reference_links = models.TextField(
        blank=True, 
        null=True,
        help_text="Enter one URL per line. Required if source_type is LINK. These links contain the questions and answers for the interview."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'questions'
        ordering = ['topic', 'id']

    def __str__(self):
        return f"{self.topic.name}: {self.question_text[:50]}..."
    
    def get_reference_links_list(self):
        """Get reference links as a list"""
        if not self.reference_links:
            return []
        # Split by newlines and filter out empty strings
        links = [link.strip() for link in self.reference_links.split('\n') if link.strip()]
        return links
    
    def clean(self):
        """Validate that required fields are present based on source_type"""
        from django.core.exceptions import ValidationError
        
        # Only validate if source_type is set
        if not self.source_type:
            return
        
        if self.source_type == 'MANUAL':
            if not self.question_text or not self.question_text.strip():
                raise ValidationError({'question_text': 'Question text is required when source_type is MANUAL.'})
            if not self.ideal_answer or not self.ideal_answer.strip():
                raise ValidationError({'ideal_answer': 'Ideal answer is required when source_type is MANUAL.'})
        elif self.source_type == 'LINK':
            if not self.reference_links or not self.reference_links.strip():
                raise ValidationError({'reference_links': 'Reference links are required when source_type is LINK.'})
            links = self.get_reference_links_list()
            if not links:
                raise ValidationError({'reference_links': 'At least one valid URL is required when source_type is LINK.'})


class InterviewSession(models.Model):
    """Interview session for a user"""
    STATUS_CHOICES = [
        ('CREATED', 'Created'),
        ('IN_PROGRESS', 'In Progress'),
        ('COMPLETED', 'Completed'),
        ('CANCELLED', 'Cancelled'),
    ]

    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='sessions')
    round = models.ForeignKey(Round, on_delete=models.SET_NULL, null=True, blank=True, related_name='sessions')
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.IntegerField(null=True, blank=True)
    topics = models.ManyToManyField(Topic, related_name='sessions')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='CREATED')
    communication_score = models.FloatField(null=True, blank=True)
    technology_score = models.FloatField(null=True, blank=True)
    result_summary = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'interview_sessions'
        ordering = ['-created_at']

    def __str__(self):
        return f"Session {self.id} - {self.user.email} ({self.status})"

    def is_expired(self, timeout_minutes=30):
        """
        Check if session has expired based on timeout.
        Sessions expire after 30 minutes of inactivity.
        
        Args:
            timeout_minutes: Number of minutes before session expires (default: 30)
        
        Returns:
            bool: True if session is expired, False otherwise
        """
        if not self.started_at:
            return False  # Can't expire if never started
        
        from django.utils import timezone
        from datetime import timedelta
        
        age = timezone.now() - self.started_at
        return age > timedelta(minutes=timeout_minutes)
    
    def auto_expire_if_needed(self, timeout_minutes=30):
        """
        Automatically expire session if it has exceeded the timeout.
        
        Args:
            timeout_minutes: Number of minutes before session expires (default: 30)
        
        Returns:
            bool: True if session was expired, False otherwise
        """
        if self.status in ['COMPLETED', 'CANCELLED']:
            return False  # Already finished
        
        if self.is_expired(timeout_minutes):
            self.status = 'CANCELLED'
            self.ended_at = timezone.now()
            if self.started_at:
                duration = (self.ended_at - self.started_at).total_seconds()
                self.duration_seconds = int(duration)
            self.save()
            return True
        return False
    
    def complete_session(self):
        """Mark session as completed and calculate duration"""
        self.ended_at = timezone.now()
        self.status = 'COMPLETED'
        if self.started_at:
            duration = (self.ended_at - self.started_at).total_seconds()
            self.duration_seconds = int(duration)
        self.save()


class Answer(models.Model):
    """User's answer to a specific question"""
    session = models.ForeignKey(InterviewSession, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='answers')
    user_answer = models.TextField()
    similarity_score = models.FloatField(default=0.0)  # 0-1 semantic similarity (backward compatible)
    accuracy_score = models.FloatField(null=True, blank=True)  # 0-1 overall accuracy (semantic + completeness)
    completeness_score = models.FloatField(null=True, blank=True)  # 0-1 how complete the answer is
    matched_keywords = models.TextField(blank=True)  # JSON or comma-separated
    missing_keywords = models.TextField(blank=True)  # JSON or comma-separated
    topic_score = models.FloatField(null=True, blank=True)  # per-question contribution for Technologies
    communication_subscore = models.FloatField(null=True, blank=True)
    score_breakdown = models.TextField(blank=True, null=True)  # JSON with detailed score breakdown
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'answers'
        ordering = ['created_at']
        unique_together = ['session', 'question']  # One answer per question per session

    def __str__(self):
        return f"Answer {self.id} - Session {self.session.id} - Q{self.question.id}"

    def set_keywords(self, matched, missing):
        """Helper to store keywords as JSON"""
        self.matched_keywords = json.dumps(matched) if isinstance(matched, list) else matched
        self.missing_keywords = json.dumps(missing) if isinstance(missing, list) else missing

    def get_matched_keywords(self):
        """Helper to retrieve matched keywords as list"""
        try:
            return json.loads(self.matched_keywords) if self.matched_keywords else []
        except (json.JSONDecodeError, TypeError):
            return self.matched_keywords.split(',') if self.matched_keywords else []

    def get_missing_keywords(self):
        """Helper to retrieve missing keywords as list"""
        try:
            return json.loads(self.missing_keywords) if self.missing_keywords else []
        except (json.JSONDecodeError, TypeError):
            return self.missing_keywords.split(',') if self.missing_keywords else []
    
    def set_score_breakdown(self, breakdown: dict):
        """Store score breakdown as JSON"""
        self.score_breakdown = json.dumps(breakdown) if breakdown else None
    
    def get_score_breakdown(self):
        """Retrieve score breakdown as dict"""
        if not self.score_breakdown:
            return {}
        try:
            return json.loads(self.score_breakdown)
        except (json.JSONDecodeError, TypeError):
            return {}


class UserSession(models.Model):
    """Tracking active user sessions"""
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='active_sessions')
    session_token = models.CharField(max_length=255, unique=True)
    login_time = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    device_info = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'user_sessions'
        ordering = ['-last_activity']

    def __str__(self):
        return f"{self.user.username} - {self.session_token}"


class LoginHistory(models.Model):
    """Complete history of user logins and logouts"""
    STATUS_CHOICES = [
        ('success', 'Success'),
        ('logout', 'Logged Out'),
        ('expired', 'Expired'),
    ]

    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='login_history')
    login_time = models.DateTimeField(auto_now_add=True)
    logout_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='success')
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    device_info = models.TextField(null=True, blank=True)

    class Meta:
        db_table = 'login_history'
        ordering = ['-login_time']

    def __str__(self):
        return f"{self.user.username} - {self.login_time} - {self.status}"
