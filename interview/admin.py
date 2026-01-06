from django.contrib import admin
from .models import UserProfile, Topic, Question, InterviewSession, Answer


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'name', 'student_id', 'role', 'enrolled_course', 'is_active', 'access_type', 'has_used_trial', 'created_at']
    list_filter = ['is_active', 'role', 'access_type', 'has_used_trial', 'enrolled_course', 'created_at']
    search_fields = ['username', 'email', 'name', 'student_id']
    readonly_fields = ['student_id', 'created_at', 'updated_at', 'password_display']
    fieldsets = (
        ('Authentication', {
            'fields': ('username', 'password', 'is_active'),
            'description': 'Set username and password. Password will be automatically hashed when saved.'
        }),
        ('User Information', {
            'fields': ('email', 'name', 'student_id', 'role')
        }),
        ('Course Enrollment', {
            'fields': ('enrolled_course',),
            'description': 'Restrict user to a specific course (Topic).'
        }),
        ('Access Control', {
            'fields': ('access_type', 'has_used_trial'),
            'description': 'Trial: One free interview. Full Access: Unlimited interviews.'
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def password_display(self, obj):
        """Show password status"""
        if obj.pk and obj.password:
            return "Password is set (hidden for security)"
        return "No password set"
    password_display.short_description = 'Password Status'
    
    def save_model(self, request, obj, form, change):
        # If password is being set and it's not already hashed, hash it
        if 'password' in form.changed_data and form.cleaned_data.get('password'):
            password = form.cleaned_data['password']
            # Only hash if it's not already a hash (doesn't start with common hash prefixes)
            if not (password.startswith('pbkdf2_') or password.startswith('bcrypt$') or password.startswith('argon2')):
                obj.set_password(password)
            else:
                # If it's already hashed, use it as is
                obj.password = password
        super().save_model(request, obj, form, change)


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ['name', 'description', 'question_count', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'question_count_display']
    fieldsets = (
        ('Topic Information', {
            'fields': ('name', 'description'),
            'description': 'Enter the topic name and optional description. Topics are used to categorize interview questions.'
        }),
        ('Statistics', {
            'fields': ('question_count_display',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def question_count(self, obj):
        """Display number of questions for this topic"""
        return obj.questions.filter(is_active=True).count()
    question_count.short_description = 'Questions'
    
    def question_count_display(self, obj):
        """Display question count in detail view"""
        if obj.pk:
            total = obj.questions.count()
            active = obj.questions.filter(is_active=True).count()
            return f"{active} active, {total} total"
        return "Save topic to see question count"
    question_count_display.short_description = 'Question Count'


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ['question_text_display', 'topic', 'source_type', 'is_active', 'has_reference_links', 'created_at']
    list_filter = ['source_type', 'topic', 'is_active', 'created_at']
    search_fields = ['question_text', 'ideal_answer', 'reference_links']
    readonly_fields = ['created_at', 'updated_at', 'reference_links_preview']
    
    def save_model(self, request, obj, form, change):
        """Override save to run validation"""
        obj.full_clean()
        super().save_model(request, obj, form, change)
    fieldsets = (
        ('Source Type', {
            'fields': ('source_type',),
            'description': 'Choose how to define this question: Manually enter Q&A or provide links to external websites containing questions and answers.'
        }),
        ('Question Details', {
            'fields': ('topic', 'question_text', 'is_active'),
            'description': 'Required if source_type is MANUAL. Question text is optional if using links.'
        }),
        ('Answer (Manual Definition)', {
            'fields': ('ideal_answer',),
            'description': 'Required if source_type is MANUAL. Leave blank if using links.',
            'classes': ('collapse',)
        }),
        ('External Links (Link-based Definition)', {
            'fields': ('reference_links', 'reference_links_preview'),
            'description': 'Required if source_type is LINK. Add URLs to external websites that contain relevant questions and answers. Enter one URL per line. These links will be used as the source of questions and answers during interviews.',
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def question_text_display(self, obj):
        """Display question text or indicate it's from links"""
        if obj.source_type == 'LINK':
            if obj.question_text:
                return f"[From Links] {obj.question_text[:50]}..."
            return "[From Links] (No preview text)"
        return obj.question_text[:100] + "..." if len(obj.question_text) > 100 else obj.question_text
    question_text_display.short_description = 'Question'
    
    def has_reference_links(self, obj):
        """Check if question has reference links"""
        return bool(obj.reference_links and obj.reference_links.strip())
    has_reference_links.boolean = True
    has_reference_links.short_description = 'Has Links'
    
    def reference_links_preview(self, obj):
        """Display reference links as clickable links"""
        from django.utils.html import format_html
        
        if not obj or not obj.reference_links:
            return "No reference links added"
        
        links = obj.get_reference_links_list()
        if not links:
            return "No reference links added"
        
        html_links = []
        for link in links:
            html_links.append(f'<a href="{link}" target="_blank" rel="noopener noreferrer">{link}</a>')
        
        return format_html('<br>'.join(html_links))
    reference_links_preview.short_description = 'Reference Links Preview'


@admin.register(InterviewSession)
class InterviewSessionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'status', 'communication_score', 'technology_score', 'started_at']
    list_filter = ['status', 'started_at', 'created_at']
    search_fields = ['user__username', 'user__email', 'user__name']
    readonly_fields = ['created_at', 'updated_at']
    filter_horizontal = ['topics']


@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    list_display = ['id', 'session', 'question', 'similarity_score', 'communication_subscore', 'created_at']
    list_filter = ['created_at', 'session__status']
    search_fields = ['user_answer', 'session__user__username', 'session__user__email']
    readonly_fields = ['created_at']
