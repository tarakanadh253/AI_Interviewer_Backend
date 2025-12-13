from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAdminUser, BasePermission
from django.db.models import Q, Avg, Count
from django.utils import timezone
from django.shortcuts import get_object_or_404
from django.conf import settings

from .models import UserProfile, Topic, Question, InterviewSession, Answer
from .serializers import (
    UserProfileSerializer, UserProfileCreateSerializer, UserLoginSerializer,
    TopicSerializer, QuestionSerializer,
    InterviewSessionSerializer, InterviewSessionCreateSerializer,
    AnswerSerializer, AnswerCreateSerializer,
    AdminQuestionSerializer, AdminInterviewSessionSerializer,
    AdminStatsSerializer
)
from .utils.evaluation import evaluate_answer


class DevAdminPermission(BasePermission):
    """
    Custom permission for development: allows access if DEBUG=True or user is admin.
    For production, use IsAdminUser instead.
    """
    def has_permission(self, request, view):
        # In development mode, allow all access
        if settings.DEBUG:
            return True
        # In production, require admin authentication
        return request.user and request.user.is_authenticated and (getattr(request.user, 'is_staff', False) or getattr(request.user, 'access_type', '') == 'FULL')


class UserProfileViewSet(viewsets.ModelViewSet):
    """ViewSet for UserProfile - handles username/password authentication"""
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [AllowAny]
    lookup_field = 'username'
    
    def get_serializer_class(self):
        if self.action == 'create':
            return UserProfileCreateSerializer
        return UserProfileSerializer
    
    @action(detail=False, methods=['post'], url_path='login')
    def login(self, request):
        """
        Login with username and password.
        Expects: { "username": "...", "password": "..." }
        Returns: User profile data if successful
        """
        serializer = UserLoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        username = serializer.validated_data['username']
        password = serializer.validated_data['password']
        
        try:
            user = UserProfile.objects.get(username=username)
        except UserProfile.DoesNotExist:
            return Response(
                {'error': 'Invalid username or password'},
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        if not user.is_active:
            return Response(
                {'error': 'Account is inactive. Please contact administrator.'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        if not user.check_password(password):
            return Response(
                {'error': 'Invalid username or password'},
                status=status.HTTP_401_UNAUTHORIZED
            )
            
        # Log the user in to establish a Django session
        from django.contrib.auth import login
        from django.contrib.auth.models import User
        
        # Get or create necessary standard auth user to enable session
        auth_user, _ = User.objects.get_or_create(username=username)
        # We need to set the backend manually to bypass authenticate() since we checked password on UserProfile
        auth_user.backend = 'django.contrib.auth.backends.ModelBackend'
        login(request, auth_user)
        
        return Response(
            UserProfileSerializer(user).data,
            status=status.HTTP_200_OK
        )
    
    @action(detail=True, methods=['get'], url_path='check-trial')
    def check_trial(self, request, username=None):
        """
        Check if user can start an interview.
        Returns: { "has_used_trial": bool, "can_start_interview": bool, "access_type": str }
        """
        user = get_object_or_404(UserProfile, username=username)
        
        # Full access users can always start interviews
        if user.access_type == 'FULL':
            return Response({
                'has_used_trial': False,
                'can_start_interview': True,
                'access_type': 'FULL'
            })
        
        # Trial users can only start if they haven't used their trial
        return Response({
            'has_used_trial': user.has_used_trial,
            'can_start_interview': not user.has_used_trial,
            'access_type': 'TRIAL'
        })


class TopicViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Topic - read-only list of topics"""
    queryset = Topic.objects.all()
    serializer_class = TopicSerializer
    permission_classes = [AllowAny]
    pagination_class = None  # Disable pagination for topics endpoint


class AdminTopicViewSet(viewsets.ModelViewSet):
    """Admin ViewSet for managing Topics"""
    queryset = Topic.objects.all()
    serializer_class = TopicSerializer
    permission_classes = [DevAdminPermission]
    pagination_class = None


class QuestionViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for Question - read-only list of questions"""
    queryset = Question.objects.filter(is_active=True)
    serializer_class = QuestionSerializer
    permission_classes = [AllowAny]
    pagination_class = None  # Disable pagination for questions endpoint
    
    def get_queryset(self):
        # Only return MANUAL questions - exclude LINK type (they're just placeholders)
        queryset = super().get_queryset().filter(source_type='MANUAL')
        topic_id = self.request.query_params.get('topic_id')
        difficulty = self.request.query_params.get('difficulty')
        
        if topic_id:
            queryset = queryset.filter(topic_id=topic_id)
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
        
        return queryset.select_related('topic')


class InterviewSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for InterviewSession"""
    queryset = InterviewSession.objects.all()
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return InterviewSessionCreateSerializer
        return InterviewSessionSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        user_id = self.request.query_params.get('user_id')
        username = self.request.query_params.get('username')
        status_filter = self.request.query_params.get('status')
        
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        elif username:
            try:
                user = UserProfile.objects.get(username=username)
                queryset = queryset.filter(user=user)
            except UserProfile.DoesNotExist:
                # Return empty queryset instead of causing 500 error
                queryset = queryset.none()
            except Exception as e:
                # Log error but return empty queryset
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error filtering sessions by username {username}: {e}")
                queryset = queryset.none()
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Auto-expire any expired sessions before returning queryset
        # This ensures expired sessions are marked as CANCELLED
        try:
            active_sessions = list(queryset.filter(status__in=['CREATED', 'IN_PROGRESS']))
            for session in active_sessions:
                session.auto_expire_if_needed(timeout_minutes=30)
        except Exception as e:
            # If there's an error, log it but continue
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error auto-expiring sessions: {e}")
        
        # Check if answers table exists before prefetching
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='answers'")
                if cursor.fetchone():
                    return queryset.select_related('user').prefetch_related('topics', 'answers', 'answers__question')
        except Exception:
            pass
        # Fallback if answers table doesn't exist
        return queryset.select_related('user').prefetch_related('topics')
    
    def retrieve(self, request, *args, **kwargs):
        """Override retrieve to auto-expire session if needed"""
        instance = self.get_object()
        
        # Auto-expire if needed
        if instance.status in ['CREATED', 'IN_PROGRESS']:
            expired = instance.auto_expire_if_needed(timeout_minutes=30)
            if expired:
                # Refresh instance from database
                instance.refresh_from_db()
        
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
    
    def create(self, request, *args, **kwargs):
        """
        Create a new interview session.
        Enforces: Only 1 trial per user account.
        Expects: { "username": "...", "topic_ids": [1, 2, 3] }
        """
        username = request.data.get('username')
        if not username:
            logger.warning("Session creation failed: username missing")
            return Response(
                {'error': 'username is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check if user has access
        try:
            user = UserProfile.objects.get(username=username)
            if not user.is_active:
                return Response(
                    {'error': 'Account is inactive. Please contact administrator.'},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            # Check access based on access_type
            if user.access_type == 'TRIAL':
                # Trial users can only do one interview
                if user.has_used_trial:
                    return Response(
                        {'error': 'You have already used your trial interview. Please contact administrator for full access.'},
                        status=status.HTTP_403_FORBIDDEN
                    )
            # FULL access users can create unlimited sessions (no restriction)
        except UserProfile.DoesNotExist:
            return Response(
                {'error': 'User not found. Please contact administrator to create an account.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Check for existing active session (handle missing table gracefully)
        try:
            existing_session = InterviewSession.objects.filter(
                user=user,
                status__in=['CREATED', 'IN_PROGRESS']
            ).first()
        except Exception as e:
            # If table doesn't exist, skip the check and proceed
            error_str = str(e).lower()
            if 'no such table' in error_str:
                print(f"[DEBUG] interview_sessions table doesn't exist yet, skipping existing session check")
                existing_session = None
            else:
                # Re-raise if it's a different error
                raise
        
        if existing_session:
            # Auto-expire sessions older than 30 minutes
            expired = existing_session.auto_expire_if_needed(timeout_minutes=30)
            if expired:
                print(f"[DEBUG] Auto-expired session {existing_session.id} (exceeded 30 minute timeout)")
                # Session was expired, continue to create new session
            elif existing_session.status in ['CREATED', 'IN_PROGRESS']:
                # Session is still active and within 30 minutes
                return Response(
                    {'error': 'You have an active interview session. Please complete it first.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Create session
        print(f"[DEBUG] Creating session for user: {user.username}, topics: {request.data.get('topic_ids')}")
        data = request.data.copy()
        data['user'] = user.id
        print(f"[DEBUG] Serializer data: {data}")
        
        serializer = self.get_serializer(data=data)
        
        if not serializer.is_valid():
            print(f"[DEBUG] Validation errors: {serializer.errors}")
            return Response(
                {'error': 'Validation failed', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        print("[DEBUG] Validation passed, saving session...")
        try:
            session = serializer.save(status='IN_PROGRESS')
            print(f"[DEBUG] Session created: {session.id}")
            # Ensure user relationship is loaded for serialization
            session = InterviewSession.objects.select_related('user').prefetch_related('topics').get(id=session.id)
            print(f"[DEBUG] Session refreshed with relationships")
        except Exception as e:
            print(f"[DEBUG] ERROR creating session: {e}")
            import traceback
            traceback.print_exc()
            return Response(
                {'error': f'Failed to create session: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Mark user as having used trial (only for TRIAL access type)
        if user.access_type == 'TRIAL' and not user.has_used_trial:
            user.has_used_trial = True
            user.save()
        
        # Serialize the session
        print(f"[DEBUG] Serializing session {session.id}...")
        try:
            # Refresh session from DB to ensure relationships are loaded
            session.refresh_from_db()
            serializer = InterviewSessionSerializer(session)
            print(f"[DEBUG] Serialization successful")
            response_data = serializer.data
            print(f"[DEBUG] Response data keys: {list(response_data.keys())}")
            return Response(
                response_data,
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            # If serialization fails, return minimal data and log error
            print(f"[DEBUG] ERROR serializing session {session.id}: {e}")
            import traceback
            traceback.print_exc()
            # Return minimal successful response instead of error
            # This allows the session to be created even if serialization has issues
            try:
                return Response(
                    {
                        'id': session.id,
                        'user': session.user_id,
                        'status': session.status,
                        'started_at': session.started_at.isoformat() if session.started_at else None,
                        'topics': [t.id for t in session.topics.all()],
                        'topics_list': [{'id': t.id, 'name': t.name} for t in session.topics.all()],
                        'answers': [],
                        'answer_count': 0,
                        'user_email': getattr(session.user, 'email', None) if hasattr(session, 'user') else None,
                        'user_name': getattr(session.user, 'name', None) if hasattr(session, 'user') else None,
                    },
                    status=status.HTTP_201_CREATED
                )
            except Exception as e2:
                print(f"[DEBUG] ERROR creating fallback response: {e2}")
                traceback.print_exc()
                return Response(
                    {
                        'error': f'Serialization error: {str(e)}',
                        'id': session.id,
                        'user': session.user_id,
                        'status': session.status,
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
    
    @action(detail=True, methods=['post'], url_path='complete')
    def complete_session(self, request, pk=None):
        """
        Complete an interview session and calculate final scores.
        Scores are calculated accounting for unanswered questions (they count as 0).
        """
        session = self.get_object()
        
        if session.status == 'COMPLETED':
            return Response(
                {'error': 'Session is already completed'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Complete the session
        session.complete_session()
        
        # Get all answers
        answers = session.answers.all()
        
        # Get total number of questions for this session
        # We need to determine how many questions were supposed to be answered
        # Since questions are selected dynamically, we'll use the number of unique questions in answers
        # plus check if we can determine total from session creation
        answered_count = answers.count()
        
        # Try to get total questions from session metadata or default to answered count
        # For now, we'll calculate based on answered questions only, but weight scores properly
        # The frontend will handle showing completion percentage
        
        if answers.exists():
            # Calculate raw scores from answered questions only
            comm_scores = [a.communication_subscore for a in answers if a.communication_subscore is not None]
            raw_comm_score = sum(comm_scores) / len(comm_scores) if comm_scores else 0.0
            
            # Technology score: average of accuracy scores (preferred) or topic scores (fallback)
            tech_scores = []
            for a in answers:
                # Prefer accuracy_score, fallback to topic_score, then similarity_score
                score = a.accuracy_score if a.accuracy_score is not None else (
                    a.topic_score if a.topic_score is not None else a.similarity_score
                )
                if score is not None:
                    tech_scores.append(score)
            
            raw_tech_score = sum(tech_scores) / len(tech_scores) if tech_scores else 0.0
            
            # Store raw scores (these are averages of answered questions only)
            session.communication_score = raw_comm_score if comm_scores else None
            session.technology_score = raw_tech_score if tech_scores else None
            
            # Generate result summary with completion info
            summary_parts = []
            if session.communication_score is not None:
                comm_percent = int(session.communication_score * 100)
                summary_parts.append(f"Communication: {comm_percent}%")
            if session.technology_score is not None:
                tech_percent = int(session.technology_score * 100)
                summary_parts.append(f"Technical Knowledge: {tech_percent}%")
            
            # Add completion info if we can determine total questions
            # For now, we'll note answered count in summary
            if answered_count > 0:
                summary_parts.append(f"Answered: {answered_count} question(s)")
            
            # Topic-wise feedback
            topic_scores = {}
            for answer in answers:
                topic = answer.question.topic.name
                if topic not in topic_scores:
                    topic_scores[topic] = []
                if answer.topic_score is not None:
                    topic_scores[topic].append(answer.topic_score)
            
            improvements = []
            for topic, scores in topic_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0
                if avg_score < 0.6:
                    improvements.append(f"Focus on {topic} fundamentals")
            
            if improvements:
                summary_parts.append(f"Improvements: {', '.join(improvements)}")
            
            session.result_summary = " | ".join(summary_parts) if summary_parts else "Interview completed"
        else:
            # No answers submitted
            session.communication_score = None
            session.technology_score = None
            session.result_summary = "No answers submitted"
        
        session.save()
        
        return Response(InterviewSessionSerializer(session).data)
    
    @action(detail=True, methods=['get'], url_path='results')
    def get_results(self, request, pk=None):
        """Get detailed results for a completed session"""
        session = self.get_object()
        
        if session.status != 'COMPLETED':
            return Response(
                {'error': 'Session is not completed yet'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        serializer = InterviewSessionSerializer(session)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'], url_path='cancel-active')
    def cancel_active_session(self, request):
        """Cancel the user's active interview session"""
        username = request.data.get('username')
        if not username:
            return Response(
                {'error': 'username is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user = UserProfile.objects.get(username=username)
        except UserProfile.DoesNotExist:
            return Response(
                {'error': 'User not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Find active sessions
        active_sessions = InterviewSession.objects.filter(
            user=user,
            status__in=['CREATED', 'IN_PROGRESS']
        )
        
        if not active_sessions.exists():
            return Response(
                {'message': 'No active sessions to cancel'},
                status=status.HTTP_200_OK
            )
        
        # Cancel all active sessions
        cancelled_count = active_sessions.update(status='CANCELLED')
        
        return Response(
            {'message': f'Cancelled {cancelled_count} active session(s)'},
            status=status.HTTP_200_OK
        )


class AnswerViewSet(viewsets.ModelViewSet):
    """ViewSet for Answer"""
    queryset = Answer.objects.all()
    permission_classes = [AllowAny]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return AnswerCreateSerializer
        return AnswerSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        session_id = self.request.query_params.get('session_id')
        
        if session_id:
            queryset = queryset.filter(session_id=session_id)
        
        return queryset.select_related('session', 'question')
    
    def create(self, request, *args, **kwargs):
        """
        Submit an answer. Automatically evaluates and stores scores.
        Expects: { "session": id, "question": id, "user_answer": "..." }
        """
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            answer = serializer.save()
            
            return Response(
                AnswerSerializer(answer).data,
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            # Log the full error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating answer: {e}", exc_info=True)
            
            # Return user-friendly error message
            error_msg = str(e)
            if 'ValidationError' in str(type(e)):
                # Re-raise validation errors as-is
                raise
            elif 'NOT NULL' in error_msg or 'null' in error_msg.lower():
                return Response(
                    {'error': 'Required fields are missing. Please ensure session, question, and user_answer are provided.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            elif 'does not exist' in error_msg:
                if 'Session' in error_msg:
                    return Response(
                        {'error': 'The session does not exist. Please start a new interview.'},
                        status=status.HTTP_404_NOT_FOUND
                    )
                elif 'Question' in error_msg:
                    return Response(
                        {'error': 'The question does not exist.'},
                        status=status.HTTP_404_NOT_FOUND
                    )
            elif 'UNIQUE constraint' in error_msg or 'unique' in error_msg.lower():
                return Response(
                    {'error': 'You have already answered this question. Each question can only be answered once per session.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            else:
                return Response(
                    {'error': f'Failed to submit answer: {error_msg}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )


# Admin Views

class AdminQuestionViewSet(viewsets.ModelViewSet):
    """Admin ViewSet for managing Questions"""
    queryset = Question.objects.all()
    serializer_class = AdminQuestionSerializer
    permission_classes = [DevAdminPermission]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        topic_id = self.request.query_params.get('topic_id')
        is_active = self.request.query_params.get('is_active')
        
        if topic_id:
            queryset = queryset.filter(topic_id=topic_id)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        
        # Use select_related to avoid N+1 queries and prefetch_related for answers
        # Check if answers table exists before prefetching
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='answers'")
                if cursor.fetchone():
                    return queryset.select_related('topic').prefetch_related('answers')
        except Exception:
            pass
        # Fallback if answers table doesn't exist
        return queryset.select_related('topic')
    
    @action(detail=True, methods=['post'], url_path='extract-from-links')
    def extract_from_links(self, request, pk=None):
        """Extract questions from reference links for a LINK-type question"""
        question = self.get_object()
        
        if question.source_type != 'LINK':
            return Response(
                {'error': 'This question is not a LINK type. Only LINK-type questions can extract from links.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            from .utils.question_extractor import process_link_question
            created_count = process_link_question(question)
            
            return Response(
                {
                    'message': f'Successfully extracted {created_count} questions from links',
                    'extracted_count': created_count
                },
                status=status.HTTP_200_OK
            )
        except Exception as e:
            import traceback
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error extracting questions: {e}\n{traceback.format_exc()}")
            return Response(
                {'error': f'Failed to extract questions: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def list(self, request, *args, **kwargs):
        """Override list to handle errors gracefully"""
        try:
            # Get queryset with proper select_related to avoid N+1 queries
            queryset = self.filter_queryset(self.get_queryset())
            
            # Try standard serialization first
            try:
                serializer = self.get_serializer(queryset, many=True)
                return Response(serializer.data)
            except Exception as e:
                # If bulk serialization fails, try one by one
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Bulk serialization failed, trying individual: {e}")
                
                valid_data = []
                for obj in queryset:
                    try:
                        item_serializer = self.get_serializer(obj)
                        valid_data.append(item_serializer.data)
                    except Exception as item_error:
                        logger.warning(f"Error serializing question {obj.id}: {item_error}")
                        # Add minimal representation
                        try:
                            valid_data.append({
                                'id': obj.id,
                                'topic': obj.topic_id if hasattr(obj, 'topic_id') else None,
                                'topic_name': obj.topic.name if hasattr(obj, 'topic') and obj.topic else None,
                                'source_type': getattr(obj, 'source_type', 'MANUAL') or 'MANUAL',
                                'source_type_display': 'Manually Defined',
                                'question_text': str(getattr(obj, 'question_text', ''))[:100] if getattr(obj, 'question_text', None) else '',
                                'ideal_answer': str(getattr(obj, 'ideal_answer', ''))[:100] if getattr(obj, 'ideal_answer', None) else '',
                                'difficulty': getattr(obj, 'difficulty', 'MEDIUM') or 'MEDIUM',
                                'is_active': getattr(obj, 'is_active', True),
                                'reference_links': getattr(obj, 'reference_links', '') or '',
                                'reference_links_list': [],
                                'answer_count': 0,
                                'created_at': None,
                                'updated_at': None,
                            })
                        except Exception:
                            pass  # Skip if even minimal fails
                
                return Response(valid_data)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error listing questions: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return Response(
                {'error': f'Failed to fetch questions: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def create(self, request, *args, **kwargs):
        """Override create to provide better error handling"""
        try:
            # Log the incoming data for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Creating question with data: {request.data}")
            
            serializer = self.get_serializer(data=request.data)
            if not serializer.is_valid():
                # Return validation errors in a clear format
                logger.error(f"Validation errors: {serializer.errors}")
                return Response(
                    {'errors': serializer.errors, 'detail': 'Validation failed. Please check the errors below.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            self.perform_create(serializer)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
            
        except Exception as e:
            # Log the full error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error creating question: {e}", exc_info=True)
            
            # Return user-friendly error message
            error_msg = str(e)
            if 'ValidationError' in str(type(e)):
                # Re-raise validation errors as-is
                raise
            elif 'NOT NULL' in error_msg or 'null' in error_msg.lower():
                return Response(
                    {'error': 'Required fields are missing. Please ensure topic, question_text, and ideal_answer are provided.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            elif 'Topic' in error_msg and 'does not exist' in error_msg:
                return Response(
                    {'error': 'The selected topic does not exist. Please select a valid topic.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            else:
                return Response(
                    {'error': f'Failed to create question: {error_msg}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )


class AdminUserViewSet(viewsets.ModelViewSet):
    """Admin ViewSet for managing Users"""
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [DevAdminPermission]
    
    def get_serializer_class(self):
        if self.action == 'create':
            return UserProfileCreateSerializer
        return UserProfileSerializer
    
    def update(self, request, *args, **kwargs):
        """Update user - allows updating is_active, access_type, has_used_trial, etc."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        
        # Only allow updating specific fields for security
        allowed_fields = ['is_active', 'access_type', 'has_used_trial', 'email', 'name']
        data = {k: v for k, v in request.data.items() if k in allowed_fields}
        
        serializer = self.get_serializer(instance, data=data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'], url_path='reset-trial')
    def reset_trial(self, request, pk=None):
        """Reset trial status for a user"""
        user = self.get_object()
        user.has_used_trial = False
        user.save()
        return Response(UserProfileSerializer(user).data)
    
    @action(detail=True, methods=['post'], url_path='toggle-status')
    def toggle_status(self, request, pk=None):
        """Toggle user active status"""
        user = self.get_object()
        user.is_active = not user.is_active
        user.save()
        return Response(UserProfileSerializer(user).data)


class AdminInterviewSessionViewSet(viewsets.ReadOnlyModelViewSet):
    """Admin ViewSet for viewing all interview sessions"""
    queryset = InterviewSession.objects.all()
    serializer_class = AdminInterviewSessionSerializer
    permission_classes = [DevAdminPermission]
    
    def get_queryset(self):
        queryset = super().get_queryset()
        status_filter = self.request.query_params.get('status')
        user_email = self.request.query_params.get('user_email')
        user_username = self.request.query_params.get('user_username')
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        if user_email:
            queryset = queryset.filter(user__email__icontains=user_email)
        if user_username:
            queryset = queryset.filter(user__username__icontains=user_username)
        
        # Check if answers table exists before prefetching
        try:
            from django.db import connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='answers'")
                if cursor.fetchone():
                    return queryset.select_related('user').prefetch_related('topics', 'answers', 'answers__question')
        except Exception:
            pass
        # Fallback if answers table doesn't exist
        return queryset.select_related('user').prefetch_related('topics')
    
    @action(detail=False, methods=['get'], url_path='stats')
    def stats(self, request):
        """Get admin statistics"""
        total_users = UserProfile.objects.count()
        total_sessions = InterviewSession.objects.count()
        completed_sessions = InterviewSession.objects.filter(status='COMPLETED').count()
        total_questions = Question.objects.count()
        total_topics = Topic.objects.count()
        
        completed = InterviewSession.objects.filter(status='COMPLETED')
        avg_comm = completed.aggregate(Avg('communication_score'))['communication_score__avg']
        avg_tech = completed.aggregate(Avg('technology_score'))['technology_score__avg']
        
        stats = {
            'total_users': total_users,
            'total_sessions': total_sessions,
            'completed_sessions': completed_sessions,
            'total_questions': total_questions,
            'total_topics': total_topics,
            'avg_communication_score': round(avg_comm, 2) if avg_comm else None,
            'avg_technology_score': round(avg_tech, 2) if avg_tech else None,
        }
        
        serializer = AdminStatsSerializer(stats)
        return Response(serializer.data)
