from rest_framework import serializers
from .models import UserProfile, Topic, Question, InterviewSession, Answer, Round, UserSession, LoginHistory





class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for UserProfile"""
    access_type = serializers.CharField(default='TRIAL', required=False, allow_null=True)
    
    class Meta:
        model = UserProfile
        fields = ['id', 'username', 'email', 'name', 'is_active', 'role', 'access_type', 'plain_password', 'has_used_trial', 'student_id', 'enrolled_course', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at', 'enrolled_course', 'student_id']
        extra_kwargs = {
            'password': {'write_only': True}
        }
    
    def to_representation(self, instance):
        """Standard representation with fallback for access_type"""
        data = super().to_representation(instance)
        
        # Ensure access_type is valid
        if not data.get('access_type') or data.get('access_type') not in ['TRIAL', 'FULL', 'ADMIN']:
            data['access_type'] = 'TRIAL'
        
        return data


class UserProfileCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating UserProfile with password"""
    password = serializers.CharField(write_only=True, required=True, min_length=6)
    access_type = serializers.ChoiceField(
        choices=UserProfile.ACCESS_TYPE_CHOICES,
        default='TRIAL',
        required=False
    )
    
    class Meta:
        model = UserProfile
        fields = ['username', 'password', 'email', 'name', 'is_active', 'role', 'access_type', 'enrolled_course']
    
    def create(self, validated_data):
        password = validated_data.pop('password', None)
        if not password:
            raise serializers.ValidationError({'password': 'Password is required'})
        
        # Ensure access_type has a default if not provided or invalid
        access_type = validated_data.pop('access_type', 'TRIAL')
        if access_type not in ['TRIAL', 'FULL', 'ADMIN']:
            access_type = 'TRIAL'
        
        # Create user instance with all required fields
        try:
            user = UserProfile(
                username=validated_data.get('username'),
                email=validated_data.get('email'),
                name=validated_data.get('name'),
                is_active=validated_data.get('is_active', True),
                role=validated_data.get('role', 'USER'),
                access_type=access_type,
                enrolled_course=validated_data.get('enrolled_course'),
                has_used_trial=False
            )
            user.set_password(password)
            user.save()
            return user
        except Exception as e:
            # Provide better error message
            error_msg = str(e)
            if 'UNIQUE constraint' in error_msg or 'unique' in error_msg.lower():
                raise serializers.ValidationError({'username': 'A user with this username already exists.'})
            elif 'NOT NULL constraint' in error_msg:
                raise serializers.ValidationError({'error': 'Required field is missing.'})
            else:
                raise serializers.ValidationError({'error': f'Failed to create user: {error_msg}'})


class UserLoginSerializer(serializers.Serializer):
    """Serializer for user login"""
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True, write_only=True)


class RoundSerializer(serializers.ModelSerializer):
    """Serializer for Round"""
    question_count = serializers.SerializerMethodField()
    topic_name = serializers.CharField(source='topic.name', read_only=True)
    
    class Meta:
        model = Round
        fields = ['id', 'topic', 'topic_name', 'level', 'name', 'question_count', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']
        
    def get_question_count(self, obj):
        if hasattr(obj, 'question_count'):
            return obj.question_count
        return obj.questions.filter(is_active=True).count()


class TopicSerializer(serializers.ModelSerializer):
    """Serializer for Topic"""
    question_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Topic
        fields = ['id', 'name', 'description', 'question_count', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_question_count(self, obj):
        if hasattr(obj, 'question_count'):
            return obj.question_count
        return obj.questions.filter(is_active=True).count()


class QuestionSerializer(serializers.ModelSerializer):
    """Serializer for Question"""
    topic_name = serializers.CharField(source='topic.name', read_only=True)
    round_name = serializers.CharField(source='round.name', read_only=True, allow_null=True)
    reference_links_list = serializers.SerializerMethodField()
    source_type_display = serializers.CharField(source='get_source_type_display', read_only=True)
    
    class Meta:
        model = Question
        fields = [
            'id', 'topic', 'topic_name', 'round', 'round_name', 'source_type', 'source_type_display',
            'question_text', 'ideal_answer', 'is_active', 
            'reference_links', 'reference_links_list', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_reference_links_list(self, obj):
        """Get reference links as a list"""
        return obj.get_reference_links_list() if obj else []
    
    def validate(self, data):
        """Validate based on source_type"""
        source_type = data.get('source_type', self.instance.source_type if self.instance else 'MANUAL')
        
        if source_type == 'MANUAL':
            if not data.get('question_text', '').strip():
                raise serializers.ValidationError({'question_text': 'Question text is required when source_type is MANUAL.'})
            if not data.get('ideal_answer', '').strip():
                raise serializers.ValidationError({'ideal_answer': 'Ideal answer is required when source_type is MANUAL.'})
        elif source_type == 'LINK':
            reference_links = data.get('reference_links', '')
            if not reference_links or not reference_links.strip():
                raise serializers.ValidationError({'reference_links': 'Reference links are required when source_type is LINK.'})
            # Check if at least one valid link exists
            links = [link.strip() for link in reference_links.split('\n') if link.strip()]
            if not links:
                raise serializers.ValidationError({'reference_links': 'At least one valid URL is required when source_type is LINK.'})
        
        return data


class AnswerSerializer(serializers.ModelSerializer):
    """Serializer for Answer"""
    question_text = serializers.CharField(source='question.question_text', read_only=True)
    question_id = serializers.IntegerField(source='question.id', read_only=True)
    matched_keywords_list = serializers.SerializerMethodField()
    missing_keywords_list = serializers.SerializerMethodField()
    score_breakdown_dict = serializers.SerializerMethodField()
    
    class Meta:
        model = Answer
        fields = [
            'id', 'session', 'question', 'question_id', 'question_text',
            'user_answer', 'similarity_score', 'accuracy_score', 'completeness_score',
            'matched_keywords', 'missing_keywords', 'matched_keywords_list', 'missing_keywords_list',
            'topic_score', 'communication_subscore', 'score_breakdown', 'score_breakdown_dict',
            'created_at'
        ]
        read_only_fields = [
            'id', 'similarity_score', 'accuracy_score', 'completeness_score',
            'matched_keywords', 'missing_keywords', 'topic_score', 
            'communication_subscore', 'score_breakdown', 'created_at'
        ]
    
    def get_matched_keywords_list(self, obj):
        return obj.get_matched_keywords()
    
    def get_missing_keywords_list(self, obj):
        return obj.get_missing_keywords()
    
    def get_score_breakdown_dict(self, obj):
        return obj.get_score_breakdown()


class InterviewSessionSerializer(serializers.ModelSerializer):
    """Serializer for InterviewSession"""
    user_email = serializers.SerializerMethodField()
    user_name = serializers.SerializerMethodField()
    topics_list = serializers.SerializerMethodField()
    answers = serializers.SerializerMethodField()  # Changed to method field to handle missing table
    answer_count = serializers.SerializerMethodField()
    
    class Meta:
        model = InterviewSession
        fields = [
            'id', 'user', 'user_email', 'user_name', 'started_at', 'ended_at',
            'duration_seconds', 'topics', 'topics_list', 'status',
            'communication_score', 'technology_score', 'result_summary',
            'answers', 'answer_count', 'round', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'started_at', 'ended_at', 'duration_seconds',
            'communication_score', 'technology_score', 'result_summary',
            'created_at', 'updated_at'
        ]
    
    def get_user_email(self, obj):
        """Get user email safely"""
        if not obj:
            return None
        try:
            # Try to get user_id first
            user_id = getattr(obj, 'user_id', None)
            if not user_id:
                # Try to get from user relationship
                if hasattr(obj, 'user'):
                    user = obj.user
                    if user:
                        return getattr(user, 'email', None)
                return None
            
            # If we have user_id but user isn't loaded, fetch it
            from .models import UserProfile
            try:
                user = UserProfile.objects.get(id=user_id)
                return getattr(user, 'email', None)
            except UserProfile.DoesNotExist:
                return None
        except Exception:
            return None
    
    def get_user_name(self, obj):
        """Get user name safely"""
        if not obj:
            return None
        try:
            # Try to get user_id first
            user_id = getattr(obj, 'user_id', None)
            if not user_id:
                # Try to get from user relationship
                if hasattr(obj, 'user'):
                    user = obj.user
                    if user:
                        return getattr(user, 'name', None)
                return None
            
            # If we have user_id but user isn't loaded, fetch it
            from .models import UserProfile
            try:
                user = UserProfile.objects.get(id=user_id)
                return getattr(user, 'name', None)
            except UserProfile.DoesNotExist:
                return None
        except Exception:
            return None
    
    def get_topics_list(self, obj):
        """Get topics list safely"""
        if not obj:
            return []
        try:
            return [{'id': t.id, 'name': t.name, 'question_count': t.questions.filter(is_active=True).count()} for t in obj.topics.all()]
        except Exception:
            return []
    
    def get_answers(self, obj):
        """Get answers list safely"""
        if not obj or not hasattr(obj, 'id'):
            return []
        
        try:
            # Return actual answers for completed sessions or sessions with answers
            # Use all() which will be lazy; if table doesn't exist, this might error on evaluation
            # but we catch it below.
            answers = obj.answers.all().select_related('question')
            if answers.exists():
                return [
                    {
                        'id': answer.id,
                        'question': answer.question_id,
                        'question_id': answer.question_id,
                        'question_text': answer.question.question_text if answer.question else "Unknown Question",
                        'user_answer': answer.user_answer,
                        'similarity_score': answer.similarity_score,
                        'accuracy_score': answer.accuracy_score,
                        'completeness_score': answer.completeness_score,
                        'communication_subscore': answer.communication_subscore,
                        'topic_score': answer.topic_score,
                        'created_at': answer.created_at.isoformat() if answer.created_at else None,
                    }
                    for answer in answers
                ]
            return []
        except Exception as e:
            # Log error if needed, or just return empty
            # print(f"Error getting answers for session {obj.id}: {e}")
            return []
    
    def get_answer_count(self, obj):
        """Get answer count safely"""
        if not obj or not hasattr(obj, 'id'):
            return 0
        try:
            return obj.answers.count()
        except Exception:
            return 0


class InterviewSessionCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating InterviewSession"""
    topic_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        required=True
    )
    username = serializers.CharField(write_only=True, required=False)
    round_id = serializers.IntegerField(write_only=True, required=False, allow_null=True)
    
    class Meta:
        model = InterviewSession
        fields = ['user', 'username', 'topic_ids', 'status', 'round_id']
    
    def validate_topic_ids(self, value):
        if not value:
            raise serializers.ValidationError("At least one topic must be selected.")
        return value
    
    def create(self, validated_data):
        """Create session with error handling"""
        try:
            topic_ids = validated_data.pop('topic_ids', [])
            username = validated_data.pop('username', None)  # Remove username, user is already set in view
            round_id = validated_data.pop('round_id', None)
            
            if not topic_ids:
                raise serializers.ValidationError({'topic_ids': 'At least one topic must be selected.'})
            
            # Validate topics exist
            from .models import Topic, Round
            existing_topics = Topic.objects.filter(id__in=topic_ids)
            if existing_topics.count() != len(topic_ids):
                raise serializers.ValidationError({'topic_ids': 'One or more topics do not exist.'})
            
            # Create session
            session = InterviewSession.objects.create(**validated_data)
            session.topics.set(topic_ids)
            
            if round_id:
                try:
                    round_obj = Round.objects.get(id=round_id)
                    session.round = round_obj
                    session.save()
                except Round.DoesNotExist:
                     pass # Should likely raise validation error but keeping it safe for now
            
            return session
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in InterviewSessionCreateSerializer.create: {e}", exc_info=True)
            raise


class AnswerCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating Answer with automatic evaluation"""
    
    class Meta:
        model = Answer
        fields = ['session', 'question', 'user_answer']
    
    def create(self, validated_data):
        from .utils.evaluation import evaluate_answer
        
        session = validated_data['session']
        question = validated_data['question']
        user_answer = validated_data['user_answer']
        
        # Get ideal answer (handle link-based questions)
        ideal_answer = question.ideal_answer or ""
        if question.source_type == 'LINK' and not ideal_answer:
            # For link-based questions, use a generic evaluation
            ideal_answer = "Reference material available in provided links."
        
        # Evaluate the answer using advanced NLP
        evaluation = evaluate_answer(user_answer, ideal_answer)
        
        # Use accuracy_score if available, otherwise fall back to similarity_score
        topic_score = evaluation.get('accuracy_score', evaluation.get('similarity_score', 0.0))
        
        # Create answer with evaluation results
        answer = Answer.objects.create(
            session=session,
            question=question,
            user_answer=user_answer,
            similarity_score=evaluation.get('similarity_score', 0.0),
            accuracy_score=evaluation.get('accuracy_score'),
            completeness_score=evaluation.get('completeness_score'),
            communication_subscore=evaluation.get('communication_subscore', 0.0),
            topic_score=topic_score,  # Use accuracy score for topic score
        )
        
        # Store keywords
        answer.set_keywords(
            evaluation.get('matched_keywords', []),
            evaluation.get('missing_keywords', [])
        )
        
        # Store score breakdown
        if 'score_breakdown' in evaluation:
            answer.set_score_breakdown(evaluation['score_breakdown'])
        
        answer.save()
        
        # Update session scores (will be recalculated when session is completed)
        return answer


# Admin serializers for viewing results and managing Q&A

class AdminQuestionSerializer(serializers.ModelSerializer):
    """Admin serializer for Question with full details"""
    topic_name = serializers.SerializerMethodField()
    round_name = serializers.SerializerMethodField()
    answer_count = serializers.SerializerMethodField()
    reference_links_list = serializers.SerializerMethodField()
    source_type_display = serializers.SerializerMethodField()
    
    class Meta:
        model = Question
        fields = [
            'id', 'topic', 'topic_name', 'round', 'round_name', 'source_type', 'source_type_display',
            'question_text', 'ideal_answer', 'is_active', 
            'reference_links', 'reference_links_list', 'answer_count', 
            'created_at', 'updated_at'
        ]
        extra_kwargs = {
            'topic': {'required': True},
            'is_active': {'required': False, 'allow_null': True},
            'source_type': {'required': False, 'allow_null': True, 'allow_blank': True},
            'question_text': {'required': False, 'allow_blank': True, 'allow_null': True},
            'ideal_answer': {'required': False, 'allow_blank': True, 'allow_null': True},
            'reference_links': {'required': False, 'allow_blank': True, 'allow_null': True},
            'round': {'required': False, 'allow_null': True}
        }
    
    def get_topic_name(self, obj):
        """Get topic name safely"""
        if not obj:
            return None
        try:
            # Try direct access first (if topic is already loaded)
            if hasattr(obj, 'topic'):
                try:
                    topic = obj.topic
                    if topic:
                        return topic.name
                except Exception:
                    pass
            
            # Try to get topic_id and fetch separately
            if hasattr(obj, 'topic_id') and obj.topic_id:
                try:
                    from .models import Topic
                    topic = Topic.objects.get(id=obj.topic_id)
                    return topic.name
                except Topic.DoesNotExist:
                    return None
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def get_round_name(self, obj):
        """Get round name safely"""
        if obj and obj.round:
            return obj.round.name
        return None
    
    def get_answer_count(self, obj):
        """Get answer count"""
        if not obj:
            return 0
        try:
            if hasattr(obj, 'answer_count'):
                return obj.answer_count
            
            if hasattr(obj, 'answers'):
                try:
                    return obj.answers.count()
                except Exception as e:
                    # If table doesn't exist, return 0
                    if 'no such table' in str(e).lower() or 'does not exist' in str(e).lower():
                        return 0
                    # Try direct database query
                    try:
                        from .models import Answer
                        return Answer.objects.filter(question_id=obj.id).count()
                    except Exception:
                        pass
        except Exception:
            pass
        return 0
    
    def get_reference_links_list(self, obj):
        """Get reference links as a list"""
        if not obj:
            return []
        try:
            if hasattr(obj, 'get_reference_links_list'):
                try:
                    return obj.get_reference_links_list()
                except Exception:
                    pass
            
            # Fallback: manual parsing
            if hasattr(obj, 'reference_links') and obj.reference_links:
                try:
                    links = str(obj.reference_links)
                    return [link.strip() for link in links.split('\n') if link.strip()]
                except Exception:
                    pass
        except Exception:
            pass
        return []
    
    def get_source_type_display(self, obj):
        """Get source type display name"""
        if not obj:
            return None
        try:
            # Try Django's built-in get_FOO_display() method first
            if hasattr(obj, 'get_source_type_display'):
                try:
                    return obj.get_source_type_display()
                except Exception:
                    pass
            
            # Fallback: use the choices from the model
            if hasattr(obj, 'source_type') and obj.source_type:
                from .models import Question
                choices = dict(Question.SOURCE_TYPE_CHOICES)
                return choices.get(obj.source_type, obj.source_type)
        except Exception as e:
            # If all else fails, just return the source_type value
            try:
                return obj.source_type if hasattr(obj, 'source_type') else None
            except Exception:
                pass
        return None
    
    def validate(self, data):
        """Validate based on source_type"""
        # Get source_type, defaulting to MANUAL if not provided
        source_type = data.get('source_type')
        if not source_type:
            # Try to get from instance if updating
            source_type = self.instance.source_type if self.instance else 'MANUAL'
            data['source_type'] = source_type
        
        # Ensure source_type is valid
        if source_type not in ['MANUAL', 'LINK']:
            source_type = 'MANUAL'
            data['source_type'] = 'MANUAL'
        
        # Validate topic exists
        topic = data.get('topic')
        if topic:
            # If topic is an integer ID, validate it exists
            if isinstance(topic, int):
                from .models import Topic
                try:
                    Topic.objects.get(id=topic)
                except Topic.DoesNotExist:
                    raise serializers.ValidationError({'topic': f'Topic with id {topic} does not exist.'})
        
        if source_type == 'MANUAL':
            question_text = data.get('question_text') or ''
            ideal_answer = data.get('ideal_answer') or ''
            
            # Convert None to empty string
            if question_text is None:
                question_text = ''
            if ideal_answer is None:
                ideal_answer = ''
            
            if not str(question_text).strip():
                raise serializers.ValidationError({'question_text': 'Question text is required when source_type is MANUAL.'})
            if not str(ideal_answer).strip():
                raise serializers.ValidationError({'ideal_answer': 'Ideal answer is required when source_type is MANUAL.'})
        elif source_type == 'LINK':
            reference_links = data.get('reference_links') or ''
            if reference_links is None:
                reference_links = ''
            
            if not str(reference_links).strip():
                raise serializers.ValidationError({'reference_links': 'Reference links are required when source_type is LINK.'})
            links = [link.strip() for link in str(reference_links).split('\n') if link.strip()]
            if not links:
                raise serializers.ValidationError({'reference_links': 'At least one valid URL is required when source_type is LINK.'})
        
        return data
    
    def create(self, validated_data):
        """Override create to handle validation errors gracefully"""
        try:
            # Import Topic at the top of the method
            from .models import Topic, Round
            
            # Handle topic - it might be an ID or a Topic object
            topic = validated_data.get('topic')
            if not topic:
                raise serializers.ValidationError({'topic': 'Topic is required.'})
            
            # If it's an integer, convert to Topic object
            if isinstance(topic, int):
                try:
                    topic_obj = Topic.objects.get(id=topic)
                    validated_data['topic'] = topic_obj
                except Topic.DoesNotExist:
                    raise serializers.ValidationError({'topic': f'Topic with id {topic} does not exist.'})
            # If it's already a Topic object, keep it
            elif not isinstance(topic, Topic):
                raise serializers.ValidationError({'topic': 'Invalid topic provided. Expected topic ID (integer).'})
            
            # Handle round if present
            round_data = validated_data.get('round')
            if round_data:
                if isinstance(round_data, int):
                    try:
                        round_obj = Round.objects.get(id=round_data)
                        validated_data['round'] = round_obj
                    except Round.DoesNotExist:
                        raise serializers.ValidationError({'round': f'Round with id {round_data} does not exist.'})
            
            # Ensure source_type has a default
            if 'source_type' not in validated_data or not validated_data['source_type']:
                validated_data['source_type'] = 'MANUAL'
            
            # Ensure reference_links defaults to empty string
            if validated_data.get('reference_links') is None:
                validated_data['reference_links'] = ''
            
            # If source_type is LINK, extract questions from links immediately
            source_type = validated_data.get('source_type', 'MANUAL')
            if source_type == 'LINK':
                # Create the question object first (it will be a placeholder)
                question_obj = super().create(validated_data)
                
                # Extract questions from links immediately (synchronously for now)
                try:
                    from .utils.question_extractor import process_link_question
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Extracting questions from links for question {question_obj.id}...")
                    created_count = process_link_question(question_obj)
                    if created_count > 0:
                        logger.info(f"Successfully extracted {created_count} questions from links for question {question_obj.id}")
                    else:
                        logger.warning(f"No questions extracted from links for question {question_obj.id}")
                except Exception as e:
                    import logging
                    import traceback
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error extracting questions from links: {e}\n{traceback.format_exc()}")
                
                return question_obj
            
            # Clean up None/undefined values - set empty strings for optional fields
            if validated_data.get('question_text') is None:
                validated_data['question_text'] = ''
            if validated_data.get('ideal_answer') is None:
                validated_data['ideal_answer'] = ''
            
            # Set defaults

            if 'is_active' not in validated_data:
                validated_data['is_active'] = True
            
            # Create the question
            question = Question.objects.create(**validated_data)
            return question
        except serializers.ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Provide better error messages
            error_msg = str(e)
            if 'no such column' in error_msg.lower():
                # Column is missing - need to run migrations
                raise serializers.ValidationError({
                    'error': f'Database schema error: {error_msg}. Please run: python manage.py migrate'
                })
            elif 'NOT NULL constraint' in error_msg or 'null' in error_msg.lower():
                raise serializers.ValidationError({'error': 'Required fields are missing. Please check topic, question_text, and ideal_answer.'})
            elif 'UNIQUE constraint' in error_msg or 'unique' in error_msg.lower():
                raise serializers.ValidationError({'error': 'A question with these details already exists.'})
            else:
                raise serializers.ValidationError({'error': f'Failed to create question: {error_msg}'})


class AdminInterviewSessionSerializer(serializers.ModelSerializer):
    """Admin serializer for InterviewSession with detailed info"""
    user_email = serializers.EmailField(source='user.email', read_only=True)
    user_name = serializers.CharField(source='user.name', read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)
    topics_list = serializers.SerializerMethodField()
    answers = AnswerSerializer(many=True, read_only=True)
    
    class Meta:
        model = InterviewSession
        fields = [
            'id', 'user', 'user_email', 'user_name', 'user_username',
            'started_at', 'ended_at', 'duration_seconds', 'topics', 'topics_list',
            'status', 'communication_score', 'technology_score', 'result_summary',
            'answers', 'created_at', 'updated_at'
        ]
    
    def get_topics_list(self, obj):
        """Get topics list safely"""
        if not obj:
            return []
        try:
            return [{'id': t.id, 'name': t.name} for t in obj.topics.all()]
        except Exception:
            return []


class AdminStatsSerializer(serializers.Serializer):
    """Serializer for admin statistics"""
    total_users = serializers.IntegerField()
    total_sessions = serializers.IntegerField()
    completed_sessions = serializers.IntegerField()
    total_questions = serializers.IntegerField()
    total_topics = serializers.IntegerField()
    avg_communication_score = serializers.FloatField(allow_null=True)
    avg_technology_score = serializers.FloatField(allow_null=True)


class UserSessionSerializer(serializers.ModelSerializer):
    """Serializer for UserSession"""
    user_name = serializers.CharField(source='user.name', read_only=True)
    user_email = serializers.EmailField(source='user.email', read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)
    status = serializers.SerializerMethodField()

    class Meta:
        model = UserSession
        fields = [
            'id', 'user', 'user_username', 'user_name', 'user_email', 
            'login_time', 'last_activity', 'is_active', 'ip_address', 
            'device_info', 'status'
        ]

    def get_status(self, obj):
        from django.utils import timezone
        from datetime import timedelta
        if not obj.is_active:
            return "Offline"
        if timezone.now() - obj.last_activity < timedelta(minutes=10):
            return "Online"
        return "Idle"


class LoginHistorySerializer(serializers.ModelSerializer):
    """Serializer for LoginHistory"""
    user_name = serializers.CharField(source='user.name', read_only=True)
    user_email = serializers.EmailField(source='user.email', read_only=True)
    user_username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = LoginHistory
        fields = [
            'id', 'user', 'user_username', 'user_name', 'user_email', 
            'login_time', 'logout_time', 'status', 'ip_address', 'device_info'
        ]

