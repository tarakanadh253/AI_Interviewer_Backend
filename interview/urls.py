from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    UserProfileViewSet, TopicViewSet, QuestionViewSet,
    InterviewSessionViewSet, AnswerViewSet, RoundViewSet,
    AdminQuestionViewSet, AdminInterviewSessionViewSet, AdminTopicViewSet,
    AdminUserViewSet, AdminRoundViewSet
)

router = DefaultRouter()
router.register(r'users', UserProfileViewSet, basename='user')
router.register(r'topics', TopicViewSet, basename='topic')
router.register(r'rounds', RoundViewSet, basename='round')
router.register(r'questions', QuestionViewSet, basename='question')
router.register(r'sessions', InterviewSessionViewSet, basename='session')
router.register(r'answers', AnswerViewSet, basename='answer')

# Admin routes
router.register(r'admin/users', AdminUserViewSet, basename='admin-user')
router.register(r'admin/topics', AdminTopicViewSet, basename='admin-topic')
router.register(r'admin/rounds', AdminRoundViewSet, basename='admin-round')
router.register(r'admin/questions', AdminQuestionViewSet, basename='admin-question')
router.register(r'admin/sessions', AdminInterviewSessionViewSet, basename='admin-session')

urlpatterns = [
    path('', include(router.urls)),
]

