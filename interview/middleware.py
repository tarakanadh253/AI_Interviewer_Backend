import logging
import traceback
from django.http import JsonResponse
from django.conf import settings

logger = logging.getLogger(__name__)

class JsonExceptionMiddleware:
    """
    Middleware to catch all unhandled exceptions and return them as JSON.
    This ensures that 500 errors provide useful debug info to the frontend
    instead of opaque HTML pages.
    """
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            response = self.get_response(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled Exception in Middleware: {e}", exc_info=True)
            return self.handle_exception(request, e)
            
    def process_exception(self, request, exception):
        # This catches exceptions raised by views
        return self.handle_exception(request, exception)
        
    def handle_exception(self, request, exception):
        # Get DB host for debugging
        db_host = 'Unknown'
        try:
            db_host = settings.DATABASES['default'].get('HOST', 'Unknown')
        except:
            pass
            
        return JsonResponse({
            'error': str(exception),
            'error_type': exception.__class__.__name__,
            'db_host_debug': db_host,
            'detail': traceback.format_exc()
        }, status=500)


class UserActivityMiddleware:
    """Middleware to update last_activity for UserSession"""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            try:
                from .models import UserProfile, UserSession
                from django.utils import timezone
                
                user_profile = UserProfile.objects.filter(username=request.user.username).first()
                if user_profile:
                    session_key = request.session.session_key
                    if session_key:
                        UserSession.objects.filter(
                            user=user_profile, 
                            session_token=session_key
                        ).update(last_activity=timezone.now(), is_active=True)
            except Exception:
                # Silently fail to not block requests
                pass
                
        response = self.get_response(request)
        return response
