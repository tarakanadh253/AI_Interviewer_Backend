from rest_framework.authentication import SessionAuthentication

class CsrfExemptSessionAuthentication(SessionAuthentication):
    """
    Custom SessionAuthentication that bypasses CSRF checks.
    Useful for cross-origin usage where 'X-CSRFToken' header cannot be easily set by the client.
    WARNING: Disables CSRF protection. Use with caution.
    """
    def enforce_csrf(self, request):
        return  # Skip CSRF check
