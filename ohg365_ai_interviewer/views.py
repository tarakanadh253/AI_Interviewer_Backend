from django.http import JsonResponse, HttpResponse

def health_check(request):
    """
    Simple health check view for the root path.
    Helpful for load balancers and to confirm the app is running.
    """
    return JsonResponse({
        "status": "online", 
        "message": "AI Interviewer Backend is running",
        "version": "1.0.0"
    })
