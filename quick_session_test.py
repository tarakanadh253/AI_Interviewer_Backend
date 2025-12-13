import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

output = []

try:
    django.setup()
    output.append("Django OK")
    
    from interview.models import UserProfile, Topic
    from rest_framework.test import APIRequestFactory
    from interview.views import InterviewSessionViewSet
    
    user = UserProfile.objects.first()
    topics = Topic.objects.all()[:2]
    topic_ids = [t.id for t in topics]
    
    output.append(f"User: {user.username if user else 'NONE'}")
    output.append(f"Topics: {topic_ids}")
    
    factory = APIRequestFactory()
    request = factory.post('/api/sessions/', {
        'username': user.username if user else '',
        'topic_ids': topic_ids
    }, format='json')
    
    viewset = InterviewSessionViewSet()
    viewset.request = request
    viewset.format_kwarg = None
    
    output.append("Calling create...")
    response = viewset.create(request)
    output.append(f"Status: {response.status_code}")
    if response.status_code != 201:
        output.append(f"Error: {response.data}")
        
except Exception as e:
    output.append(f"EXCEPTION: {e}")
    import traceback
    output.extend(traceback.format_exc().split('\n'))

with open('session_test_output.txt', 'w') as f:
    f.write('\n'.join(output))

print('\n'.join(output))
