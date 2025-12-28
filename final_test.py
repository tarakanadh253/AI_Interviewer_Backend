#!/usr/bin/env python
"""Final test of question bank endpoint"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

results = []

try:
    django.setup()
    results.append("Django setup: SUCCESS")
    
    # Test the endpoint
    from rest_framework.test import APIRequestFactory
    from interview.views import AdminQuestionViewSet
    
    factory = APIRequestFactory()
    request = factory.get('/api/admin/questions/')
    viewset = AdminQuestionViewSet()
    viewset.request = request
    viewset.format_kwarg = None
    
    response = viewset.list(request)
    results.append(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        results.append(f"SUCCESS! Questions returned: {len(response.data)}")
        if response.data:
            first_q = response.data[0]
            results.append(f"First question has source_type: {'source_type' in first_q}")
            results.append(f"First question source_type value: {first_q.get('source_type')}")
            results.append(f"First question keys: {', '.join(first_q.keys())}")
    else:
        results.append(f"ERROR: {response.data}")
        
except Exception as e:
    results.append(f"ERROR: {e}")
    import traceback
    results.append(traceback.format_exc())

# Write results
with open('final_test_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

# Print to console
for line in results:
    print(line)
