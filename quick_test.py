#!/usr/bin/env python
"""Quick test to diagnose the issue"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

try:
    django.setup()
    print("Django setup: OK")
    
    from interview.models import Question
    print("Import models: OK")
    
    questions = Question.objects.all()
    print(f"Total questions: {questions.count()}")
    
    if questions.exists():
        q = questions.first()
        print(f"First question ID: {q.id}")
        print(f"Has source_type attr: {hasattr(q, 'source_type')}")
        
        try:
            st = q.source_type
            print(f"source_type value: {st}")
        except Exception as e:
            print(f"ERROR accessing source_type: {e}")
            import traceback
            traceback.print_exc()
        
        # Test serializer
        from interview.serializers import AdminQuestionSerializer
        print("\nTesting serializer...")
        try:
            serializer = AdminQuestionSerializer(q)
            data = serializer.data
            print(f"Serialization: SUCCESS")
            print(f"Keys: {list(data.keys())}")
            print(f"source_type in data: {'source_type' in data}")
            print(f"source_type value: {data.get('source_type')}")
        except Exception as e:
            print(f"Serialization ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        # Test viewset
        print("\nTesting viewset...")
        from rest_framework.test import APIRequestFactory
        from interview.views import AdminQuestionViewSet
        
        factory = APIRequestFactory()
        request = factory.get('/api/admin/questions/')
        viewset = AdminQuestionViewSet()
        viewset.request = request
        viewset.format_kwarg = None
        
        try:
            response = viewset.list(request)
            print(f"ViewSet status: {response.status_code}")
            if response.status_code == 200:
                print(f"Questions returned: {len(response.data)}")
            else:
                print(f"Error: {response.data}")
        except Exception as e:
            print(f"ViewSet ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No questions in database")
        
except Exception as e:
    print(f"FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
