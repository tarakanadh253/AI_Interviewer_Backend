#!/usr/bin/env python
"""Test the question bank endpoint"""
import os, sys, django
sys.path.insert(0, '.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')

output = []

try:
    django.setup()
    output.append("✓ Django setup: OK")
    
    # Check database
    import sqlite3
    conn = sqlite3.connect('db.sqlite3')
    c = conn.cursor()
    c.execute('PRAGMA table_info(questions)')
    cols = [r[1] for r in c.fetchall()]
    conn.close()
    output.append(f"✓ Database columns: {', '.join(cols)}")
    output.append(f"✓ Has source_type: {'source_type' in cols}")
    output.append(f"✓ Has reference_links: {'reference_links' in cols}")
    
    # Test model
    from interview.models import Question
    questions = Question.objects.all()
    output.append(f"✓ Questions in DB: {questions.count()}")
    
    if questions.exists():
        q = questions.first()
        output.append(f"✓ First question ID: {q.id}")
        
        # Test accessing source_type
        try:
            st = q.source_type
            output.append(f"✓ source_type value: {st}")
        except Exception as e:
            output.append(f"✗ ERROR accessing source_type: {e}")
        
        # Test serializer
        from interview.serializers import AdminQuestionSerializer
        try:
            serializer = AdminQuestionSerializer(q)
            data = serializer.data
            output.append(f"✓ Serialization: SUCCESS")
            output.append(f"✓ source_type in data: {'source_type' in data}")
            output.append(f"✓ source_type value: {data.get('source_type')}")
        except Exception as e:
            output.append(f"✗ Serialization ERROR: {e}")
            import traceback
            output.append(traceback.format_exc())
        
        # Test viewset
        from rest_framework.test import APIRequestFactory
        from interview.views import AdminQuestionViewSet
        
        factory = APIRequestFactory()
        request = factory.get('/api/admin/questions/')
        viewset = AdminQuestionViewSet()
        viewset.request = request
        viewset.format_kwarg = None
        
        try:
            response = viewset.list(request)
            output.append(f"✓ ViewSet status: {response.status_code}")
            if response.status_code == 200:
                output.append(f"✓ Questions returned: {len(response.data)}")
            else:
                output.append(f"✗ Error response: {response.data}")
        except Exception as e:
            output.append(f"✗ ViewSet ERROR: {e}")
            import traceback
            output.append(traceback.format_exc())
    else:
        output.append("⚠ No questions in database")
        
except Exception as e:
    output.append(f"✗ FATAL ERROR: {e}")
    import traceback
    output.append(traceback.format_exc())

# Write to file
with open('test_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

# Also print
print('\n'.join(output))
