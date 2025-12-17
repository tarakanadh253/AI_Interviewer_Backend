
import os
import django
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.models import Answer

print(f"Answers: {Answer.objects.count()}")
recent = Answer.objects.order_by('-created_at')[:3]
for a in recent:
    print(f"ID={a.id} Sim={a.similarity_score} Acc={a.accuracy_score}")
