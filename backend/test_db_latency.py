import time
import os
import django
import sys
from django.db import connection

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

def check_latency():
    print("Testing database connection...")
    start_time = time.time()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        print(f"Connection Successful! Result: {row}")
        print(f"Query taken: {duration:.2f} ms")
    except Exception as e:
        print(f"Connection Failed: {e}")

if __name__ == "__main__":
    check_latency()
