#!/usr/bin/env python
"""Check actual database schema"""
import os
import sys
import django

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from django.db import connection

print("=" * 70)
print("CHECKING ACTUAL DATABASE SCHEMA")
print("=" * 70)

with connection.cursor() as cursor:
    cursor.execute("PRAGMA table_info(user_profiles)")
    cols = cursor.fetchall()
    
    print("\nCurrent columns in user_profiles:")
    for col in cols:
        print(f"  {col[1]}: {col[2]} (NOT NULL: {col[3]}, DEFAULT: {col[4]})")
    
    col_names = [col[1] for col in cols]
    print(f"\nColumn names: {', '.join(col_names)}")
    
    # Check what's missing
    required = ['id', 'username', 'password', 'email', 'is_active', 'access_type', 'has_used_trial', 'created_at', 'updated_at']
    missing = [c for c in required if c not in col_names]
    
    if missing:
        print(f"\n✗ MISSING COLUMNS: {', '.join(missing)}")
    else:
        print("\n✓ All required columns exist!")
