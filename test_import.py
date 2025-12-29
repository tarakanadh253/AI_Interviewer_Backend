#!/usr/bin/env python
"""Quick test to verify imports work"""
import sys

print("Testing imports...")
print("-" * 40)

# Test bs4
try:
    import bs4
    from bs4 import BeautifulSoup
    print("✓ bs4 imported successfully")
    print(f"  Location: {bs4.__file__}")
except Exception as e:
    print(f"✗ bs4 import failed: {e}")
    import traceback
    traceback.print_exc()

# Test requests
try:
    import requests
    print(f"✓ requests imported successfully")
    print(f"  Version: {requests.__version__}")
    print(f"  Location: {requests.__file__}")
except Exception as e:
    print(f"✗ requests import failed: {e}")
    import traceback
    traceback.print_exc()

print("-" * 40)
print("Import test complete!")
