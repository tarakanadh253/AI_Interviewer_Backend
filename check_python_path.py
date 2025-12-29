#!/usr/bin/env python
"""Check Python path and package locations"""
import sys
import os

print("="*60)
print("Python Environment Information")
print("="*60)

print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\nPython path (sys.path):")
for i, path in enumerate(sys.path, 1):
    print(f"  {i}. {path}")

print("\nTrying to import bs4...")
try:
    import bs4
    print(f"  ✓ SUCCESS! bs4 imported from: {bs4.__file__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    
    # Check if D:\python-packages exists
    custom_path = r"D:\python-packages"
    if os.path.exists(custom_path):
        print(f"\n  Found custom packages directory: {custom_path}")
        if custom_path not in sys.path:
            print(f"  ⚠ WARNING: {custom_path} is NOT in sys.path!")
            print(f"  Adding it to sys.path...")
            sys.path.insert(0, custom_path)
            try:
                import bs4
                print(f"  ✓ SUCCESS after adding to path! bs4 imported from: {bs4.__file__}")
            except ImportError:
                print(f"  ✗ Still failed after adding to path")

print("\nTrying to import requests...")
try:
    import requests
    print(f"  ✓ SUCCESS! requests imported from: {requests.__file__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")

print("\n" + "="*60)


