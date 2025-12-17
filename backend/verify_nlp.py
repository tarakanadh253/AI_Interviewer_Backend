
import os
import sys

print("Checking NLP environment...")
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    print("Attempting to import sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print("Success: sentence_transformers imported.")
except ImportError as e:
    print(f"ERROR: Could not import sentence_transformers: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: Unexpected error importing sentence_transformers: {e}")
    sys.exit(1)

try:
    print("Attempting to load model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Success: Model loaded.")
    
    print("Running test encoding...")
    embedding = model.encode(["This is a test."])
    print(f"Success: Encoding shape: {embedding.shape}")
except Exception as e:
    print(f"ERROR: Failed to load or run model: {e}")
    sys.exit(1)

print("NLP Test Completed Successfully.")
