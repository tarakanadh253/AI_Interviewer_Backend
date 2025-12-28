"""
Test script to verify the evaluation system is working properly
and using NLP models for accurate scoring (not random/mock scores)
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ohg365_ai_interviewer.settings')
django.setup()

from interview.utils.evaluation import (
    evaluate_answer,
    calculate_semantic_similarity,
    calculate_accuracy_score,
    NLP_AVAILABLE,
    _nlp_model
)

def test_evaluation():
    """Test the evaluation system with sample answers"""
    
    print("=" * 60)
    print("Testing Answer Evaluation System")
    print("=" * 60)
    
    # Check NLP model status
    print(f"\nNLP Model Status:")
    print(f"  - NLP Available: {NLP_AVAILABLE}")
    print(f"  - Model Loaded: {_nlp_model is not None}")
    
    if not NLP_AVAILABLE or _nlp_model is None:
        print("\n⚠️  WARNING: NLP model is not available!")
        print("   The system will use fallback keyword matching.")
        print("   Please ensure sentence-transformers is installed:")
        print("   pip install sentence-transformers scikit-learn numpy")
    else:
        print("  ✅ NLP model is ready for use")
    
    # Test cases
    test_cases = [
        {
            "name": "Exact Match",
            "user_answer": "Python is a high-level programming language known for its simplicity and readability.",
            "ideal_answer": "Python is a high-level programming language known for its simplicity and readability."
        },
        {
            "name": "Similar Meaning",
            "user_answer": "Python is a programming language that is easy to learn and read.",
            "ideal_answer": "Python is a high-level programming language known for its simplicity and readability."
        },
        {
            "name": "Partial Match",
            "user_answer": "Python is a programming language.",
            "ideal_answer": "Python is a high-level programming language known for its simplicity and readability."
        },
        {
            "name": "Different Answer",
            "user_answer": "Java is an object-oriented programming language.",
            "ideal_answer": "Python is a high-level programming language known for its simplicity and readability."
        },
        {
            "name": "Empty Answer",
            "user_answer": "",
            "ideal_answer": "Python is a high-level programming language known for its simplicity and readability."
        }
    ]
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"  User Answer: {test['user_answer'][:60]}...")
        print(f"  Ideal Answer: {test['ideal_answer'][:60]}...")
        
        # Evaluate
        result = evaluate_answer(test['user_answer'], test['ideal_answer'])
        
        print(f"\n  Results:")
        print(f"    - Semantic Similarity: {result['similarity_score']:.4f}")
        print(f"    - Accuracy Score: {result['accuracy_score']:.4f}")
        print(f"    - Completeness Score: {result['completeness_score']:.4f}")
        print(f"    - Communication Score: {result['communication_subscore']:.4f}")
        print(f"    - Matched Keywords: {len(result['matched_keywords'])}")
        print(f"    - Missing Keywords: {len(result['missing_keywords'])}")
        
        # Verify scores are reasonable
        if test['name'] == "Exact Match":
            if result['accuracy_score'] < 0.8:
                print(f"    ⚠️  WARNING: Exact match should have high accuracy (>0.8)")
        elif test['name'] == "Different Answer":
            if result['accuracy_score'] > 0.5:
                print(f"    ⚠️  WARNING: Different answer should have low accuracy (<0.5)")
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Evaluation Test Complete")
    print("=" * 60)
    print("\n✅ If scores vary appropriately based on answer similarity,")
    print("   the evaluation system is working correctly!")
    print("\n⚠️  If scores are random or don't match expectations,")
    print("   check that the NLP model is properly loaded.")

if __name__ == "__main__":
    test_evaluation()

