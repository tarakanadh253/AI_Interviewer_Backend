"""
Advanced NLP-based answer evaluation system
Uses semantic similarity, keyword extraction, and multiple scoring factors
"""
import re
import string
import json
from typing import Dict, List, Tuple
import logging

import os
logger = logging.getLogger(__name__)

# Lazy loading imports to prevent immediate crash on startup
# Lazy loading imports to prevent immediate crash on startup
NLP_AVAILABLE = True # We assume it is available and let the lazy import fail if not
    
# Global variable to hold the model, but initialized lazily
_nlp_model = None

def get_model():
    """Lazily load the NLP model to prevent memory spikes on startup"""
    global _nlp_model
    global NLP_AVAILABLE

    if _nlp_model is None:
        try:
            # On Render Free Tier, heavy ML models often cause OOM (Out Of Memory) kills.
            # We default to disabling the heavy model on Render to ensure reliability.
            is_on_render = os.environ.get('RENDER')
            # Check if we should disable heavy NLP (default to True if not specified to be safe, or check env)
            # For this debugging session, let's make it robust:
            enable_heavy_nlp = os.environ.get('ENABLE_HEAVY_NLP', 'True').lower() == 'true'

            if is_on_render and not enable_heavy_nlp:
                logger.warning("Running on Render: Heavy NLP model disabled to prevent OOM. Using keyword fallback.")
                NLP_AVAILABLE = False
                return None
            
            # Additional check for local development to avoid hanging if library is missing
            try:
                import sentence_transformers
            except ImportError:
                 logger.warning("sentence_transformers not installed. Using keyword fallback.")
                 NLP_AVAILABLE = False
                 return None

            logger.info("Lazily loading NLP model 'all-MiniLM-L6-v2'...")
            from sentence_transformers import SentenceTransformer
            _nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Warmup
            _nlp_model.encode(["test"], normalize_embeddings=True)
            logger.info("NLP model loaded successfully")
        except ImportError as e:
            logger.error(f"NLP libraries not available: {e}")
            NLP_AVAILABLE = False
            return None
        except Exception as e:
            logger.error(f"Failed to load NLP model: {e}")
            NLP_AVAILABLE = False # Don't raise, just fallback
            return None
            
    return _nlp_model
            
    return _nlp_model

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        _stopwords = set(stopwords.words('english'))
        _lemmatizer = WordNetLemmatizer()
    except Exception as e:
        logger.warning(f"NLTK setup issue: {e}")
        NLTK_AVAILABLE = False
        _stopwords = set()
        _lemmatizer = None
except ImportError:
    NLTK_AVAILABLE = False
    _stopwords = set()
    _lemmatizer = None
    logger.warning("NLTK not available. Using basic stopwords.")

# Fallback stopwords if NLTK not available
FALLBACK_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
    'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
    'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
    'would', 'make', 'like', 'into', 'him', 'two', 'more',
    'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
    'come', 'made', 'may', 'part', 'over', 'new', 'sound', 'take',
    'only', 'little', 'work', 'know', 'place', 'year', 'live', 'me',
    'back', 'give', 'most', 'thing', 'our', 'just',
    'name', 'good', 'sentence', 'man', 'think', 'say', 'great', 'where',
    'help', 'through', 'much', 'before', 'line', 'right', 'too', 'mean',
    'old', 'any', 'same', 'tell', 'boy', 'follow', 'came', 'want',
    'show', 'also', 'around', 'form', 'three', 'small', 'set', 'put',
    'end', 'does', 'another', 'well', 'large', 'must', 'big', 'even',
    'such', 'because', 'turn', 'here', 'why', 'ask', 'went', 'men',
    'read', 'need', 'land', 'different', 'home', 'us', 'move', 'try',
    'kind', 'hand', 'picture', 'again', 'change', 'off', 'play', 'spell',
    'air', 'away', 'animal', 'house', 'point', 'page', 'letter', 'mother',
    'answer', 'found', 'study', 'still', 'learn', 'should', 'america',
    'world', 'high', 'every', 'near', 'add', 'food', 'between', 'own',
    'below', 'country', 'plant', 'last', 'school', 'father', 'keep',
    'tree', 'never', 'start', 'city', 'earth', 'eye', 'light', 'thought',
    'head', 'under', 'story', 'saw', 'left', 'don', 'few', 'while',
    'along', 'might', 'close', 'something', 'seem', 'next', 'hard',
    'open', 'example', 'begin', 'life', 'always', 'those', 'both',
    'paper', 'together', 'got', 'group', 'often', 'run', 'important',
    'until', 'children', 'side', 'feet', 'car', 'mile', 'night', 'walk',
    'white', 'sea', 'began', 'grow', 'took', 'river', 'four', 'carry',
    'state', 'once', 'book', 'hear', 'stop', 'without', 'second',
    'later', 'miss', 'idea', 'enough', 'eat', 'face', 'watch', 'far',
    'indian', 'really', 'almost', 'let', 'above', 'girl', 'sometimes',
    'mountain', 'cut', 'young', 'talk', 'soon', 'list', 'song', 'leave',
    'family'
}

# Use NLTK stopwords if available, otherwise fallback
STOPWORDS = _stopwords if NLTK_AVAILABLE and _stopwords else FALLBACK_STOPWORDS

# Filler words that indicate poor communication
FILLER_WORDS = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically', 'literally', 'sort of', 'kind of'}


def extract_keywords_advanced(text: str, min_length: int = 3) -> List[str]:
    """
    Advanced keyword extraction using NLP techniques.
    
    Args:
        text: Input text
        min_length: Minimum length of keywords to include
    
    Returns:
        List of unique keywords (lemmatized and lowercased)
    """
    if not text or not text.strip():
        return []
    
    # Use NLTK tokenization if available
    if NLTK_AVAILABLE and _lemmatizer:
        try:
            tokens = word_tokenize(text.lower())
            # Lemmatize and filter
            keywords = []
            for token in tokens:
                if len(token) >= min_length and token.isalpha():
                    # Remove stopwords
                    if token not in STOPWORDS:
                        # Lemmatize
                        lemmatized = _lemmatizer.lemmatize(token)
                        keywords.append(lemmatized)
            return list(dict.fromkeys(keywords))  # Remove duplicates while preserving order
        except Exception as e:
            logger.warning(f"Error in advanced keyword extraction: {e}. Using basic extraction.")
            return extract_keywords_basic(text, min_length)
    
    # Fallback to basic extraction if NLTK not available
    return extract_keywords_basic(text, min_length)


def extract_keywords_basic(text: str, min_length: int = 3) -> List[str]:
    """
    Basic keyword extraction (fallback method).
    
    Args:
        text: Input text
        min_length: Minimum length of keywords to include
    
    Returns:
        List of unique keywords (lowercased)
    """
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Filter: remove stopwords, single letters, and short words
    keywords = [
        word for word in words
        if len(word) >= min_length
        and word not in STOPWORDS
        and word.isalpha()
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)
    
    return unique_keywords


def calculate_semantic_similarity(user_answer: str, ideal_answer: str) -> float:
    """
    Calculate semantic similarity using sentence transformers.
    Falls back to keyword-based similarity if NLP not available.
    
    Args:
        user_answer: User's answer text
        ideal_answer: Ideal answer text
    
    Returns:
        Similarity score between 0 and 1
    """
    if not user_answer or not ideal_answer:
        return 0.0
    
    # Use NLP model if available
    try:
        model = get_model()
        if NLP_AVAILABLE and model is not None:
            # Get embeddings (normalized by default in sentence-transformers)
            embeddings = model.encode([user_answer, ideal_answer], normalize_embeddings=True)
            
            # Calculate cosine similarity
            # For normalized embeddings, cosine similarity is already in range [0, 1]
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Apply brevity penalty
            # If the user's answer is significantly shorter than the ideal answer, penalize the score
            # This prevents short, keyword-heavy answers from getting high scores
            user_len = len(user_answer.split())
            ideal_len = len(ideal_answer.split())
            
            if ideal_len > 0 and user_len < ideal_len:
                # Calculate ratio, but don't penalize too harshly for concise correct answers
                # If user answer is less than 50% of ideal length, apply penalty
                ratio = user_len / ideal_len
                if ratio < 0.5:
                    # Exponential penalty similar to BLEU score
                    # Limits the maximum possible score for very short answers
                    penalty = np.exp(1 - (0.5 / max(ratio, 0.1)))
                    # Blend the penalty: 70% raw similarity, 30% penalized
                    similarity = similarity * penalty
            
            # Clamp to 0-1 range (should already be in this range for normalized embeddings)
            # Cosine similarity with normalized vectors ranges from -1 to 1, but typically 0 to 1
            # If negative, set to 0; otherwise use as-is
            similarity = max(0.0, min(1.0, float(similarity)))
            
            logger.debug(f"Semantic similarity calculated: {similarity:.4f}")
            return similarity
    except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}. Using fallback.", exc_info=True)
    
    # Fallback to keyword similarity if NLP is unavailable or fails
    logger.info("NLP model unavailable or failed. Falling back to keyword-based similarity.")
    return calculate_keyword_similarity(user_answer, ideal_answer)


def calculate_keyword_similarity(user_answer: str, ideal_answer: str) -> float:
    """
    Calculate similarity score based on keyword overlap (fallback method).
    
    Args:
        user_answer: User's answer text
        ideal_answer: Ideal answer text
    
    Returns:
        Similarity score between 0 and 1
    """
    ideal_keywords = extract_keywords_advanced(ideal_answer)
    user_keywords = extract_keywords_advanced(user_answer)
    
    if not ideal_keywords:
        return 1.0  # If no keywords in ideal answer, consider it a match
    
    # Count matched keywords
    ideal_set = set(ideal_keywords)
    user_set = set(user_keywords)
    matched = ideal_set.intersection(user_set)
    
    # Calculate Jaccard similarity (intersection over union)
    union = ideal_set.union(user_set)
    if not union:
        return 0.0
    
    jaccard_similarity = len(matched) / len(union)
    
    # Also calculate coverage (how much of ideal answer is covered)
    coverage = len(matched) / len(ideal_set) if ideal_set else 0.0
    
    # Weighted combination: 60% coverage, 40% Jaccard
    similarity = (0.6 * coverage) + (0.4 * jaccard_similarity)
    
    return min(1.0, max(0.0, similarity))


def calculate_completeness_score(user_answer: str, ideal_answer: str) -> float:
    """
    Calculate how complete the user's answer is compared to ideal answer.
    Based on key concepts coverage.
    
    Args:
        user_answer: User's answer text
        ideal_answer: Ideal answer text
    
    Returns:
        Completeness score between 0 and 1
    """
    ideal_keywords = extract_keywords_advanced(ideal_answer, min_length=4)  # Longer keywords for concepts
    user_keywords = extract_keywords_advanced(user_answer, min_length=4)
    
    if not ideal_keywords:
        return 1.0
    
    ideal_set = set(ideal_keywords)
    user_set = set(user_keywords)
    matched = ideal_set.intersection(user_set)
    
    # Completeness is how many key concepts are covered
    completeness = len(matched) / len(ideal_set)
    
    return min(1.0, max(0.0, completeness))


def calculate_communication_score(user_answer: str) -> float:
    """
    Calculate communication subscore based on multiple factors:
    - Length and structure
    - Punctuation usage
    - Presence of filler words
    - Sentence coherence
    - Vocabulary diversity
    
    Args:
        user_answer: User's answer text
    
    Returns:
        Communication score between 0 and 1
    """
    if not user_answer or not user_answer.strip():
        return 0.0
    
    # Normalize text
    text = user_answer.strip()
    text_lower = text.lower()
    
    # Word count
    if NLTK_AVAILABLE:
        try:
            words = word_tokenize(text_lower)
            sentences = sent_tokenize(text)
        except:
            words = text_lower.split()
            sentences = re.split(r'[.!?]+', text)
    else:
        words = text_lower.split()
        sentences = re.split(r'[.!?]+', text)
    
    word_count = len([w for w in words if w.isalpha()])
    sentence_count = len([s for s in sentences if s.strip()])
    
    if word_count == 0:
        return 0.0
    
    scores = {}
    
    # 1. Length score (optimal range: 30-150 words) - 25% weight
    if word_count < 15:
        length_score = (word_count / 15.0) * 0.5  # Very short answers heavily penalized
    elif word_count <= 150:
        # Optimal range: 30-100 words gets full score
        if 30 <= word_count <= 100:
            length_score = 1.0
        elif word_count < 30:
            length_score = 0.5 + (word_count / 30.0) * 0.5
        else:
            length_score = 1.0 - ((word_count - 100) / 50.0) * 0.3  # Slight penalty for very long
        length_score = max(0.3, length_score)
    else:
        length_score = 0.7 - min((word_count - 150) / 100.0, 0.3)  # Too long penalized
    
    scores['length'] = max(0.0, min(1.0, length_score)) * 0.25
    
    # 2. Sentence structure score - 20% weight
    if sentence_count > 0:
        avg_words_per_sentence = word_count / sentence_count
        # Optimal: 10-25 words per sentence
        if 10 <= avg_words_per_sentence <= 25:
            structure_score = 1.0
        elif avg_words_per_sentence < 10:
            structure_score = avg_words_per_sentence / 10.0
        else:
            structure_score = max(0.5, 1.0 - ((avg_words_per_sentence - 25) / 20.0))
    else:
        structure_score = 0.3  # No sentence structure detected
    
    scores['structure'] = structure_score * 0.20
    
    # 3. Punctuation score - 15% weight
    punctuation_chars = sum(1 for char in text if char in '.,!?;:')
    punctuation_score = min(1.0, punctuation_chars / max(word_count / 15, 1))
    scores['punctuation'] = punctuation_score * 0.15
    
    # 4. Vocabulary diversity - 15% weight
    unique_words = len(set(words))
    diversity_ratio = unique_words / word_count if word_count > 0 else 0
    # Good diversity: 0.5-0.8 ratio
    if 0.5 <= diversity_ratio <= 0.8:
        diversity_score = 1.0
    elif diversity_ratio < 0.5:
        diversity_score = diversity_ratio / 0.5
    else:
        diversity_score = max(0.7, 1.0 - ((diversity_ratio - 0.8) / 0.2))
    
    scores['diversity'] = diversity_score * 0.15
    
    # 5. Filler word penalty - 10% weight
    filler_count = sum(1 for filler in FILLER_WORDS if filler in text_lower)
    filler_penalty = min(0.5, filler_count * 0.1)  # Max penalty of 0.5
    scores['filler_penalty'] = -filler_penalty * 0.10
    
    # 6. Coherence score (basic) - 15% weight
    # Check for transition words and logical flow indicators
    transition_words = {'first', 'second', 'then', 'next', 'finally', 'however', 'therefore', 
                       'because', 'since', 'although', 'furthermore', 'moreover', 'additionally'}
    transition_count = sum(1 for word in words if word in transition_words)
    coherence_score = min(1.0, transition_count / max(word_count / 50, 1))
    scores['coherence'] = coherence_score * 0.15
    
    # Combine all scores
    total_score = sum(scores.values())
    
    return max(0.0, min(1.0, total_score))


def calculate_accuracy_score(user_answer: str, ideal_answer: str) -> float:
    """
    Calculate accuracy score based on semantic similarity and completeness.
    This provides a proper, non-random score by matching user's answer
    with the ideal answer using NLP techniques.
    
    Args:
        user_answer: User's answer text (transcribed from voice)
        ideal_answer: Ideal/predefined answer text
    
    Returns:
        Accuracy score between 0 and 1 (properly calculated, not random)
    """
    if not user_answer or not ideal_answer:
        return 0.0
    
    # Semantic similarity (how well the meaning matches) - Uses NLP model
    semantic_sim = calculate_semantic_similarity(user_answer, ideal_answer)
    
    # Completeness (how many key concepts are covered)
    completeness = calculate_completeness_score(user_answer, ideal_answer)
    
    # Weighted combination: 70% semantic similarity, 30% completeness
    # This ensures both meaning match and concept coverage are considered
    accuracy = (0.7 * semantic_sim) + (0.3 * completeness)
    
    # Ensure score is in valid range
    accuracy = min(1.0, max(0.0, accuracy))
    
    logger.debug(f"Accuracy score: semantic={semantic_sim:.4f}, completeness={completeness:.4f}, final={accuracy:.4f}")
    
    return accuracy


def evaluate_answer(user_answer: str, ideal_answer: str) -> Dict:
    """
    Comprehensive evaluation of user's answer against ideal answer.
    Uses NLP techniques for semantic analysis and multiple scoring factors.
    
    This function properly matches user's transcribed voice answer with the
    predefined ideal answer using advanced NLP models (sentence transformers)
    to provide accurate, non-random scores.
    
    Args:
        user_answer: User's answer text (transcribed from voice)
        ideal_answer: Ideal/predefined answer text
    
    Returns:
        Dictionary with detailed evaluation:
        - similarity_score: float (0-1) - Semantic similarity
        - accuracy_score: float (0-1) - Overall accuracy (semantic + completeness)
        - completeness_score: float (0-1) - How complete the answer is
        - communication_subscore: float (0-1) - Communication quality
        - matched_keywords: List[str] - Keywords found in user answer
        - missing_keywords: List[str] - Important keywords missing
        - score_breakdown: Dict - Detailed breakdown of scores
    """
    if user_answer is None:
        logger.info("None user answer provided, returning zero scores")
        return {
            "similarity_score": 0.0,
            "accuracy_score": 0.0,
            "completeness_score": 0.0,
            "communication_subscore": 0.0,
            "matched_keywords": [],
            "missing_keywords": [],
            "score_breakdown": {
                "semantic_similarity": 0.0,
                "keyword_coverage": 0.0,
                "communication_quality": 0.0
            }
        }
    
    if not user_answer or not user_answer.strip():
        logger.info("Empty user answer provided, returning zero scores")
        return {
            "similarity_score": 0.0,
            "accuracy_score": 0.0,
            "completeness_score": 0.0,
            "communication_subscore": 0.0,
            "matched_keywords": [],
            "missing_keywords": [],
            "score_breakdown": {
                "semantic_similarity": 0.0,
                "keyword_coverage": 0.0,
                "communication_quality": 0.0
            }
        }
    
    if not ideal_answer or not ideal_answer.strip():
        logger.warning("Empty ideal answer provided, cannot evaluate properly")
        return {
            "similarity_score": 0.0,
            "accuracy_score": 0.0,
            "completeness_score": 0.0,
            "communication_subscore": calculate_communication_score(user_answer),
            "matched_keywords": [],
            "missing_keywords": [],
            "score_breakdown": {
                "semantic_similarity": 0.0,
                "keyword_coverage": 0.0,
                "communication_quality": calculate_communication_score(user_answer)
            }
        }
    
    # Log evaluation start
    logger.info(f"Evaluating answer: user_answer length={len(user_answer)}, ideal_answer length={len(ideal_answer)}")
    logger.info(f"NLP model available: {NLP_AVAILABLE and _nlp_model is not None}")
    
    # Extract keywords
    ideal_keywords = extract_keywords_advanced(ideal_answer, min_length=3)
    user_keywords = extract_keywords_advanced(user_answer, min_length=3)
    
    ideal_set = set(ideal_keywords)
    user_set = set(user_keywords)
    
    # Find matched and missing keywords
    matched_keywords = list(ideal_set.intersection(user_set))
    missing_keywords = list(ideal_set - user_set)
    
    # Calculate various scores using NLP
    semantic_similarity = calculate_semantic_similarity(user_answer, ideal_answer)
    completeness_score = calculate_completeness_score(user_answer, ideal_answer)
    accuracy_score = calculate_accuracy_score(user_answer, ideal_answer)
    communication_subscore = calculate_communication_score(user_answer)
    
    # Keyword coverage
    keyword_coverage = len(matched_keywords) / len(ideal_set) if ideal_set else 1.0
    
    # Log results for debugging
    logger.info(f"Evaluation results - Semantic: {semantic_similarity:.4f}, "
                f"Accuracy: {accuracy_score:.4f}, Completeness: {completeness_score:.4f}, "
                f"Communication: {communication_subscore:.4f}")
    
    return {
        "similarity_score": semantic_similarity,  # Keep for backward compatibility
        "accuracy_score": accuracy_score,  # Overall accuracy (semantic + completeness)
        "completeness_score": completeness_score,  # Completeness metric
        "communication_subscore": communication_subscore,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "score_breakdown": {
            "semantic_similarity": semantic_similarity,
            "keyword_coverage": keyword_coverage,
            "completeness": completeness_score,
            "communication_quality": communication_subscore,
            "accuracy": accuracy_score
        }
    }
