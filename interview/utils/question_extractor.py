"""
Utility to extract questions and answers from external links
"""
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def extract_questions_from_url(url: str) -> List[Dict[str, str]]:
    """
    Extract questions and answers from a URL.
    Returns a list of dicts with 'question' and 'answer' keys.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        
        # Try to find Q&A patterns
        questions = []
        
        # Pattern 1: Q: ... A: ... or Question: ... Answer: ...
        qa_pattern = re.compile(
            r'(?:Q|Question)[\s:]+(.+?)(?:A|Answer)[\s:]+(.+?)(?=(?:Q|Question)[\s:]|$)',
            re.IGNORECASE | re.DOTALL
        )
        matches = qa_pattern.findall(text)
        for match in matches:
            if len(match) == 2:
                questions.append({
                    'question': match[0].strip(),
                    'answer': match[1].strip()
                })
        
        # Pattern 2: Numbered questions (1. ... 2. ...)
        if not questions:
            numbered_pattern = re.compile(
                r'\d+[\.\)]\s*(.+?)(?=\d+[\.\)]|$)',
                re.DOTALL
            )
            numbered_items = numbered_pattern.findall(text)
            # Split into pairs (assuming even number of items)
            for i in range(0, len(numbered_items) - 1, 2):
                questions.append({
                    'question': numbered_items[i].strip(),
                    'answer': numbered_items[i + 1].strip() if i + 1 < len(numbered_items) else ''
                })
        
        # Pattern 3: Look for common question words followed by answers
        if not questions:
            # Find sentences ending with "?"
            question_sentences = re.findall(r'[^.!?]*\?[^.!?]*', text)
            # For each question, try to find the next paragraph as answer
            for i, q in enumerate(question_sentences[:10]):  # Limit to 10
                questions.append({
                    'question': q.strip(),
                    'answer': 'See reference material for detailed answer.'
                })
        
        # Clean up questions (remove very short ones)
        questions = [
            q for q in questions 
            if len(q['question']) > 10 and len(q['answer']) > 5
        ]
        
        return questions[:20]  # Limit to 20 questions per URL
        
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error extracting questions from {url}: {e}")
        return []


def extract_questions_from_links(links: List[str]) -> List[Dict[str, str]]:
    """
    Extract questions from multiple URLs.
    Returns a combined list of questions.
    """
    all_questions = []
    for link in links:
        if link.strip():
            questions = extract_questions_from_url(link.strip())
            all_questions.extend(questions)
    return all_questions


def process_link_question(question_obj) -> int:
    """
    Process a Question object with source_type='LINK'.
    Extracts questions from reference_links and creates Question objects.
    Returns the number of questions created.
    """
    from interview.models import Question
    
    if question_obj.source_type != 'LINK':
        return 0
    
    links = question_obj.get_reference_links_list()
    if not links:
        logger.warning(f"Question {question_obj.id} has no reference links")
        return 0
    
    extracted = extract_questions_from_links(links)
    if not extracted:
        logger.warning(f"No questions extracted from links for question {question_obj.id}")
        return 0
    
    created_count = 0
    for qa in extracted:
        # Create a new Question object for each extracted Q&A
        try:
            Question.objects.create(
                topic=question_obj.topic,
                source_type='MANUAL',  # Extracted questions are now manual
                question_text=qa['question'],
                ideal_answer=qa['answer'],
                is_active=True
            )
            created_count += 1
        except Exception as e:
            logger.error(f"Error creating question from extracted data: {e}")
    
    return created_count
