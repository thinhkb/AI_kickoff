"""
Normalizes the input question before routing.
"""
import re
import unicodedata
from src.preprocess.time_normalizer import normalize_time_expressions
from src.preprocess.text_cleaner import clean_text


def normalize_question(question: str) -> str:
    """
    Full question normalization pipeline:
    1. Unicode normalization (NFKC)
    2. Whitespace cleanup
    3. Time phrase normalization
    4. Text cleaning
    """
    if not question:
        return ""

    # Unicode NFKC normalization
    q = unicodedata.normalize("NFKC", question)

    # Basic whitespace cleanup
    q = q.strip()
    q = re.sub(r"\s+", " ", q)

    # Clean text (punctuation, special chars)
    q = clean_text(q)

    # Normalize time expressions
    q = normalize_time_expressions(q)

    return q
