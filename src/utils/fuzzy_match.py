"""
Reusable fuzzy matching utilities for alias normalization.
"""
import unicodedata
import re
from typing import List, Tuple, Optional, Dict

from rapidfuzz import fuzz, process


def normalize_text(text: str) -> str:
    """
    Normalize text for matching:
    - Unicode NFKC normalization
    - Lowercase
    - Strip extra whitespace
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_accents(text: str) -> str:
    """Remove Vietnamese diacritics for accent-insensitive matching."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def exact_match(query: str, candidates: Dict[str, str]) -> Optional[str]:
    """
    Exact match after normalization.
    candidates: {display_name: backend_value}
    Returns backend_value if matched, else None.
    """
    q = normalize_text(query)
    for name, value in candidates.items():
        if normalize_text(name) == q or normalize_text(value) == q:
            return value
    return None


def fuzzy_match_best(
    query: str,
    candidates: Dict[str, str],
    threshold: int = 80,
) -> Optional[Tuple[str, str, float]]:
    """
    Fuzzy match query against candidate names.
    Returns (matched_name, backend_value, score) or None.
    """
    if not candidates:
        return None

    q = normalize_text(query)
    names = list(candidates.keys())
    normalized_names = [normalize_text(n) for n in names]

    result = process.extractOne(q, normalized_names, scorer=fuzz.WRatio)
    if result is None:
        return None

    matched_norm, score, idx = result
    if score >= threshold:
        name = names[idx]
        return (name, candidates[name], score)

    return None


def fuzzy_match_top_k(
    query: str,
    candidates: Dict[str, str],
    k: int = 5,
    threshold: int = 60,
) -> List[Tuple[str, str, float]]:
    """
    Return top-k fuzzy matches above threshold.
    Returns list of (matched_name, backend_value, score).
    """
    if not candidates:
        return []

    q = normalize_text(query)
    names = list(candidates.keys())
    normalized_names = [normalize_text(n) for n in names]

    results = process.extract(q, normalized_names, scorer=fuzz.WRatio, limit=k)
    output = []
    for matched_norm, score, idx in results:
        if score >= threshold:
            output.append((names[idx], candidates[names[idx]], score))
    return output


def accent_insensitive_match(
    query: str,
    candidates: Dict[str, str],
) -> Optional[str]:
    """
    Match after removing Vietnamese accents.
    Returns backend_value if matched, else None.
    """
    q = remove_accents(normalize_text(query))
    for name, value in candidates.items():
        if remove_accents(normalize_text(name)) == q:
            return value
        if remove_accents(normalize_text(value)) == q:
            return value
    return None
