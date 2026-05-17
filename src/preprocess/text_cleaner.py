"""
Utility functions for text cleanup and canonicalization.
"""
import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and canonicalize text:
    - Normalize unicode
    - Fix common encoding issues
    - Standardize punctuation
    - Remove control characters
    """
    if not text:
        return ""

    # Remove control characters except newlines/tabs
    text = "".join(
        c for c in text
        if unicodedata.category(c)[0] != "C" or c in "\n\t\r"
    )

    # Normalize quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Normalize dashes
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Fix double spaces
    text = re.sub(r" {2,}", " ", text)

    # Fix spaces before punctuation
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    """
    Aggressive normalization for matching purposes:
    - Lowercase
    - Remove accents
    - Remove special characters
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).lower()
    # Remove Vietnamese accents for matching
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Keep only alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
