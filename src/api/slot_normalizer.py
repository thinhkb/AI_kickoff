"""
Normalizes aliases, abbreviations, misspellings, and backend-specific values.

Three-layer normalization strategy:
1. Exact normalization (lowercase, trim, accent-insensitive, dict lookup)
2. Fuzzy normalization (edit distance, typo correction)
3. Semantic normalization (embedding-based similarity)
"""
from typing import Any, Dict, List, Optional

from src.utils.fuzzy_match import (
    exact_match,
    accent_insensitive_match,
    fuzzy_match_best,
    normalize_text,
)
from src.utils.logging_utils import logger


class SlotNormalizer:
    """
    Normalizes slot values against known alias dictionaries.
    """

    def __init__(self, alias_dict: Dict[str, Dict[str, str]] = None):
        self.alias_dict = alias_dict or {}

    def normalize_slots(
        self,
        slots: Dict[str, Any],
        api_params: List[str],
    ) -> Dict[str, Any]:
        """
        Normalize all slot values.
        """
        normalized = {}
        for param, value in slots.items():
            param_clean = param.strip()
            aliases = self.alias_dict.get(param_clean, {})

            if isinstance(value, list):
                normalized[param_clean] = [
                    self._normalize_single(v, aliases)
                    for v in value
                ]
            elif isinstance(value, str) and aliases:
                normalized[param_clean] = self._normalize_single(value, aliases)
            else:
                normalized[param_clean] = value

        return normalized

    def _normalize_single(
        self,
        value: Any,
        aliases: Dict[str, str],
    ) -> Any:
        """Normalize a single value through the three-layer strategy."""
        if not isinstance(value, str) or not aliases:
            return value

        # Layer 1: Exact match
        result = exact_match(value, aliases)
        if result is not None:
            return result

        # Layer 1b: Accent-insensitive match
        result = accent_insensitive_match(value, aliases)
        if result is not None:
            return result

        # Layer 2: Fuzzy match
        result = fuzzy_match_best(value, aliases, threshold=80)
        if result is not None:
            name, backend_value, score = result
            logger.debug(f"  Fuzzy matched '{value}' → '{backend_value}' (score={score:.0f})")
            return backend_value

        # Layer 3: If nothing matches, return original value
        # (Semantic matching would go here with embedding model)
        logger.debug(f"  No match found for '{value}', keeping original")
        return value
