"""
JSON serialization, parsing, and formatting helpers.
"""
import json
import re
from typing import Any, Optional, Dict


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a JSON string with fallback for common issues.
    Handles:
    - Standard JSON
    - JSON with trailing commas
    - JSON embedded in markdown code blocks
    """
    if not text:
        return None

    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as a pretty JSON string."""
    return json.dumps(data, ensure_ascii=False, indent=indent)


def compact_json(data: Any) -> str:
    """Format data as a compact JSON string (no extra whitespace)."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
