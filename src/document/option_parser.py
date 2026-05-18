"""
Parses A/B/C/D answer options from the note field.
"""
import re
from typing import Dict, Optional, List


def parse_options(note: str) -> Dict[str, str]:
    """
    Parse multiple-choice options from the note field.

    Expected formats:
    - "A, text\n B, text\n C, text\n D, text"
    - "A. text\nB. text\nC. text\nD. text"

    Returns: {"A": "text", "B": "text", ...}
    """
    if not note:
        return {}

    options = {}

    # Prefer line-start option markers. The previous pattern also matched
    # normal Vietnamese words ending in c./d. inside option text.
    pattern = re.compile(
        r"(?:^|\n)\s*([A-Da-d])\s*[,.\):]\s*(.+?)(?=(?:\n\s*[A-Da-d]\s*[,.\):])|\s*$)",
        re.DOTALL,
    )

    matches = pattern.findall(note)
    if matches:
        for letter, text in matches:
            options[letter.upper()] = text.strip()
    else:
        # Try splitting by newline and looking for letter prefixes
        lines = note.strip().split("\n")
        for line in lines:
            line = line.strip()
            m = re.match(r"([A-Da-d])\s*[,.\):]\s*(.+)", line)
            if m:
                options[m.group(1).upper()] = m.group(2).strip()

    return options


def format_options(options: Dict[str, str]) -> str:
    """Format options back to a readable string."""
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def extract_answer_from_result(result_str: str) -> Optional[str]:
    """
    Extract the answer letter(s) from a result string.
    Expected format: {"numbers": 1, "result": "A"} or {"numbers": 2, "result": "A,B"}
    """
    import json
    try:
        data = json.loads(result_str)
        return data.get("result", "")
    except (json.JSONDecodeError, TypeError):
        # Try regex fallback
        m = re.search(r'"result"\s*:\s*"([A-D,\s]+)"', result_str)
        if m:
            return m.group(1).strip()
    return None
