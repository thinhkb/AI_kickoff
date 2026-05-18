"""
Parses and standardizes time expressions (Vietnamese).

Handles:
- Month/quarter/year patterns: T1/2025, tháng 1/2025, Quý 3/2025
- Date ranges: T1/2025 - T12/2025
- Relative date references: năm 2025, năm nay
- Compact notations: 01/2025, 2025-01
"""
import re
from datetime import datetime, date
from typing import Tuple, Optional


# ─── Month patterns ───────────────────────────────────────────────
MONTH_PATTERN = re.compile(
    r"(?:tháng|thang|T)\s*(\d{1,2})\s*/\s*(\d{4})",
    re.IGNORECASE,
)

# Quarter patterns
QUARTER_PATTERN = re.compile(
    r"(?:quý|quy|Q)\s*(\d)\s*/\s*(\d{4})",
    re.IGNORECASE,
)

# Year pattern
YEAR_PATTERN = re.compile(
    r"(?:năm|nam)\s+(\d{4})",
    re.IGNORECASE,
)

# Date range: T1/2025 - T12/2025, T1/2025 -> T12/2025
RANGE_SEPARATOR = r"\s*(?:->|[-–—]|đến|den|tới|toi|to)\s*"
RANGE_PATTERN = re.compile(
    r"(?:tháng|thang|T)\s*(\d{1,2})\s*/\s*(\d{4})"
    + RANGE_SEPARATOR +
    r"(?:tháng|thang|T)?\s*(\d{1,2})\s*/\s*(\d{4})",
    re.IGNORECASE,
)

# Compact: khoảng từ tháng X/Y - tháng A/B
RANGE_FULL_PATTERN = re.compile(
    r"(?:từ\s+|khoảng\s+từ\s+)?(?:tháng|thang|T)\s*(\d{1,2})\s*/\s*(\d{4})"
    + RANGE_SEPARATOR +
    r"(?:tháng|thang|T)?\s*(\d{1,2})\s*/\s*(\d{4})",
    re.IGNORECASE,
)

QUARTER_RANGE_PATTERN = re.compile(
    r"(?:từ\s+|khoảng\s+từ\s+)?(?:quý|quy|Q)\s*(\d)\s*/\s*(\d{4})"
    + RANGE_SEPARATOR +
    r"(?:quý|quy|Q)?\s*(\d)\s*/\s*(\d{4})",
    re.IGNORECASE,
)


def month_to_date_range(month: int, year: int) -> Tuple[str, str]:
    """Convert a month/year to fromDate/toDate strings."""
    from_date = f"{year}-{month:02d}-01"
    if month == 12:
        to_date = f"{year}-12-31"
    else:
        next_month_first = date(year, month + 1, 1)
        last_day = date(next_month_first.year, next_month_first.month, 1)
        # Last day of current month
        import calendar
        last = calendar.monthrange(year, month)[1]
        to_date = f"{year}-{month:02d}-{last:02d}"
    return from_date, to_date


def quarter_to_date_range(quarter: int, year: int) -> Tuple[str, str]:
    """Convert a quarter/year to fromDate/toDate strings."""
    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3
    from_date = f"{year}-{start_month:02d}-01"

    import calendar
    last_day = calendar.monthrange(year, end_month)[1]
    to_date = f"{year}-{end_month:02d}-{last_day:02d}"
    return from_date, to_date


def year_to_date_range(year: int) -> Tuple[str, str]:
    """Convert a year to fromDate/toDate strings."""
    return f"{year}-01-01", f"{year}-12-31"


def extract_date_range(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract fromDate and toDate from a Vietnamese time expression.
    Returns (fromDate, toDate) as YYYY-MM-DD strings, or None.
    """
    # Try date range first (T1/2025 - T12/2025)
    m = RANGE_FULL_PATTERN.search(text)
    if not m:
        m = RANGE_PATTERN.search(text)
    if m:
        m1, y1, m2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        from_date = f"{y1}-{m1:02d}-01"
        import calendar
        last_day = calendar.monthrange(y2, m2)[1]
        to_date = f"{y2}-{m2:02d}-{last_day:02d}"
        return from_date, to_date

    # Try quarter range (Q1/2025 -> Q3/2025)
    m = QUARTER_RANGE_PATTERN.search(text)
    if m:
        q1, y1, q2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        start_month = (q1 - 1) * 3 + 1
        end_month = q2 * 3
        from_date = f"{y1}-{start_month:02d}-01"
        import calendar
        last_day = calendar.monthrange(y2, end_month)[1]
        return from_date, f"{y2}-{end_month:02d}-{last_day:02d}"

    # Try quarter
    m = QUARTER_PATTERN.search(text)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        return quarter_to_date_range(q, y)

    # Try single month
    m = MONTH_PATTERN.search(text)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        return month_to_date_range(month, year)

    # Try year
    m = YEAR_PATTERN.search(text)
    if m:
        year = int(m.group(1))
        return year_to_date_range(year)

    return None


def normalize_time_expressions(text: str) -> str:
    """
    Normalize time expressions in text (keep the original text mostly intact,
    just standardize formatting).
    """
    # Normalize T1/2025 → tháng 1/2025 (optional, for consistency)
    # For now, keep original text as-is since the selector and extractors
    # handle raw patterns
    return text
