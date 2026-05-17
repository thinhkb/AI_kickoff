"""
File reading / writing helpers.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import openpyxl


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items: List[Dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_excel_sheet(
    path: str | Path,
    sheet_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read an Excel sheet into a list of dicts."""
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    result = []
    for row in rows[1:]:
        d = dict(zip(headers, row))
        if any(v is not None for v in d.values()):
            result.append(d)
    wb.close()
    return result


def write_csv(
    items: List[Dict[str, Any]],
    path: str | Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Write a list of dicts to a CSV file.
    Uses utf-8-sig encoding (UTF-8 with BOM) so Excel on Windows
    correctly displays Vietnamese/Unicode characters.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not items:
        return
    if fieldnames is None:
        fieldnames = list(items[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)

