"""
Extracts text blocks and page metadata from PDF documents.
Uses text-layer parsing as the primary path (PyMuPDF / pdfplumber).
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
from src.utils.logging_utils import logger


def extract_text_from_pdf(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF using PyMuPDF text-layer parsing.

    Returns a list of page dicts:
    [
        {
            "page": 1,
            "text": "...",
            "has_images": True/False,
            "has_tables": True/False,
            "text_coverage": 0.95  # ratio of text area to page area
        },
        ...
    ]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning(f"PDF not found: {pdf_path}")
        return []

    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Check for images
            images = page.get_images(full=True)
            has_images = len(images) > 0

            # Estimate text coverage
            page_area = page.rect.width * page.rect.height
            text_blocks = page.get_text("blocks")
            text_area = sum(
                (b[2] - b[0]) * (b[3] - b[1])
                for b in text_blocks
                if b[6] == 0  # text blocks only
            )
            text_coverage = text_area / page_area if page_area > 0 else 0

            pages.append({
                "page": page_num + 1,
                "text": text.strip(),
                "has_images": has_images,
                "has_tables": False,  # Will be detected in table_parser
                "text_coverage": round(text_coverage, 3),
            })
        doc.close()
    except Exception as e:
        logger.error(f"Error parsing {pdf_path}: {e}")

    return pages


def extract_text_blocks(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Extract structured text blocks with position info.
    Useful for detecting headings and maintaining reading order.
    """
    pdf_path = Path(pdf_path)
    blocks = []

    try:
        doc = fitz.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_blocks = page.get_text("dict")["blocks"]

            for block in text_blocks:
                if block["type"] != 0:  # Skip image blocks
                    continue

                # Extract text from lines/spans
                block_text = ""
                max_font_size = 0
                is_bold = False

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span["text"]
                        max_font_size = max(max_font_size, span["size"])
                        if "bold" in span.get("font", "").lower():
                            is_bold = True
                    block_text += "\n"

                block_text = block_text.strip()
                if not block_text:
                    continue

                blocks.append({
                    "page": page_num + 1,
                    "text": block_text,
                    "bbox": block["bbox"],
                    "font_size": max_font_size,
                    "is_bold": is_bold,
                    "is_heading": is_bold and max_font_size > 12,
                })
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting blocks from {pdf_path}: {e}")

    return blocks


def needs_ocr(page_info: Dict[str, Any], min_coverage: float = 0.1) -> bool:
    """
    Determine if a page needs OCR fallback.
    Pages with very low text coverage likely contain scanned content.
    """
    return (
        page_info["text_coverage"] < min_coverage
        or (len(page_info["text"].strip()) < 50 and page_info["has_images"])
    )
