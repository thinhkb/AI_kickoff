"""
MinerU PDF parser that integrates layout-preserved markdown extraction.
Supports:
1. Loading pre-extracted Markdown from data/processed/mineru_markdown/
2. On-the-fly extraction via magic-pdf CLI (if available)
3. PyMuPDF fallback (converting text pages into unified Markdown with page separators)
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF fallback

from configs.paths import MINERU_MD_DIR, MINERU_OUT_DIR
from src.utils.logging_utils import logger


class MinerUParser:
    """
    MinerU PDF parser.
    """

    def __init__(self, mineru_md_dir: Path = MINERU_MD_DIR, mineru_out_dir: Path = MINERU_OUT_DIR):
        self.mineru_md_dir = Path(mineru_md_dir)
        self.mineru_out_dir = Path(mineru_out_dir)

    def parse_pdf(self, pdf_path: str | Path) -> str:
        """
        Parse a PDF file into a layout-preserved Markdown string.
        Attempts to read pre-extracted markdown, run magic-pdf CLI, or fall back to PyMuPDF.
        """
        pdf_path = Path(pdf_path)
        pdf_stem = pdf_path.stem

        # Step 1: Check for pre-extracted Markdown in data/processed/mineru_markdown/
        # Check direct file: mineru_markdown/Public_001.md
        direct_md_path = self.mineru_md_dir / f"{pdf_stem}.md"
        if direct_md_path.exists():
            logger.info(f"MinerU Parser: Loaded pre-extracted Markdown for {pdf_stem}")
            with open(direct_md_path, "r", encoding="utf-8") as f:
                return f.read()

        # Check subdirectories: mineru_markdown/Public_001/Public_001.md
        sub_md_path = self.mineru_md_dir / pdf_stem / f"{pdf_stem}.md"
        if sub_md_path.exists():
            logger.info(f"MinerU Parser: Loaded pre-extracted Markdown from subdirectory for {pdf_stem}")
            with open(sub_md_path, "r", encoding="utf-8") as f:
                return f.read()

        # Step 2: Attempt on-the-fly conversion using magic-pdf CLI
        if shutil.which("magic-pdf"):
            logger.info(f"MinerU Parser: magic-pdf CLI detected, attempting conversion for {pdf_stem}...")
            try:
                # Lệnh tiêu chuẩn: magic-pdf -p <pdf_path> -o <mineru_out_dir> -m auto
                cmd = [
                    "magic-pdf",
                    "-p", str(pdf_path),
                    "-o", str(self.mineru_out_dir),
                    "-m", "auto"
                ]
                
                # Execute command
                result = subprocess.run(
                    cmd,
                    cwd=str(self.mineru_out_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                
                # Scan mineru_out_dir/pdf_stem/pdf_stem.md
                generated_md_path = self.mineru_out_dir / pdf_stem / f"{pdf_stem}.md"
                if generated_md_path.exists():
                    logger.info(f"MinerU Parser: Successfully parsed {pdf_stem} using magic-pdf CLI")
                    with open(generated_md_path, "r", encoding="utf-8") as f:
                        return f.read()
                else:
                    logger.warning("MinerU Parser: magic-pdf ran but output file was not found where expected.")
            except Exception as e:
                logger.error(f"MinerU Parser: magic-pdf CLI execution failed: {e}")

        # Step 3: Fall back to PyMuPDF and format as clean Markdown with page dividers
        logger.warning(f"MinerU Parser: Pre-extracted Markdown not found and magic-pdf CLI not available. Falling back to PyMuPDF for {pdf_stem}...")
        return self._pymupdf_fallback(pdf_path)

    def _pymupdf_fallback(self, pdf_path: Path) -> str:
        """
        Fallback parser using PyMuPDF.
        Converts text blocks into a pseudo-markdown format preserving pages.
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return ""

        markdown_pages = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                
                # Format page with page indicators
                page_md = f"<!-- PAGE_{page_num + 1} -->\n\n{text}"
                markdown_pages.append(page_md)
            doc.close()
        except Exception as e:
            logger.error(f"MinerU Parser Fallback: Error parsing {pdf_path.name} via PyMuPDF: {e}")
            return ""

        # Join pages with standard horizontal rule separators
        return "\n\n------\n\n".join(markdown_pages)
