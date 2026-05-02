"""
Parser — converts raw files (PDF/Markdown) into raw text with Math-Formula handling.
Uses `marker-pdf` (Surya AI) to perform Deep Learning OCR for perfect Math + Tables syntax.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.log import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """
    Parse raw document files into markdown content.

    Supported formats:
        - PDF  → extracted via marker-pdf (Surya AI ML Models - High GPU/CPU resource)
        - .md  → passthrough
        - .txt → passthrough
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}

    def __init__(self):
        # We will load models lazily to avoid heavy loading on startup if not processing PDFs
        self.marker_models_dict = None

    def _init_marker_models(self):
        """Lazy load marker PDF deep learning models."""
        if self.marker_models_dict is None:
            logger.info("Loading marker-pdf AI models (this will take a while and consume RAM/VRAM)...")
            from marker.models import create_model_dict
            self.marker_models_dict = create_model_dict()
            logger.info("marker-pdf models loaded successfully.")

    def parse(self, file_path: str) -> str:
        """
        Parse a file and return its aggregated text content.
        Injects `<!-- PAGE: X -->` tags into the text if possible to preserve chunk boundaries.

        Args:
            file_path: Absolute path to the source document.

        Returns:
            Text/markdown string of the document content.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        logger.info(f"Parsing file: {path.name} (type={ext})")

        if ext == ".pdf":
            return self._parse_pdf_via_marker(file_path)
        else:
            return self._parse_text(file_path)

    # ── PDF via Marker-PDF ───────────────────────────────────────────────

    def _parse_pdf_via_marker(self, file_path: str) -> str:
        """
        Extract markdown (text, tables, perfect LaTeX math formulas) from PDF using marker-pdf.
        Injects accurate <!-- PAGE: N --> markers based on marker's <span id="page-X-Y"> tags.
        """
        try:
            from marker.converters.pdf import PdfConverter
            from marker.output import text_from_rendered
        except ImportError:
            raise ImportError(
                "marker-pdf is required for AI-powered PDF parsing. "
                "Install it with: pip install marker-pdf"
            )

        self._init_marker_models()

        logger.info(f"Converting PDF with Marker AI: {file_path}")
        converter = PdfConverter(artifact_dict=self.marker_models_dict)
        
        # Render PDF via Surya AI ecosystem
        rendered = converter(file_path)
        
        # Extract markdown strings
        try:
            markdown_text, _, _ = text_from_rendered(rendered)
        except Exception:
            markdown_text = str(text_from_rendered(rendered))

        if not markdown_text:
            return ""

        # Convert marker's <span id="page-X-Y"> tags to <!-- PAGE: N --> markers
        import re
        
        def _replace_page_span(m):
            page_0idx = int(m.group(1))
            return f"\n<!-- PAGE: {page_0idx + 1} -->\n"
        
        page_text = re.sub(
            r'<span\s+id="page-(\d+)-\d+">\s*</span>',
            _replace_page_span,
            markdown_text,
        )
        
        # Ensure there's at least a PAGE:1 at the start if not present
        if "<!-- PAGE:" not in page_text[:100]:
            page_text = f"<!-- PAGE: 1 -->\n{page_text}"
        
        logger.info(f"PDF parsed via Marker: {len(page_text)} bytes generated.")
        return page_text

    # ── Text / Markdown passthrough ──────────────────────────────────────

    def _parse_text(self, file_path: str) -> str:
        """Read text/markdown file as-is and inject a single page marker."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        logger.info(f"Text file read: {len(content)} chars")
        return f"<!-- PAGE: 1 -->\n{content}"
