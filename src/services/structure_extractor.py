"""
Structure Extractor — extracts headers, sections, and tables from markdown.
Normalized to support math formulas and Markdown outputs from pymupdf4llm.
"""

from __future__ import annotations

import re
from typing import Optional

from src.models.schemas import HeaderNode, StructuredDocument, TableBlock
from src.utils.log import get_logger

logger = get_logger(__name__)

# ── Header & Component detection patterns ─────────────────────────────────────

# Markdown headers: # Header, ## Header, etc.
RE_MD_HEADER = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Numbered headers: 1. Header, 1.2. Header, 1.2.3 Header
RE_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+)$", re.MULTILINE)

# Letter headers: A. Header, B. Header
RE_LETTER = re.compile(r"^([A-Z])\.\s+(.+)$", re.MULTILINE)

# Roman numeral headers: I. Header, II. Header, IV. Header
RE_ROMAN = re.compile(r"^(I{1,3}|IV|V|VI{0,3}|IX|X{0,3})\.\s+(.+)$", re.MULTILINE)

# Markdown table detection
RE_TABLE_START = re.compile(r"^\|(.+)\|$", re.MULTILINE)
RE_TABLE_SEPARATOR = re.compile(r"^\|[\s\-:]+\|$", re.MULTILINE)

# Page marker injected by Parser
RE_PAGE_MARKER = re.compile(r"^<!-- PAGE:\s*(\d+)\s*-->$")


class StructureExtractor:
    """
    Extract document structure: headers hierarchy, flat sections, and tables.
    """

    # Maximum length for a line to be considered a "fake header"
    FAKE_HEADER_MAX_LEN = 80

    def extract(self, markdown: str, file_title: str) -> StructuredDocument:
        """
        Parse markdown into a StructuredDocument with header tree,
        flat sections, tables, and precise page tracking.
        """
        logger.info(f"Extracting structure from: {file_title}")

        normalized = self._normalize_headers(markdown)
        tables = self._extract_tables(normalized, file_title)
        header_tree = self._build_header_tree(normalized)
        flat_sections = self._flatten_to_sections(normalized)

        doc = StructuredDocument(
            file_title=file_title,
            raw_markdown=markdown,
            header_tree=header_tree,
            tables=tables,
            flat_sections=flat_sections,
        )

        logger.info(
            f"Structure extracted: {len(flat_sections)} sections, "
            f"{len(tables)} tables, tree depth={self._tree_depth(header_tree)}"
        )
        return doc

    # ── Header Normalization ─────────────────────────────────────────────

    def _normalize_headers(self, text: str) -> str:
        lines = text.split("\n")
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            if RE_PAGE_MARKER.match(stripped):
                result.append(line)
                continue

            if not stripped:
                result.append(line)
                continue

            if stripped.startswith("#"):
                result.append(line)
                continue

            m = RE_NUMBERED.match(stripped)
            if m:
                num_parts = m.group(1).rstrip(".").split(".")
                level = min(len(num_parts), 6)
                title = m.group(2).strip()
                result.append(f"{'#' * level} {m.group(1)} {title}")
                continue

            m = RE_ROMAN.match(stripped)
            if m:
                result.append(f"# {m.group(1)}. {m.group(2).strip()}")
                continue

            m = RE_LETTER.match(stripped)
            if m:
                result.append(f"## {m.group(1)}. {m.group(2).strip()}")
                continue

            if self._is_fake_header(stripped, lines, i):
                result.append(f"### {stripped}")
                continue

            result.append(line)

        return "\n".join(result)

    def _is_fake_header(self, line: str, lines: list[str], index: int) -> bool:
        if len(line) > self.FAKE_HEADER_MAX_LEN:
            return False
        if line.endswith((".", "!", "?", ",", ";", ":")):
            return False
        if "|" in line:
            return False

        prev_blank = index == 0 or lines[index - 1].strip() == ""
        next_blank = index >= len(lines) - 1 or lines[index + 1].strip() == ""

        return prev_blank and next_blank

    # ── Header Tree Builder ──────────────────────────────────────────────

    def _build_header_tree(self, markdown: str) -> list[HeaderNode]:
        lines = markdown.split("\n")
        root_nodes: list[HeaderNode] = []
        stack: list[HeaderNode] = []

        current_content_lines: list[str] = []
        current_page = 1

        def _flush_content():
            if stack and current_content_lines:
                text = "\n".join(current_content_lines).strip()
                if text:
                    stack[-1].content += ("\n" + text if stack[-1].content else text)
            current_content_lines.clear()

        for line in lines:
            stripped = line.strip()
            
            pm = RE_PAGE_MARKER.match(stripped)
            if pm:
                current_page = int(pm.group(1))
                continue

            m = RE_MD_HEADER.match(stripped)
            if m:
                _flush_content()
                level = len(m.group(1))
                title = m.group(2).strip()
                node = HeaderNode(level=level, title=title, page=current_page)

                while stack and stack[-1].level >= level:
                    stack.pop()

                if stack:
                    stack[-1].children.append(node)
                else:
                    root_nodes.append(node)

                stack.append(node)
            else:
                current_content_lines.append(line)

        _flush_content()
        return root_nodes

    # ── Flatten to Sections ──────────────────────────────────────────────

    def _flatten_to_sections(self, markdown: str) -> list[dict]:
        lines = markdown.split("\n")
        sections: list[dict] = []
        header_stack: list[tuple[int, str]] = []
        current_content: list[str] = []
        current_title = ""
        current_page = 1

        def _flush_section(page_to_save: int):
            content = "\n".join(current_content).strip()
            if content or current_title:
                path = " > ".join(t for _, t in header_stack)
                sections.append({
                    "header_path": path,
                    "header_title": current_title,
                    "content": content,
                    "page": page_to_save,
                })
            current_content.clear()

        last_flush_page = 1

        for line in lines:
            stripped = line.strip()
            
            pm = RE_PAGE_MARKER.match(stripped)
            if pm:
                current_page = int(pm.group(1))
                continue

            m = RE_MD_HEADER.match(stripped)
            if m:
                _flush_section(last_flush_page)
                last_flush_page = current_page
                level = len(m.group(1))
                title = m.group(2).strip()

                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                header_stack.append((level, title))
                current_title = title
            else:
                current_content.append(line)

        _flush_section(last_flush_page)

        logger.debug(f"Flattened into {len(sections)} sections")
        return sections

    # ── Table Extraction ─────────────────────────────────────────────────

    def _extract_tables(self, markdown: str, file_title: str) -> list[TableBlock]:
        tables: list[TableBlock] = []
        lines = markdown.split("\n")
        i = 0
        current_page = 1

        while i < len(lines):
            stripped = lines[i].strip()
            pm = RE_PAGE_MARKER.match(stripped)
            if pm:
                current_page = int(pm.group(1))
                i += 1
                continue

            if RE_TABLE_START.match(stripped):
                if i + 1 < len(lines) and RE_TABLE_SEPARATOR.match(lines[i + 1].strip()):
                    table_lines = [lines[i]]
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith("|"):
                        table_lines.append(lines[j])
                        j += 1

                    table = self._parse_table(table_lines, file_title)
                    if table:
                        table.page = current_page
                        table.header_path = self._find_header_above(lines, i)
                        tables.append(table)

                    i = j
                    continue
            i += 1

        logger.debug(f"Extracted {len(tables)} tables")
        return tables

    def _parse_table(self, table_lines: list[str], file_title: str) -> Optional[TableBlock]:
        if len(table_lines) < 3:
            return None

        header_line = table_lines[0].strip().strip("|")
        columns = [c.strip() for c in header_line.split("|")]

        rows = []
        for row_line in table_lines[2:]:
            stripped = row_line.strip()
            if not stripped.startswith("|"):
                break
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            row_dict = {}
            for idx, col in enumerate(columns):
                row_dict[col] = cells[idx] if idx < len(cells) else ""
            rows.append(row_dict)

        raw_markdown = "\n".join(table_lines)

        return TableBlock(
            markdown=raw_markdown,
            rows=rows,
        )

    def _find_header_above(self, lines: list[str], index: int) -> str:
        for i in range(index - 1, -1, -1):
            m = RE_MD_HEADER.match(lines[i].strip())
            if m:
                return m.group(2).strip()
        return ""

    # ── Utility ──────────────────────────────────────────────────────────

    def _tree_depth(self, nodes: list[HeaderNode], depth: int = 0) -> int:
        if not nodes:
            return depth
        return max(self._tree_depth(n.children, depth + 1) for n in nodes)
