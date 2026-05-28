"""
MinerUMarkdownChunker.
Splits layout-preserved Markdown (from MinerU) into semantically rich chunks.
Tracks headings hierarchically, preserves tables whole, and tags formula/table chunk types.
"""
import re
from typing import List, Dict, Any, Optional

from src.schemas import DocumentChunk
from configs.constants import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logging_utils import logger


class MinerUMarkdownChunker:
    """
    Layout-aware chunker for Markdown documents.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_markdown(self, markdown_text: str, doc_id: str) -> List[DocumentChunk]:
        """
        Chunk a Markdown document into overlapping semantic chunks.
        Tracks heading hierarchy and keeps tables/formulas intact.
        """
        if not markdown_text.strip():
            return []

        # Step 1: Split into pages if page indicators are present
        # Supports: '------', '---', or '<!-- PAGE_N -->'
        pages = self._split_into_pages(markdown_text)
        
        all_chunks = []
        chunk_counter = 0
        
        # Heading path tracking across the whole document
        # e.g., ["1. Introduction", "1.1 Overview"]
        current_headings = []

        for page_idx, page_content in pages:
            # Check if there is an explicit page marker in the content
            # e.g. <!-- PAGE_3 -->
            page_num = self._extract_page_num(page_content, default_page=page_idx)
            
            # Remove page comments from content to keep text clean
            clean_content = re.sub(r"<!--\s*PAGE_\d+\s*-->", "", page_content).strip()
            if not clean_content:
                continue

            # Step 2: Extract blocks (headings, tables, paragraphs) from page
            blocks = self._extract_blocks(clean_content)
            
            current_chunk_tokens = []
            current_chunk_length = 0
            
            for block in blocks:
                block_text = block["text"]
                block_type = block["type"]
                
                # If block is a heading, update our heading path hierarchy
                if block_type == "heading":
                    current_headings = self._update_headings(current_headings, block_text)
                    # We can also treat headings as part of the chunk or use them for context
                    heading_str = " > ".join(current_headings)
                    
                    # If we already have accumulated content, flush the current chunk before starting a new section
                    if current_chunk_tokens:
                        all_chunks.append(self._create_chunk(
                            doc_id=doc_id,
                            page=page_num,
                            chunk_id=chunk_counter,
                            tokens=current_chunk_tokens,
                            headings=current_headings,
                            chunk_type="text"
                        ))
                        chunk_counter += 1
                        current_chunk_tokens = []
                        current_chunk_length = 0
                    
                    # Prepend heading context to the first chunk of the section
                    current_chunk_tokens.append(f"## {block_text}")
                    current_chunk_length += len(block_text)
                    continue

                # If block is a table, we keep it whole and create a separate "table" chunk
                if block_type == "table":
                    # Flush any current text chunk
                    if current_chunk_tokens:
                        all_chunks.append(self._create_chunk(
                            doc_id=doc_id,
                            page=page_num,
                            chunk_id=chunk_counter,
                            tokens=current_chunk_tokens,
                            headings=current_headings,
                            chunk_type="text"
                        ))
                        chunk_counter += 1
                        current_chunk_tokens = []
                        current_chunk_length = 0
                    
                    # Create dedicated table chunk
                    heading_str = " > ".join(current_headings)
                    # Prepend heading context to table so retrieval finds it easily
                    table_context = f"[Bảng biểu thuộc mục: {heading_str}]\n" if heading_str else ""
                    table_text = table_context + block_text
                    
                    all_chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        page=page_num,
                        chunk_id=chunk_counter,
                        text=table_text,
                        heading=heading_str,
                        chunk_type="table",
                        metadata={"is_table": True}
                    ))
                    chunk_counter += 1
                    continue

                # For standard paragraphs or formulas, check if we need to split
                is_formula = (block_type == "formula")
                block_len = len(block_text)
                
                # If adding this block exceeds chunk_size, flush the current chunk
                if current_chunk_length + block_len > self.chunk_size and current_chunk_tokens:
                    all_chunks.append(self._create_chunk(
                        doc_id=doc_id,
                        page=page_num,
                        chunk_id=chunk_counter,
                        tokens=current_chunk_tokens,
                        headings=current_headings,
                        chunk_type="text"
                    ))
                    chunk_counter += 1
                    
                    # Implement overlap: keep last few tokens/sentences if possible
                    overlap_tokens = []
                    overlap_len = 0
                    for token in reversed(current_chunk_tokens):
                        if overlap_len + len(token) <= self.chunk_overlap:
                            overlap_tokens.insert(0, token)
                            overlap_len += len(token)
                        else:
                            break
                    current_chunk_tokens = overlap_tokens
                    current_chunk_length = overlap_len

                current_chunk_tokens.append(block_text)
                current_chunk_length += block_len

            # Flush any remaining tokens at the end of the page
            if current_chunk_tokens:
                all_chunks.append(self._create_chunk(
                    doc_id=doc_id,
                    page=page_num,
                    chunk_id=chunk_counter,
                    tokens=current_chunk_tokens,
                    headings=current_headings,
                    chunk_type="text"
                ))
                chunk_counter += 1

        return all_chunks

    def _split_into_pages(self, text: str) -> List[tuple[int, str]]:
        """Split document by horizontal rules or page comments."""
        # Check if there are explicit page indicators
        if "<!-- PAGE_" in text:
            # Keep the page comment in the content so we can extract it in _extract_page_num
            parts = re.split(r"(?=<!--\s*PAGE_\d+\s*-->)", text)
            pages = []
            for i, p in enumerate(parts):
                if p.strip():
                    pages.append((i + 1, p))
            return pages

        # Check for horizontal rules: \n\n------\n\n or \n\n---\n\n
        hr_pattern = r"\n\n-{3,}\n\n"
        if re.search(hr_pattern, text):
            parts = re.split(hr_pattern, text)
            return [(i + 1, p) for i, p in enumerate(parts) if p.strip()]

        # Otherwise treat the entire text as page 1
        return [(1, text)]

    def _extract_page_num(self, content: str, default_page: int) -> int:
        """Extract page number from comment <!-- PAGE_N -->."""
        match = re.search(r"<!--\s*PAGE_(\d+)\s*-->", content)
        if match:
            return int(match.group(1))
        return default_page

    def _extract_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify headings, tables, formulas, and paragraphs in text.
        """
        blocks = []
        lines = text.split("\n")
        
        in_table = False
        table_lines = []
        
        in_code_or_math = False
        math_block_lines = []
        
        current_paragraph = []

        for line in lines:
            line_stripped = line.strip()
            
            # Check for math blocks: $$ or similar
            if line_stripped.startswith("$$"):
                if in_code_or_math:
                    math_block_lines.append(line)
                    blocks.append({
                        "type": "formula",
                        "text": "\n".join(math_block_lines)
                    })
                    math_block_lines = []
                    in_code_or_math = False
                else:
                    # Flush paragraph first
                    if current_paragraph:
                        blocks.append({"type": "text", "text": "\n".join(current_paragraph)})
                        current_paragraph = []
                    in_code_or_math = True
                    math_block_lines.append(line)
                continue

            if in_code_or_math:
                math_block_lines.append(line)
                continue

            # Check for Table markdown: lines starting/ending with |
            is_table_line = (line_stripped.startswith("|") and line_stripped.endswith("|")) or \
                             (in_table and line_stripped.startswith("|"))
            
            if is_table_line:
                if not in_table:
                    # Flush existing paragraph
                    if current_paragraph:
                        blocks.append({"type": "text", "text": "\n".join(current_paragraph)})
                        current_paragraph = []
                    in_table = True
                table_lines.append(line)
                continue
            
            if in_table:
                # Table ended
                blocks.append({
                    "type": "table",
                    "text": "\n".join(table_lines)
                })
                table_lines = []
                in_table = False

            # Check for Headings: lines starting with #
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", line_stripped)
            if heading_match:
                # Flush paragraph
                if current_paragraph:
                    blocks.append({"type": "text", "text": "\n".join(current_paragraph)})
                    current_paragraph = []
                
                blocks.append({
                    "type": "heading",
                    "text": heading_match.group(2).strip(),
                    "level": len(heading_match.group(1))
                })
                continue

            # Otherwise, standard text line
            if line_stripped:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    blocks.append({"type": "text", "text": "\n".join(current_paragraph)})
                    current_paragraph = []

        # Flush any remaining blocks
        if in_table and table_lines:
            blocks.append({"type": "table", "text": "\n".join(table_lines)})
        if in_code_or_math and math_block_lines:
            blocks.append({"type": "formula", "text": "\n".join(math_block_lines)})
        if current_paragraph:
            blocks.append({"type": "text", "text": "\n".join(current_paragraph)})

        # Clean empty blocks
        blocks = [b for b in blocks if b["text"].strip()]
        
        # Post-process blocks to label dense math formulas
        for b in blocks:
            if b["type"] == "text":
                # If paragraph contains LaTeX inline formulas e.g. $E = mc^2$ or many mathematical symbols, tag as formula
                math_symbols = len(re.findall(r"[\+\-\*\/=\<\>\$\\\{\}\_\^]", b["text"]))
                if math_symbols > 15 and len(b["text"]) < 400:
                    b["type"] = "formula"

        return blocks

    def _update_headings(self, current_headings: List[str], new_heading: str) -> List[str]:
        """
        Update the heading path hierarchy.
        Since we don't have level numbers in _extract_blocks for simplicity, we assume
        a flat list or we can try to guess level by numbers like '1.', '1.1', etc.
        Let's parse numbering to adjust the tree depth!
        """
        # Clean heading
        heading_clean = new_heading.strip()
        
        # Try to parse section numbering like: "1. Introduction" or "1.2.1 Installation"
        num_match = re.match(r"^(\d+(?:\.\d+)*)\b", heading_clean)
        if num_match:
            dots = num_match.group(1).count(".")
            level = dots + 1
            # Trim headings to appropriate level
            trimmed = current_headings[:level - 1]
            trimmed.append(heading_clean)
            return trimmed
        
        # If no numbers, let's keep it simple: if it's short it's a section, if there are many sections we limit depth
        if len(current_headings) >= 3:
            return current_headings[:-1] + [heading_clean]
        else:
            return current_headings + [heading_clean]

    def _create_chunk(
        self,
        doc_id: str,
        page: int,
        chunk_id: int,
        tokens: List[str],
        headings: List[str],
        chunk_type: str = "text"
    ) -> DocumentChunk:
        """Create a DocumentChunk with heading context prepended for optimal RAG."""
        heading_str = " > ".join(headings)
        body_text = "\n".join(tokens)
        
        # Prepend heading context so the chunk carries its section name
        if heading_str:
            full_text = f"[Mục: {heading_str}]\n{body_text}"
        else:
            full_text = body_text

        # Detect formula signatures
        is_formula = "chunk_type" == "formula" or len(re.findall(r"[\$\\\{\}\_\^]", body_text)) > 20
        c_type = "formula" if is_formula else chunk_type

        return DocumentChunk(
            doc_id=doc_id,
            page=page,
            chunk_id=chunk_id,
            text=full_text,
            heading=heading_str,
            chunk_type=c_type,
            metadata={"is_formula": is_formula}
        )
