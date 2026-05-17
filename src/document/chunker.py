"""
Splits parsed PDF content into semantic chunks with metadata.
"""
import re
from typing import List, Dict, Any

from src.schemas import DocumentChunk
from configs.constants import CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.logging_utils import logger


def chunk_pages(
    pages: List[Dict[str, Any]],
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Split page texts into overlapping chunks.
    Tries to break at sentence boundaries.
    """
    chunks = []
    chunk_counter = 0

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]

        if not text.strip():
            continue

        # Split into sentences (Vietnamese-aware)
        sentences = split_sentences(text)

        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            if current_len + sent_len > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    page=page_num,
                    chunk_id=chunk_counter,
                    text=chunk_text.strip(),
                    chunk_type="text",
                ))
                chunk_counter += 1

                # Overlap: keep last few sentences
                overlap_text = ""
                overlap_sents = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) <= chunk_overlap:
                        overlap_sents.insert(0, s)
                        overlap_text = " ".join(overlap_sents)
                    else:
                        break
                current_chunk = overlap_sents
                current_len = len(overlap_text)

            current_chunk.append(sent)
            current_len += sent_len

        # Last chunk for this page
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip():
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    page=page_num,
                    chunk_id=chunk_counter,
                    text=chunk_text.strip(),
                    chunk_type="text",
                ))
                chunk_counter += 1

    return chunks


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling Vietnamese text.
    """
    # Split on sentence-ending punctuation followed by space or newline
    parts = re.split(r'(?<=[.!?])\s+', text)

    # Also split on double newlines (paragraph breaks)
    result = []
    for part in parts:
        sub_parts = re.split(r'\n\s*\n', part)
        result.extend(sub_parts)

    # Clean up
    result = [s.strip() for s in result if s.strip()]
    return result


def chunk_text_blocks(
    blocks: List[Dict[str, Any]],
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
) -> List[DocumentChunk]:
    """
    Alternative chunking using structured text blocks.
    Preserves heading information.
    """
    chunks = []
    chunk_counter = 0
    current_heading = ""
    current_text = []
    current_len = 0

    for block in blocks:
        if block.get("is_heading"):
            # Save previous chunk
            if current_text:
                chunk_text = "\n".join(current_text)
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    page=block["page"],
                    chunk_id=chunk_counter,
                    text=chunk_text.strip(),
                    heading=current_heading,
                    chunk_type="text",
                ))
                chunk_counter += 1

            current_heading = block["text"]
            current_text = []
            current_len = 0
        else:
            text = block["text"]
            if current_len + len(text) > chunk_size and current_text:
                chunk_text = "\n".join(current_text)
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    page=block["page"],
                    chunk_id=chunk_counter,
                    text=chunk_text.strip(),
                    heading=current_heading,
                    chunk_type="text",
                ))
                chunk_counter += 1
                current_text = []
                current_len = 0

            current_text.append(text)
            current_len += len(text)

    # Final chunk
    if current_text:
        chunks.append(DocumentChunk(
            doc_id=doc_id,
            page=blocks[-1]["page"] if blocks else 0,
            chunk_id=chunk_counter,
            text="\n".join(current_text).strip(),
            heading=current_heading,
            chunk_type="text",
        ))

    return chunks
