"""
Build the document knowledge base from PDF files.
Exports: document_chunks.jsonl
"""
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import PDF_DIR, DOCUMENT_CHUNKS_FILE, ensure_dirs
from src.document.pdf_parser import extract_text_from_pdf
from src.document.chunker import chunk_pages
from src.utils.io_utils import write_jsonl
from src.utils.logging_utils import logger


def main():
    ensure_dirs()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")

    all_chunks = []
    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        doc_id = pdf_path.stem  # e.g. "Public_001"

        # Extract text from PDF
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            logger.warning(f"  No text extracted from {pdf_path.name}")
            continue

        # Chunk the pages
        chunks = chunk_pages(pages, doc_id=doc_id)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Export
    chunk_dicts = [c.to_dict() for c in all_chunks]
    write_jsonl(chunk_dicts, DOCUMENT_CHUNKS_FILE)
    logger.info(f"Wrote document chunks to {DOCUMENT_CHUNKS_FILE}")

    # Stats
    doc_ids = set(c.doc_id for c in all_chunks)
    logger.info(f"  Documents: {len(doc_ids)}")
    logger.info(f"  Avg chunks/doc: {len(all_chunks) / max(len(doc_ids), 1):.1f}")


if __name__ == "__main__":
    main()
