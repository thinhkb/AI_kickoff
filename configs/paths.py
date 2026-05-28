"""
Centralized file and folder paths.
"""
import os
import re
from pathlib import Path
from typing import Tuple

# ─── Project root ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Raw data ─────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "Document_config_data"
API_CONFIG_FILE = DATA_DIR / "API_config_data" / "Tài liệu config API.xlsx"
EXAMPLE_DATA_FILE = DATA_DIR / "example_data" / "example_data.xlsx"
TEST_DATA_FILE = DATA_DIR / "test_data" / "Test_data.xlsx"

# ─── Processed data ──────────────────────────────────────────────
PROCESSED_DIR = DATA_DIR / "processed"
DOCUMENT_CHUNKS_FILE = PROCESSED_DIR / "document_chunks.jsonl"
API_REGISTRY_FILE = PROCESSED_DIR / "api_registry.jsonl"
ALIAS_DICTIONARY_FILE = PROCESSED_DIR / "alias_dictionary.json"
SELECTOR_TRAIN_FILE = PROCESSED_DIR / "selector_train.jsonl"

# ─── MinerU paths ─────────────────────────────────────────────────
MINERU_MD_DIR = PROCESSED_DIR / "mineru_markdown"
MINERU_OUT_DIR = PROCESSED_DIR / "mineru_output"

# ─── Cache ────────────────────────────────────────────────────────
CACHE_DIR = DATA_DIR / "cache"
DOC_INDEX_DIR = CACHE_DIR / "doc_index"
API_INDEX_DIR = CACHE_DIR / "api_index"
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"
SELECTOR_MODEL_DIR = CACHE_DIR / "selector_model"

# ─── Outputs ──────────────────────────────────────────────────────
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.jsonl"
SUBMISSION_FILE = OUTPUTS_DIR / "submission.csv"

# ─── Ensure directories exist ─────────────────────────────────────
def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for d in [
        PROCESSED_DIR, MINERU_MD_DIR, MINERU_OUT_DIR, CACHE_DIR,
        DOC_INDEX_DIR, API_INDEX_DIR, EMBEDDINGS_DIR, SELECTOR_MODEL_DIR,
        OUTPUTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# ─── Versioned output paths ──────────────────────────────────────
def get_next_version() -> int:
    """
    Scan outputs/ for existing versioned files and return the next version number.
    Looks for patterns like predictions_v1.jsonl, submission_v3.csv, etc.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    max_version = 0
    pattern = re.compile(r"(?:predictions|submission)_v(\d+)\.")
    for f in OUTPUTS_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            max_version = max(max_version, int(m.group(1)))
    return max_version + 1


def get_versioned_paths(version: int = None) -> Tuple[Path, Path]:
    """
    Return (predictions_path, submission_path) with version suffix.
    If version is None, auto-detect the next version.

    Example: version=3 → predictions_v3.jsonl, submission_v3.csv
    """
    if version is None:
        version = get_next_version()
    predictions_path = OUTPUTS_DIR / f"predictions_v{version}.jsonl"
    submission_path = OUTPUTS_DIR / f"submission_v{version}.csv"
    return predictions_path, submission_path


# ─── Config printer ──────────────────────────────────────────────
def print_current_config():
    """Print the current configuration for traceability."""
    from configs import model_config as mc
    from configs import constants as c

    lines = [
        "",
        "=" * 65,
        "  CURRENT CONFIGURATION",
        "=" * 65,
        "",
        "  [Models]",
        f"    Embedding     : {mc.EMBEDDING_MODEL_NAME}",
        f"    Reranker      : {mc.RERANKER_MODEL_NAME}",
        f"    LLM           : {mc.LLM_MODEL_NAME}",
        f"    LLM endpoint  : {mc.LLM_API_BASE}",
        f"    Selector type : {mc.SELECTOR_TYPE}",
        "",
        "  [Embedding]",
        f"    Dimension     : {mc.EMBEDDING_DIMENSION}",
        f"    Max seq len   : {mc.EMBEDDING_MAX_SEQ_LENGTH}",
        f"    Batch size    : {mc.EMBEDDING_BATCH_SIZE}",
        f"    Normalize     : {mc.EMBEDDING_NORMALIZE}",
        "",
        "  [Reranker]",
        f"    Max seq len   : {mc.RERANKER_MAX_SEQ_LENGTH}",
        f"    Batch size    : {mc.RERANKER_BATCH_SIZE}",
        "",
        "  [LLM]",
        f"    Max tokens    : {mc.LLM_MAX_TOKENS}",
        f"    Temperature   : {mc.LLM_TEMPERATURE}",
        "",
        "  [Retrieval]",
        f"    Doc top-k     : {c.DOC_RETRIEVAL_TOP_K}",
        f"    Doc rerank k  : {c.DOC_RERANK_TOP_K}",
        f"    API top-k     : {c.API_RETRIEVAL_TOP_K}",
        f"    API rerank k  : {c.API_RERANK_TOP_K}",
        "",
        "  [Chunking]",
        f"    Chunk size    : {c.CHUNK_SIZE}",
        f"    Chunk overlap : {c.CHUNK_OVERLAP}",
        "",
        "  [Selector]",
        f"    N-gram range  : {mc.SELECTOR_NGRAM_RANGE}",
        f"    Max features  : {mc.SELECTOR_MAX_FEATURES}",
        f"    Char n-grams  : {mc.SELECTOR_USE_CHAR_NGRAMS}",
        f"    Threshold     : {c.SELECTOR_THRESHOLD}",
        "",
        "=" * 65,
        "",
    ]
    print("\n".join(lines))
