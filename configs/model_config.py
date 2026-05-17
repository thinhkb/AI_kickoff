"""
Model names, checkpoint paths, token limits, and inference configuration.
"""

# ─── Embedding Model ─────────────────────────────────────────────
# Accuracy-first: Qwen/Qwen3-Embedding-8B
# Practical backup: BAAI/bge-m3
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MAX_SEQ_LENGTH = 8192
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_NORMALIZE = True

# Query instructions for different tasks
EMBEDDING_QUERY_INSTRUCTION_DOC = (
    "Given a Vietnamese technical question, retrieve the most relevant "
    "PDF passages that answer it."
)
EMBEDDING_QUERY_INSTRUCTION_API = (
    "Given a Vietnamese dashboard/API question, retrieve the API "
    "specification that should be called."
)

# ─── Reranker Model ──────────────────────────────────────────────
# Accuracy-first: Qwen/Qwen3-Reranker-8B
# Practical backup: BAAI/bge-reranker-v2-m3
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
RERANKER_MAX_SEQ_LENGTH = 1024
RERANKER_BATCH_SIZE = 16

# ─── LLM (fallback/verifier) ─────────────────────────────────────
# Accuracy-first: Qwen/Qwen3-32B
# Served via vLLM or SGLang
LLM_MODEL_NAME = "Qwen/Qwen3-32B"
LLM_API_BASE = "http://localhost:8000/v1"  # vLLM endpoint
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.0

# ─── OCR / Layout Models ─────────────────────────────────────────
OCR_MODEL = "PP-OCRv5"
LAYOUT_MODEL = "PP-StructureV3"
OCR_LANG = "vi"

# ─── Selector Model ──────────────────────────────────────────────
SELECTOR_TYPE = "tfidf_lr"  # tfidf_lr | tfidf_svc | embedding_lr
SELECTOR_NGRAM_RANGE = (1, 3)
SELECTOR_MAX_FEATURES = 50000
SELECTOR_USE_CHAR_NGRAMS = True
