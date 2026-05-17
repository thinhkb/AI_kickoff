"""
Shared constants for the Viettel AI Race chatbot.
"""

# ─── Function codes ───────────────────────────────────────────────
FUNC_CALL_DOCUMENT = "call_document"
FUNC_CALL_API = "call_api"
FUNCTION_CODES = [FUNC_CALL_DOCUMENT, FUNC_CALL_API]

# ─── Schema keys ──────────────────────────────────────────────────
KEY_ID = "id"
KEY_QUESTION = "fun_question"
KEY_NOTE = "note"
KEY_FUNC_CODE = "func_code"
KEY_FUNC_ANSWER = "func_param"
KEY_TIME_RESPONSE = "time_response"

# Output keys (submission format)
OUT_ID = "id"
OUT_FUNC_CODE = "function_code"
OUT_FUNC_ANSWER = "function_answer"
OUT_TIME_RESPONSE = "time_response"

# ─── Retrieval settings ──────────────────────────────────────────
DOC_RETRIEVAL_TOP_K = 50
DOC_RERANK_TOP_K = 10
API_RETRIEVAL_TOP_K = 30
API_RERANK_TOP_K = 5

# Chunk settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ─── Selector settings ───────────────────────────────────────────
SELECTOR_THRESHOLD = 0.5

# ─── Time response limit (seconds) ───────────────────────────────
MAX_TIME_RESPONSE = 15.0
