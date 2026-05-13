"""
Embedder — hỗ trợ 2 provider qua env EMBED_PROVIDER:
  local  → BAAI/bge-m3 chạy trên máy (GPU tự detect)
  gemini → Google Gemini gemini-embedding-001 (cần GEMINI_API_KEY)

Dim: local=1024, gemini=3072
"""
from __future__ import annotations

import os
import time
import requests as req
from src.models.schemas import Chunk
from src.utils.log import StageLogger, get_logger

logger = get_logger(__name__)

PROVIDER   = os.getenv("EMBED_PROVIDER", "local").lower()  # "local" | "gemini"
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
BATCH_SIZE = 32


# ── Gemini Embedder ───────────────────────────────────────────────────────────

class GeminiEmbedder:
    DIM      = 1024 # gemini-embedding-001 mặc định 3072 chiều
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    MODEL_ID = "gemini-embedding-001"

    def __init__(self, api_key: str = GEMINI_KEY):
        if not api_key:
            raise ValueError("GEMINI_API_KEY chưa set trong .env")
        self._key = api_key
        self.endpoint = f"{self.BASE_URL}/{self.MODEL_ID}:embedContent"
        logger.info(f"Gemini embedder ready — {self.MODEL_ID} (dim={self.DIM})")

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        with StageLogger(f"Embed {len(chunks)} chunks (Gemini)", logger):
            for i, chunk in enumerate(chunks):
                chunk.embedding = self._embed_one(chunk.text, "RETRIEVAL_DOCUMENT")
                if (i + 1) % 50 == 0:
                    time.sleep(0.1)
        return chunks

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query, "RETRIEVAL_QUERY")

    def _embed_one(self, text: str, task: str) -> list[float]:
        payload = {
            "content": {"parts": [{"text": text}]},
            "taskType": task,
            "outputDimensionality": 1024,
        }
        for attempt in range(3):
            try:
                r = req.post(
                    self.endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self._key,
                    },
                    json=payload,
                    timeout=10,
                )
                r.raise_for_status()
                return r.json()["embedding"]["values"]
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    raise
    

# ── Local BGE-M3 Embedder ─────────────────────────────────────────────────────

class LocalEmbedder:
    DIM   = 1024
    MODEL = "BAAI/bge-m3"

    def __init__(self, use_fp16: bool = True):
        self._model      = None
        self._model_type = None
        self._use_fp16   = use_fp16

    def _load(self):
        if self._model is not None:
            return
        with StageLogger(f"Load {self.MODEL}", logger):
            try:
                from FlagEmbedding import BGEM3FlagModel
                self._model      = BGEM3FlagModel(self.MODEL, use_fp16=self._use_fp16)
                self._model_type = "flag"
                logger.info(f"BGE-M3 via FlagEmbedding (fp16={self._use_fp16})")
            except Exception as e:
                logger.warning(f"FlagEmbedding lỗi ({e}) — fallback sentence-transformers")
                from sentence_transformers import SentenceTransformer
                self._model      = SentenceTransformer(self.MODEL)
                self._model_type = "st"
                logger.info("BGE-M3 via sentence-transformers")

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        self._load()
        texts = [c.text for c in chunks]
        with StageLogger(f"Embed {len(chunks)} chunks (BGE-M3 local)", logger):
            embeddings = self._batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks

    def embed_query(self, query: str) -> list[float]:
        self._load()
        return self._batch([query])[0]

    def _batch(self, texts: list[str]) -> list[list[float]]:
        all_embs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            if self._model_type == "flag":
                out = self._model.encode(
                    batch,
                    batch_size=BATCH_SIZE,
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                all_embs.extend(out["dense_vecs"].tolist())
            else:
                vecs = self._model.encode(
                    batch,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                all_embs.extend(vecs.tolist())
        return all_embs


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder():
    if PROVIDER == "gemini":
        return GeminiEmbedder()
    return LocalEmbedder()


def get_embedding_dim() -> int:
    return GeminiEmbedder.DIM if PROVIDER == "gemini" else LocalEmbedder.DIM
#-----Nếu có mạng dùng api, không có mạng fallback local model. Cả 2 embedding dim=1024 để thống nhất với qdrant collection

def _has_internet() -> bool:
    try:
        socket.setdefaulttimeout(3)
        socket.create_connection(("8.8.8.8", 53))
        return True
    except OSError:
        return False

def get_embedder():
    if PROVIDER == "gemini":
        if _has_internet():
            return GeminiEmbedder()
        else:
            logger.warning("Không có mạng — fallback sang LocalEmbedder")
            return LocalEmbedder()
    return LocalEmbedder()

def get_embedding_dim() -> int:
    if PROVIDER == "gemini" and _has_internet():
        return GeminiEmbedder.DIM
    return LocalEmbedder.DIM

def get_collection_name() -> str:
    if PROVIDER == "gemini" and _has_internet():
        return "rag_chunks_gemini"
    return "rag_chunks_local"
