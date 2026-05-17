"""
Wrapper for the embedding model used in document/API retrieval.
Supports sentence-transformers models (bge-m3, Qwen3-Embedding).
"""
import numpy as np
from typing import List, Optional
from src.utils.logging_utils import logger
from configs.model_config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE,
    EMBEDDING_NORMALIZE, EMBEDDING_QUERY_INSTRUCTION_DOC,
)


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding models."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.device = device
        self.model = None

    def load(self):
        """Load the model lazily."""
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self.model = SentenceTransformer(self.model_name, **kwargs)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(
        self, texts: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        normalize: bool = EMBEDDING_NORMALIZE,
        instruction: str = "",
    ) -> np.ndarray:
        """Encode texts into embeddings."""
        self.load()
        if instruction:
            texts = [instruction + t for t in texts]
        embeddings = self.model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=normalize, show_progress_bar=False,
        )
        return np.array(embeddings)

    def encode_query(self, query: str, instruction: str = EMBEDDING_QUERY_INSTRUCTION_DOC) -> np.ndarray:
        """Encode a single query with instruction prefix."""
        return self.encode([query], instruction=instruction)[0]
