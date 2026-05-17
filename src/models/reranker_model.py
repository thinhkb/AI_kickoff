"""
Wrapper for the reranking model (cross-encoder).
"""
from typing import List, Tuple
from src.utils.logging_utils import logger
from configs.model_config import RERANKER_MODEL_NAME, RERANKER_BATCH_SIZE


class RerankerModel:
    """Wrapper for cross-encoder reranking models."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or RERANKER_MODEL_NAME
        self.device = device
        self.model = None

    def load(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Loaded reranker model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise

    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = RERANKER_BATCH_SIZE):
        """Score query-document pairs."""
        self.load()
        return self.model.predict(pairs, batch_size=batch_size)
