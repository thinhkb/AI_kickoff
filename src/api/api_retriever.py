"""
Retrieves top API candidates from the API registry using BM25.
"""
import numpy as np
from typing import List, Tuple, Optional

from rank_bm25 import BM25Okapi

from src.schemas import APIEntry
from src.api.api_catalog_loader import build_api_search_text
from src.utils.logging_utils import logger
from configs.constants import API_RETRIEVAL_TOP_K


class APIRetriever:
    """
    Retrieves the most relevant APIs for a given question.
    Uses BM25 and optionally dense embeddings.
    """

    def __init__(self):
        self.apis: List[APIEntry] = []
        self.search_texts: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None

    def build_index(
        self,
        apis: List[APIEntry],
        embeddings: Optional[np.ndarray] = None,
    ):
        """Build BM25 index from API entries."""
        self.apis = apis
        self.search_texts = [build_api_search_text(api) for api in apis]

        tokenized = [text.lower().split() for text in self.search_texts]
        self.bm25 = BM25Okapi(tokenized)

        if embeddings is not None:
            self.embeddings = embeddings

        logger.info(f"API retriever index built with {len(apis)} APIs")

    def retrieve(
        self,
        query: str,
        top_k: int = API_RETRIEVAL_TOP_K,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[APIEntry, float]]:
        """
        Retrieve top-k API candidates.
        """
        if self.bm25 is None:
            return []

        # BM25 retrieval
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        if query_embedding is not None and self.embeddings is not None:
            # Hybrid: combine BM25 + dense
            dense_scores = np.dot(self.embeddings, query_embedding)

            # Normalize both
            if scores.max() > 0:
                bm25_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            else:
                bm25_norm = scores

            if dense_scores.max() > dense_scores.min():
                dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
            else:
                dense_norm = dense_scores

            scores = 0.4 * bm25_norm + 0.6 * dense_norm

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.apis[idx], float(scores[idx])))

        return results

    def retrieve_all_scored(
        self,
        query: str,
    ) -> List[Tuple[APIEntry, float]]:
        """
        Score all APIs (useful when registry is small enough to rerank all).
        """
        return self.retrieve(query, top_k=len(self.apis))
