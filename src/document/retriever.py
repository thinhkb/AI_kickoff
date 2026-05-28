"""
Searches relevant document chunks using hybrid BM25 + dense retrieval.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from rank_bm25 import BM25Okapi

from src.schemas import DocumentChunk
from src.utils.logging_utils import logger
from configs.constants import DOC_RETRIEVAL_TOP_K


class DocumentRetriever:
    """
    Hybrid retriever combining BM25 (lexical) and dense embeddings.
    """

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None
        self._embedding_model = None

    def build_index(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[np.ndarray] = None,
    ):
        """
        Build BM25 index and optionally store precomputed embeddings.
        """
        self.chunks = chunks
        logger.info(f"Building document index with {len(chunks)} chunks...")

        # BM25 index
        tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Dense embeddings
        if embeddings is not None:
            self.embeddings = embeddings
            logger.info(f"Loaded {embeddings.shape[0]} precomputed embeddings")

    def _tokenize(self, text: str) -> List[str]:
        """Proper tokenization stripping punctuation and stopwords, preserving digits."""
        import re
        stopwords = {
            "và", "là", "của", "có", "trong", "theo", "được", "từ",
            "một", "những", "các", "về", "để", "nào", "gì", "với",
            "khi", "nếu", "thì", "cho", "trên", "dưới", "bao", "nhiêu",
            "đâu", "hãy", "hay", "không", "phải", "chính", "chủ", "yếu",
            "đã", "bị", "vào", "ra", "ở", "này", "đó",
        }
        return [
            t for t in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
            if (len(t) > 1 or t.isdigit()) and t not in stopwords
        ]

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = DOC_RETRIEVAL_TOP_K,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve using BM25 only."""
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx])))
        return results

    def retrieve_dense(
        self,
        query_embedding: np.ndarray,
        top_k: int = DOC_RETRIEVAL_TOP_K,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve using dense embeddings only."""
        if self.embeddings is None:
            return []

        # Cosine similarity
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

    def retrieve_hybrid(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        top_k: int = DOC_RETRIEVAL_TOP_K,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid retrieval combining BM25 and dense scores.
        Scores are min-max normalized before fusion.
        """
        # BM25 scores
        bm25_results = self.retrieve_bm25(query, top_k=top_k * 2)

        if query_embedding is not None and self.embeddings is not None:
            # Dense scores
            dense_results = self.retrieve_dense(query_embedding, top_k=top_k * 2)

            # Merge scores
            chunk_scores: Dict[int, float] = {}

            # Normalize BM25 scores
            if bm25_results:
                bm25_max = max(s for _, s in bm25_results) or 1.0
                bm25_min = min(s for _, s in bm25_results)
                bm25_range = bm25_max - bm25_min or 1.0
                for chunk, score in bm25_results:
                    normalized = (score - bm25_min) / bm25_range
                    chunk_scores[chunk.chunk_id] = bm25_weight * normalized

            # Normalize dense scores
            if dense_results:
                dense_max = max(s for _, s in dense_results) or 1.0
                dense_min = min(s for _, s in dense_results)
                dense_range = dense_max - dense_min or 1.0
                for chunk, score in dense_results:
                    normalized = (score - dense_min) / dense_range
                    existing = chunk_scores.get(chunk.chunk_id, 0)
                    chunk_scores[chunk.chunk_id] = existing + dense_weight * normalized

            # Build final results
            all_chunks = {c.chunk_id: c for c, _ in bm25_results + dense_results}
            sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)

            results = []
            for cid in sorted_ids[:top_k]:
                results.append((all_chunks[cid], chunk_scores[cid]))
            return results
        else:
            return bm25_results[:top_k]

    def retrieve_with_metadata_filter(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = DOC_RETRIEVAL_TOP_K,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve with optional metadata filtering.
        If a doc_id is mentioned in the query (e.g., Public_441),
        filter chunks to that document first.
        """
        # Try to detect doc_id from query
        if doc_id is None:
            doc_id = self._detect_doc_id(query)

        if doc_id:
            # Filter chunks to specific document
            filtered = [c for c in self.chunks if c.doc_id == doc_id]
            if filtered:
                tokenized = [self._tokenize(c.text) for c in filtered]
                local_bm25 = BM25Okapi(tokenized)
                scores = local_bm25.get_scores(self._tokenize(query))
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = [(filtered[i], float(scores[i])) for i in top_indices if scores[i] > 0]
                if results:
                    return results
                # Broad doc-summary questions often mention only the document id
                # plus generic words. Returning leading chunks is better than an
                # empty result because option scoring can still match content.
                return [
                    (chunk, 1.0 / (idx + 1))
                    for idx, chunk in enumerate(filtered[:top_k])
                ]

        # Fall back to full retrieval
        return self.retrieve_bm25(query, top_k)

    def _detect_doc_id(self, query: str) -> Optional[str]:
        """Detect Public_123, Public 123, or TD123 references."""
        import re

        public_match = re.search(r"\bPublic[\s_-]*(\d{1,4})\b", query, re.IGNORECASE)
        if public_match:
            return f"Public_{int(public_match.group(1)):03d}"

        td_match = re.search(r"\bTD\s*(\d{1,4})\b", query, re.IGNORECASE)
        if td_match:
            return f"Public_{int(td_match.group(1)):03d}"

        return None

    def save_index(self, path: str | Path):
        """Save chunks and metadata to disk."""
        from src.utils.io_utils import write_jsonl
        write_jsonl([c.to_dict() for c in self.chunks], path)

    def load_index(self, path: str | Path):
        """Load chunks from disk and rebuild BM25."""
        from src.utils.io_utils import read_jsonl
        data = read_jsonl(path)
        self.chunks = [DocumentChunk.from_dict(d) for d in data]
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        
        # Build TF-IDF vectorizer fitted on the corpus
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [c.text for c in self.chunks]
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'\w{2,}',
                sublinear_tf=True,
                norm=None,
            )
            self.tfidf_vectorizer.fit(texts)
            logger.info("Fitted corpus-level TF-IDF vectorizer")
        except Exception as e:
            logger.warning(f"Could not fit corpus TF-IDF vectorizer: {e}")
            self.tfidf_vectorizer = None
            
        logger.info(f"Loaded {len(self.chunks)} document chunks")
