"""
Question-only routing features for the branch selector.

These features summarize how strongly the question matches the document and
API resources. They never inspect the note field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.api.api_retriever import APIRetriever
from src.api.slot_extractor import SlotExtractor
from src.document.retriever import DocumentRetriever
from src.schemas import APIEntry
from src.utils.logging_utils import logger


ROUTING_FEATURE_NAMES = [
    "doc_top_score",
    "doc_second_score",
    "doc_margin",
    "doc_hit_count",
    "doc_id_in_question",
    "api_top_score",
    "api_second_score",
    "api_margin",
    "api_hit_count",
    "api_slot_coverage",
    "api_slot_count",
    "api_body_param_count",
]


@dataclass
class RoutingFeatureExtractor:
    """Extract lightweight retrieval and schema-feasibility signals."""

    document_retriever: Optional[DocumentRetriever] = None
    api_retriever: Optional[APIRetriever] = None
    slot_extractor: Optional[SlotExtractor] = None
    top_k: int = 5

    @property
    def feature_names(self) -> List[str]:
        return list(ROUTING_FEATURE_NAMES)

    def transform(self, questions: List[str]) -> np.ndarray:
        rows = [self.extract(q) for q in questions]
        if not rows:
            return np.zeros((0, len(ROUTING_FEATURE_NAMES)), dtype=float)
        return np.asarray(rows, dtype=float)

    def extract(self, question: str) -> List[float]:
        values = self.extract_map(question)
        return [values[name] for name in ROUTING_FEATURE_NAMES]

    def extract_map(self, question: str) -> Dict[str, float]:
        doc_scores = self._document_scores(question)
        api_scores, best_api = self._api_scores(question)
        slot_coverage, slot_count, body_param_count = self._api_slot_features(
            question,
            best_api,
        )

        return {
            "doc_top_score": doc_scores["top"],
            "doc_second_score": doc_scores["second"],
            "doc_margin": doc_scores["margin"],
            "doc_hit_count": doc_scores["hit_count"],
            "doc_id_in_question": doc_scores["doc_id_in_question"],
            "api_top_score": api_scores["top"],
            "api_second_score": api_scores["second"],
            "api_margin": api_scores["margin"],
            "api_hit_count": api_scores["hit_count"],
            "api_slot_coverage": slot_coverage,
            "api_slot_count": slot_count,
            "api_body_param_count": body_param_count,
        }

    def _document_scores(self, question: str) -> Dict[str, float]:
        if self.document_retriever is None:
            return self._empty_score_map(doc_id_in_question=0.0)

        try:
            doc_id = self.document_retriever._detect_doc_id(question)
            results = self.document_retriever.retrieve_with_metadata_filter(
                question,
                top_k=self.top_k,
            )
            scores = [score for _, score in results]
            stats = self._score_stats(scores)
            stats["doc_id_in_question"] = 1.0 if doc_id else 0.0
            return stats
        except Exception as exc:
            logger.debug(f"Could not extract document routing features: {exc}")
            return self._empty_score_map(doc_id_in_question=0.0)

    def _api_scores(self, question: str) -> tuple[Dict[str, float], Optional[APIEntry]]:
        if self.api_retriever is None:
            return self._empty_score_map(), None

        try:
            results = self.api_retriever.retrieve(question, top_k=self.top_k)
            scores = [score for _, score in results]
            best_api = results[0][0] if results else None
            return self._score_stats(scores), best_api
        except Exception as exc:
            logger.debug(f"Could not extract API routing features: {exc}")
            return self._empty_score_map(), None

    def _api_slot_features(
        self,
        question: str,
        best_api: Optional[APIEntry],
    ) -> tuple[float, float, float]:
        if best_api is None or self.slot_extractor is None:
            return 0.0, 0.0, 0.0

        try:
            slots = self.slot_extractor.extract(question, best_api)
        except Exception as exc:
            logger.debug(f"Could not extract API slot routing features: {exc}")
            return 0.0, 0.0, float(len(best_api.body_params))

        body_params = [p.strip() for p in best_api.body_params if str(p).strip()]
        if not body_params:
            return 0.0, float(len(slots)), 0.0

        matched = sum(1 for param in body_params if param in slots)
        return (
            matched / max(len(body_params), 1),
            float(len(slots)),
            float(len(body_params)),
        )

    def _score_stats(self, scores: List[float]) -> Dict[str, float]:
        top = self._scale_score(scores[0]) if scores else 0.0
        second = self._scale_score(scores[1]) if len(scores) > 1 else 0.0
        return {
            "top": top,
            "second": second,
            "margin": max(top - second, 0.0),
            "hit_count": float(len(scores)),
        }

    def _empty_score_map(self, doc_id_in_question: float = 0.0) -> Dict[str, float]:
        return {
            "top": 0.0,
            "second": 0.0,
            "margin": 0.0,
            "hit_count": 0.0,
            "doc_id_in_question": doc_id_in_question,
        }

    def _scale_score(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric <= 0:
            return 0.0
        return float(np.log1p(numeric))
