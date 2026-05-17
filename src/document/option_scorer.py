"""
Scores candidate options against retrieved document evidence.
Uses reranker on each (question + option) vs evidence pair.
"""
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.schemas import DocumentChunk
from src.utils.logging_utils import logger


class OptionScorer:
    """
    Scores each answer option against retrieved evidence.

    Strategy:
    1. For each option, form a query: "question [SEP] option_text"
    2. Score each query against top evidence chunks
    3. Select the option with the highest evidence support
    """

    def __init__(self, reranker=None):
        """
        reranker: a CrossEncoder or similar model with .predict(pairs) method.
        If None, falls back to simple keyword overlap scoring.
        """
        self.reranker = reranker

    def score_options(
        self,
        question: str,
        options: Dict[str, str],
        evidence_chunks: List[DocumentChunk],
        top_k_evidence: int = 5,
    ) -> Dict[str, float]:
        """
        Score each option against evidence.
        Returns {option_letter: score}.
        """
        if not options or not evidence_chunks:
            return {}

        evidence_texts = [c.text for c in evidence_chunks[:top_k_evidence]]

        if self.reranker is not None:
            return self._score_with_reranker(question, options, evidence_texts)
        else:
            return self._score_with_overlap(question, options, evidence_texts)

    def _score_with_reranker(
        self,
        question: str,
        options: Dict[str, str],
        evidence_texts: List[str],
    ) -> Dict[str, float]:
        """Score using cross-encoder reranker."""
        scores = {}

        for letter, option_text in options.items():
            query = f"{question} {option_text}"
            pairs = [(query, ev) for ev in evidence_texts]

            try:
                pair_scores = self.reranker.predict(pairs)
                # Take max score across evidence chunks
                scores[letter] = float(np.max(pair_scores))
            except Exception as e:
                logger.warning(f"Reranker error for option {letter}: {e}")
                scores[letter] = 0.0

        return scores

    def _score_with_overlap(
        self,
        question: str,
        options: Dict[str, str],
        evidence_texts: List[str],
    ) -> Dict[str, float]:
        """
        Fallback: score using keyword overlap between option and evidence.
        """
        evidence_combined = " ".join(evidence_texts).lower()
        evidence_words = set(evidence_combined.split())

        scores = {}
        for letter, option_text in options.items():
            option_words = set(option_text.lower().split())
            if not option_words:
                scores[letter] = 0.0
                continue

            # Overlap ratio
            overlap = len(option_words & evidence_words)
            scores[letter] = overlap / len(option_words)

        return scores

    def select_best(
        self,
        scores: Dict[str, float],
        multi_threshold: float = 0.95,
    ) -> str:
        """
        Select the best answer option(s).
        If multiple options are very close to the top, return multiple.

        Returns: "A" or "A,B" etc.
        """
        if not scores:
            return "A"  # Default fallback

        sorted_options = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_score = sorted_options[0][1]

        if best_score == 0:
            return sorted_options[0][0]

        # Check if multiple options are close
        selected = [sorted_options[0][0]]
        for letter, score in sorted_options[1:]:
            if score >= best_score * multi_threshold:
                selected.append(letter)

        return ",".join(sorted(selected))
