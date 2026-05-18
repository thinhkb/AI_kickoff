"""
Scores candidate options against retrieved document evidence.
Uses reranker on each (question + option) vs evidence pair.
"""
from typing import Dict, List, Tuple, Optional
import re
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
        retriever=None,
    ) -> Dict[str, float]:
        """
        Score each option against evidence.
        Returns {option_letter: score}.
        """
        if not options or not evidence_chunks:
            return {}

        evidence_texts = [c.text for c in evidence_chunks[:top_k_evidence]]

        if self.reranker is not None and retriever is not None:
            return self._score_with_reranker_and_lexical(
                question, options, evidence_chunks, retriever
            )
        elif self.reranker is not None:
            return self._score_with_reranker(question, options, evidence_texts)
        elif retriever is not None:
            return self._score_with_hybrid_lexical(
                question, options, evidence_chunks, retriever
            )
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

    def _score_with_reranker_and_lexical(
        self,
        question: str,
        options: Dict[str, str],
        evidence_chunks: List[DocumentChunk],
        retriever,
    ) -> Dict[str, float]:
        """Fuse cross-encoder evidence scores with lexical per-option retrieval."""
        lexical = self._score_with_hybrid_lexical(question, options, evidence_chunks, retriever)
        rerank_scores = {}
        doc_hint = self._extract_doc_hint(question)

        for letter, option_text in options.items():
            option_query = f"{question} {option_text}"
            option_results = retriever.retrieve_with_metadata_filter(option_query, top_k=8)
            hint_results = []
            if doc_hint:
                hint_results = retriever.retrieve_with_metadata_filter(
                    f"{doc_hint} {option_text}",
                    top_k=8,
                )

            seen = set()
            evidence_texts = []
            for chunk in list(evidence_chunks[:8]) + [c for c, _ in option_results] + [c for c, _ in hint_results]:
                key = (chunk.doc_id, chunk.page, chunk.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                evidence_texts.append(chunk.text)
                if len(evidence_texts) >= 16:
                    break

            if not evidence_texts:
                rerank_scores[letter] = 0.0
                continue

            try:
                query = f"{question} {option_text}"
                pair_scores = self.reranker.predict([(query, text) for text in evidence_texts])
                rerank_scores[letter] = float(np.max(pair_scores))
            except Exception as e:
                logger.warning(f"Reranker error for option {letter}: {e}")
                rerank_scores[letter] = 0.0

        lexical_norm = self._minmax(lexical)
        rerank_norm = self._minmax(rerank_scores)
        return {
            letter: 0.55 * rerank_norm.get(letter, 0.0) + 0.45 * lexical_norm.get(letter, 0.0)
            for letter in options
        }

    def _score_with_hybrid_lexical(
        self,
        question: str,
        options: Dict[str, str],
        evidence_chunks: List[DocumentChunk],
        retriever,
    ) -> Dict[str, float]:
        """
        Accuracy-oriented lexical scoring for Vietnamese multiple-choice QA.

        The old scorer only checked whether option tokens appeared in the
        already-retrieved evidence, which made many ties and multi-answer
        outputs. This combines:
        - normalized option/evidence overlap,
        - BM25 retrieval score for question + option,
        - BM25 retrieval score for option text alone.
        """
        overlap_scores = self._score_with_normalized_overlap(
            question, options, evidence_chunks[:10]
        )
        qopt_scores = {}
        opt_scores = {}

        doc_hint = self._extract_doc_hint(question)
        for letter, option_text in options.items():
            qopt_query = f"{question} {option_text}"
            qopt_results = retriever.retrieve_with_metadata_filter(qopt_query, top_k=5)
            qopt_scores[letter] = self._ranked_score(qopt_results)

            opt_query = f"{doc_hint} {option_text}".strip() if doc_hint else option_text
            opt_results = retriever.retrieve_with_metadata_filter(opt_query, top_k=5)
            opt_scores[letter] = self._ranked_score(opt_results)

        overlap_norm = self._minmax(overlap_scores)
        qopt_norm = self._minmax(qopt_scores)
        opt_norm = self._minmax(opt_scores)

        scores = {}
        for letter in options:
            scores[letter] = (
                0.45 * overlap_norm.get(letter, 0.0)
                + 0.35 * qopt_norm.get(letter, 0.0)
                + 0.20 * opt_norm.get(letter, 0.0)
            )
        return scores

    def _score_with_normalized_overlap(
        self,
        question: str,
        options: Dict[str, str],
        evidence_chunks: List[DocumentChunk],
    ) -> Dict[str, float]:
        evidence = " ".join(c.text for c in evidence_chunks)
        evidence_tokens = set(self._tokenize(evidence))
        question_tokens = set(self._tokenize(question))
        evidence_lower = evidence.lower()

        scores = {}
        for letter, option_text in options.items():
            option_tokens = set(self._tokenize(option_text))
            useful_tokens = option_tokens - question_tokens
            if not useful_tokens:
                useful_tokens = option_tokens
            overlap = len(useful_tokens & evidence_tokens) / max(len(useful_tokens), 1)
            phrase_bonus = 1.0 if option_text.lower() in evidence_lower else 0.0
            scores[letter] = overlap + phrase_bonus
        return scores

    def _tokenize(self, text: str) -> List[str]:
        stopwords = {
            "và", "là", "của", "có", "trong", "theo", "được", "từ",
            "một", "những", "các", "về", "để", "nào", "gì", "với",
            "khi", "nếu", "thì", "cho", "trên", "dưới", "bao", "nhiêu",
            "đâu", "hãy", "hay", "không", "phải", "chính", "chủ", "yếu",
            "đã", "bị", "vào", "ra", "ở", "này", "đó",
        }
        return [
            t for t in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
            if len(t) > 1 and t not in stopwords
        ]

    def _extract_doc_hint(self, question: str) -> str:
        public_match = re.search(r"\bPublic[\s_-]*(\d{1,4})\b", question, re.IGNORECASE)
        if public_match:
            return f"Public_{int(public_match.group(1)):03d}"
        td_match = re.search(r"\bTD\s*(\d{1,4})\b", question, re.IGNORECASE)
        if td_match:
            return f"Public_{int(td_match.group(1)):03d}"
        return ""

    def _ranked_score(self, results: List[Tuple[DocumentChunk, float]]) -> float:
        return float(sum(score / (rank + 1) for rank, (_, score) in enumerate(results)))

    def _minmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_v, max_v = min(values), max(values)
        span = max_v - min_v
        if span <= 1e-12:
            return {k: 0.0 for k in scores}
        return {k: (v - min_v) / span for k, v in scores.items()}

    def select_best(
        self,
        scores: Dict[str, float],
        multi_threshold: Optional[float] = None,
    ) -> str:
        """
        Select the best answer option(s).
        If multiple options are very close to the top, return multiple.

        Returns: "A" or "A,B" etc.
        """
        if not scores:
            return "A"  # Default fallback

        sorted_options = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        best_score = sorted_options[0][1]

        if multi_threshold is None:
            return sorted_options[0][0]

        if best_score == 0:
            return sorted_options[0][0]

        # Check if multiple options are close
        selected = [sorted_options[0][0]]
        for letter, score in sorted_options[1:]:
            if score >= best_score * multi_threshold:
                selected.append(letter)

        return ",".join(sorted(selected))
