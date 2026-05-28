"""
Scores candidate options against retrieved document evidence.
Uses reranker on each (question + option) vs evidence pair.
"""
from typing import Dict, List, Tuple, Optional
import re
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from src.schemas import DocumentChunk
from src.utils.logging_utils import logger


# ---------------------------------------------------------------------------
# Vietnamese negation patterns (generic, not question-specific)
# ---------------------------------------------------------------------------
_NEGATION_PATTERNS = [
    r"\bkhông\s+phải\b",
    r"\bkhông\s+thuộc\b",
    r"\bkhông\s+đúng\b",
    r"\bkhông\s+phải\s+là\b",
    r"\bkhông\s+áp\s+dụng\b",
    r"\bkhông\s+bao\s+gồm\b",
    r"\bkhông\s+nằm\s+trong\b",
    r"\bkhông\s+được\s+đề\s+cập\b",
    r"\bkhông\s+liên\s+quan\b",
    r"\bnào\s+(?:là\s+)?(?:sai|không\s+đúng)\b",
    r"\bđâu\s+không\s+phải\b",
    r"\bnào\s+không\s+(?:phải|thuộc|đúng)\b",
    r"\bngoại\s+trừ\b",
    r"\bchưa\s+đúng\b",
    r"\bsai\s+(?:về|khi|là)\b",
]


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

        Combines:
        - normalized option/evidence overlap,
        - BM25 retrieval score for question + option,
        - BM25 retrieval score for option text alone,
        - TF-IDF cosine similarity (when sklearn is available).
        """
        overlap_scores = self._score_with_normalized_overlap(
            question, options, evidence_chunks[:20]
        )
        proximity_scores = self._score_with_proximity(
            question, options, evidence_chunks[:25]
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
        proximity_norm = self._minmax(proximity_scores)
        qopt_norm = self._minmax(qopt_scores)
        opt_norm = self._minmax(opt_scores)

        # TF-IDF scoring (sklearn-based) using pre-fitted corpus vectorizer
        tfidf_scores = self._score_with_tfidf_sklearn(
            question, options, evidence_chunks[:25], retriever
        )
        tfidf_norm = self._minmax(tfidf_scores)

        # Detect short options (like numbers, versions, or short keywords)
        is_short_options = all(len(re.findall(r"\w+", opt.lower(), flags=re.UNICODE)) <= 2 for opt in options.values())

        scores = {}
        if is_short_options:
            # For short options, proximity co-occurrence is extremely reliable, 
            # while bag-of-words BM25/TF-IDF has high frequency noise.
            for letter in options:
                scores[letter] = (
                    0.0125 * overlap_norm.get(letter, 0.0)
                    + 0.95 * proximity_norm.get(letter, 0.0)
                    + 0.0125 * tfidf_norm.get(letter, 0.0)
                    + 0.0125 * qopt_norm.get(letter, 0.0)
                    + 0.0125 * opt_norm.get(letter, 0.0)
                )
        else:
            # 5-way hybrid fusion
            for letter in options:
                scores[letter] = (
                    0.20 * overlap_norm.get(letter, 0.0)
                    + 0.35 * proximity_norm.get(letter, 0.0)
                    + 0.10 * tfidf_norm.get(letter, 0.0)
                    + 0.20 * qopt_norm.get(letter, 0.0)
                    + 0.15 * opt_norm.get(letter, 0.0)
                )
        return scores

    def _score_with_tfidf_sklearn(
        self,
        question: str,
        options: Dict[str, str],
        evidence_chunks: List[DocumentChunk],
        retriever=None,
    ) -> Dict[str, float]:
        """
        Use pre-fitted corpus TF-IDF vectorizer to compute raw TF-IDF matching dot product
        between each option and the evidence chunks.
        """
        if not _HAS_SKLEARN or not evidence_chunks:
            return {}

        vectorizer = None
        if retriever is not None and hasattr(retriever, "tfidf_vectorizer"):
            vectorizer = retriever.tfidf_vectorizer

        if vectorizer is None:
            # Fallback to local fit if corpus vectorizer is not built
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    token_pattern=r'\w{2,}',
                    max_features=5000,
                    sublinear_tf=True,
                    norm=None,
                )
                texts = [c.text for c in evidence_chunks]
                vectorizer.fit(texts)
            except Exception as e:
                logger.warning(f"Local TF-IDF fit error: {e}")
                return {}

        try:
            evidence_combined = " ".join(c.text for c in evidence_chunks)
            evidence_vec = vectorizer.transform([evidence_combined])
            
            scores = {}
            for letter, opt_text in options.items():
                opt_vec = vectorizer.transform([opt_text])
                # Compute raw dot product (sum of overlapping TF-IDF weights)
                dot_prod = float(opt_vec.dot(evidence_vec.T).toarray()[0][0])
                scores[letter] = dot_prod
            return scores
        except Exception as e:
            logger.warning(f"TF-IDF scoring error: {e}")
            return {}

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
            overlap = len(useful_tokens & evidence_tokens) / max(len(useful_tokens) ** 0.5, 1.0)
            phrase_bonus = 1.0 if option_text.lower() in evidence_lower else 0.0
            scores[letter] = overlap + phrase_bonus
        return scores

    def _score_with_proximity(
        self,
        question: str,
        options: Dict[str, str],
        evidence_chunks: List[DocumentChunk],
    ) -> Dict[str, float]:
        """
        Score options based on proximity of option-specific words to question keywords
        within sentences in the evidence, weighted by option match ratio and keyword overlap.
        """
        evidence_text = " ".join(c.text for c in evidence_chunks)
        sentences = re.split(r"[.!?\n]+", evidence_text)
        
        q_tokens = self._tokenize(question)
        static_stop = {"cảm", "biến", "xe", "tự", "hành", "tài", "liệu", "theo", "quy", "định", "nội", "dung"}
        q_keywords = [t for t in q_tokens if (len(t) > 1 or t.isdigit()) and t not in static_stop]
        
        if not q_keywords:
            q_keywords = [t for t in q_tokens if len(t) > 1 or t.isdigit()]
            
        if not q_keywords:
            return {k: 0.0 for k in options}

        scores = {}
        for letter, option_text in options.items():
            opt_tokens = self._tokenize(option_text)
            useful_tokens = [t for t in opt_tokens if t not in q_tokens]
            if not useful_tokens:
                useful_tokens = opt_tokens
            
            max_score = 0.0
            for sent in sentences:
                sent_words = re.findall(r"\w+", sent.lower(), flags=re.UNICODE)
                if not sent_words:
                    continue
                
                opt_indices = [i for i, w in enumerate(sent_words) if w in useful_tokens]
                if not opt_indices:
                    continue
                
                match_ratio = len(set(sent_words) & set(useful_tokens)) / len(useful_tokens)
                
                q_indices = {kw: [i for i, w in enumerate(sent_words) if w == kw] for kw in q_keywords}
                
                overlap_count = 0
                dist_sum = 0.0
                for kw, indices in q_indices.items():
                    if not indices:
                        continue
                    overlap_count += 1
                    min_dist = min(abs(q_idx - opt_idx) for q_idx in indices for opt_idx in opt_indices)
                    dist_sum += 1.0 / (min_dist + 1.0)
                
                if overlap_count > 0:
                    overlap_ratio = overlap_count / len(q_keywords)
                    avg_prox = dist_sum / overlap_count
                    score = match_ratio * (overlap_ratio + avg_prox)
                    if score > max_score:
                        max_score = score
            
            scores[letter] = max_score
        
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
            if (len(t) > 1 or t.isdigit()) and t not in stopwords
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

    # ------------------------------------------------------------------
    # Negation detection
    # ------------------------------------------------------------------
    @staticmethod
    def is_negation_question(question: str) -> bool:
        """Detect whether a question asks for the INCORRECT / NOT applicable option."""
        q_lower = question.lower()
        for pat in _NEGATION_PATTERNS:
            if re.search(pat, q_lower):
                return True
        return False

    def select_best(
        self,
        scores: Dict[str, float],
        multi_threshold: Optional[float] = None,
        is_negation: bool = False,
    ) -> str:
        """
        Select the best answer option(s).
        If multiple options are very close to the top, return multiple.

        When *is_negation* is True, use confidence-based inversion:
        only pick the lowest-scoring option if there's a clear gap
        separating it from the rest.

        Returns: "A" or "A,B" etc.
        """
        if not scores:
            return "A"  # Default fallback

        sorted_options = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        if is_negation and len(scores) >= 3:
            sorted_asc = sorted(scores.items(), key=lambda x: (x[1], x[0]))
            lowest_letter, lowest_score = sorted_asc[0]
            second_lowest_score = sorted_asc[1][1]
            highest_score = sorted_asc[-1][1]

            score_range = highest_score - lowest_score
            gap_to_second = second_lowest_score - lowest_score

            # Only apply negation if the lowest is a clear outlier:
            # gap between lowest and second-lowest >= 20% of range.
            if score_range > 0.05 and gap_to_second >= 0.20 * score_range:
                return lowest_letter

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
