"""
Main inference pipeline: orchestrates selector, document solver, and API solver.
"""
import json
import os
from typing import Optional, Dict, Any
from src.schemas import InputSample, PredictionResult
from src.preprocess.question_normalizer import normalize_question
from src.selector.selector_model import SelectorModel
from src.document.document_solver import DocumentSolver
from src.document.retriever import DocumentRetriever
from src.document.option_scorer import OptionScorer
from src.document.option_parser import parse_options
from src.api.api_solver import APISolver
from src.api.api_retriever import APIRetriever
from src.api.api_reranker import APIReranker
from src.api.api_catalog_loader import load_api_catalog, load_alias_dictionary
from src.models.reranker_model import RerankerModel
from src.utils.timer import Timer
from src.utils.logging_utils import logger
from configs.paths import (
    SELECTOR_MODEL_DIR, DOCUMENT_CHUNKS_FILE,
    API_CONFIG_FILE, API_REGISTRY_FILE,
)
from configs.constants import FUNC_CALL_DOCUMENT, FUNC_CALL_API


class Pipeline:
    """End-to-end inference pipeline."""

    def __init__(self):
        self.selector: Optional[SelectorModel] = None
        self.document_solver: Optional[DocumentSolver] = None
        self.api_solver: Optional[APISolver] = None
        self._loaded = False

    def load(self):
        """Load all models and indexes."""
        logger.info("Loading pipeline...")
        reranker_model = self._load_optional_reranker()

        # 1. Load selector
        self.selector = SelectorModel()
        try:
            self.selector.load(SELECTOR_MODEL_DIR)
        except Exception as e:
            logger.warning(f"Selector model not found: {e}. Run train_selector.py first.")

        # 2. Load document retriever
        doc_retriever = DocumentRetriever()
        try:
            doc_retriever.load_index(DOCUMENT_CHUNKS_FILE)
        except Exception as e:
            logger.warning(f"Document index not found: {e}. Run build_document_kb.py first.")

        self.document_solver = DocumentSolver(
            retriever=doc_retriever,
            option_scorer=OptionScorer(reranker=reranker_model),
        )

        # 3. Load API components
        try:
            apis = load_api_catalog(API_CONFIG_FILE)
            alias_dict = load_alias_dictionary(API_CONFIG_FILE)

            api_retriever = APIRetriever()
            api_retriever.build_index(apis)

            self.api_solver = APISolver(
                retriever=api_retriever,
                reranker=APIReranker(reranker_model=reranker_model),
                alias_dict=alias_dict,
            )
        except Exception as e:
            logger.warning(f"API components load error: {e}")

        self._loaded = True
        logger.info("Pipeline loaded successfully")

    def _load_optional_reranker(self):
        """Load a heavy cross-encoder only when explicitly requested."""
        if os.getenv("VIETTEL_USE_RERANKER", "0").lower() not in {"1", "true", "yes"}:
            return None
        try:
            device = os.getenv("VIETTEL_RERANKER_DEVICE") or None
            model_name = os.getenv("VIETTEL_RERANKER_MODEL") or None
            reranker = RerankerModel(model_name=model_name, device=device)
            reranker.load()
            return reranker
        except Exception as e:
            logger.warning(f"Optional reranker unavailable, using lexical scoring: {e}")
            return None

    def predict_one(self, sample: InputSample) -> PredictionResult:
        """Predict for a single sample."""
        timer = Timer().start()

        # Normalize question
        q = normalize_question(sample.question)

        # Select function
        if self.selector and self.selector._fitted:
            func_code = self.selector.predict(q)
        else:
            # Fallback: heuristic
            func_code = self._heuristic_select(q, sample.note)

        # ─── Routing safety net ───────────────────────────────────
        # If selector says call_document but note is empty (no A/B/C/D
        # options), the question is almost certainly call_api.
        # All genuine call_document questions in the contest have a note
        # with answer options.
        if func_code == FUNC_CALL_DOCUMENT and not sample.note:
            logger.info(
                f"  Routing override: call_document -> call_api "
                f"(note is empty, id={sample.id})"
            )
            func_code = FUNC_CALL_API

        # In the contest files, real call_document rows carry answer choices in
        # note. This catches selector false positives on document/table-price
        # questions such as TD640/TD643 that look numeric and API-like.
        if func_code == FUNC_CALL_API and sample.note and len(parse_options(sample.note)) >= 2:
            logger.info(
                f"  Routing override: call_api -> call_document "
                f"(multiple-choice note, id={sample.id})"
            )
            func_code = FUNC_CALL_DOCUMENT

        # Solve
        if func_code == FUNC_CALL_DOCUMENT:
            result = self.document_solver.solve(
                question=q, note=sample.note or "",
            )
        elif func_code == FUNC_CALL_API:
            result = self.api_solver.solve(question=q)
        else:
            result = json.dumps({"error": "Invalid function_code"})

        elapsed = timer.stop()

        return PredictionResult(
            id=sample.id,
            function_code=func_code,
            function_answer=result,
            time_response=elapsed,
        )

    def _heuristic_select(self, question: str, note: str = None) -> str:
        """Fallback heuristic when selector is not trained."""
        q_lower = question.lower()
        api_keywords = [
            "bao nhiêu", "slsx", "slnt", "nhân sự", "dự án",
            "leakage", "tháng", "quý", "năm 2025", "ttpm",
            "công ty", "gói thầu", "đấu thầu",
        ]
        if any(kw in q_lower for kw in api_keywords) and note is None:
            return FUNC_CALL_API
        return FUNC_CALL_DOCUMENT

    def predict_batch(self, samples: list[InputSample]) -> list[PredictionResult]:
        """Predict for a batch of samples."""
        results = []
        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i+1}/{len(samples)}...")
            results.append(self.predict_one(sample))
        return results
