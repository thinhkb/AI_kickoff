"""
Main inference pipeline: orchestrates selector, document solver, and API solver.
"""
import json
import os
from typing import Optional

from configs.constants import FUNC_CALL_API, FUNC_CALL_DOCUMENT
from configs.paths import API_CONFIG_FILE, DOCUMENT_CHUNKS_FILE, SELECTOR_MODEL_DIR
from src.api.api_catalog_loader import load_alias_dictionary, load_api_catalog
from src.api.api_retriever import APIRetriever
from src.api.api_reranker import APIReranker
from src.api.api_solver import APISolver
from src.document.document_solver import DocumentSolver
from src.document.option_scorer import OptionScorer
from src.document.retriever import DocumentRetriever
from src.models.reranker_model import RerankerModel
from src.preprocess.question_normalizer import normalize_question
from src.schemas import InputSample, PredictionResult
from src.selector.routing_features import RoutingFeatureExtractor
from src.selector.selector_model import SelectorModel
from src.utils.logging_utils import logger
from src.utils.timer import Timer


class Pipeline:
    """End-to-end inference pipeline."""

    def __init__(self):
        self.selector: Optional[SelectorModel] = None
        self.document_solver: Optional[DocumentSolver] = None
        self.api_solver: Optional[APISolver] = None
        self.routing_feature_extractor: Optional[RoutingFeatureExtractor] = None
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
        doc_index_loaded = False
        try:
            doc_retriever.load_index(DOCUMENT_CHUNKS_FILE)
            doc_index_loaded = True
        except Exception as e:
            logger.warning(f"Document index not found: {e}. Run build_document_kb.py first.")

        self.document_solver = DocumentSolver(
            retriever=doc_retriever,
            option_scorer=OptionScorer(reranker=reranker_model),
        )

        # 3. Load API components
        api_retriever = None
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

        if doc_index_loaded or api_retriever is not None:
            self.routing_feature_extractor = RoutingFeatureExtractor(
                document_retriever=doc_retriever if doc_index_loaded else None,
                api_retriever=api_retriever,
                slot_extractor=self.api_solver.slot_extractor if self.api_solver else None,
            )
        if self.selector:
            self.selector.set_routing_feature_extractor(
                self.routing_feature_extractor,
            )

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

        q = normalize_question(sample.question)

        if self.selector and self.selector._fitted:
            func_code = self.selector.predict(q)
        else:
            # Last-resort selector uses question-only retrieval evidence.
            func_code = self._heuristic_select(q)

        if func_code == FUNC_CALL_DOCUMENT:
            result = self.document_solver.solve(
                question=q,
                note=sample.note or "",
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

    def _heuristic_select(self, question: str) -> str:
        """Fallback selector when the trained model is unavailable."""
        if self.routing_feature_extractor is not None:
            features = self.routing_feature_extractor.extract_map(question)
            api_evidence = (
                features["api_top_score"]
                + features["api_margin"]
                + features["api_slot_coverage"]
            )
            doc_evidence = (
                features["doc_top_score"]
                + features["doc_margin"]
                + features["doc_id_in_question"]
            )
            if api_evidence > doc_evidence:
                return FUNC_CALL_API
            return FUNC_CALL_DOCUMENT

        raise RuntimeError(
            "Selector model is unavailable and routing indexes were not loaded. "
            "Run scripts/train_selector.py or build the document/API indexes first."
        )

    def predict_batch(self, samples: list[InputSample]) -> list[PredictionResult]:
        """Predict for a batch of samples."""
        results = []
        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i+1}/{len(samples)}...")
            results.append(self.predict_one(sample))
        return results
