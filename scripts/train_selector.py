"""
Train or calibrate the branch selector using example data.
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import (
    API_CONFIG_FILE,
    DOCUMENT_CHUNKS_FILE,
    EXAMPLE_DATA_FILE,
    SELECTOR_MODEL_DIR,
    ensure_dirs,
    print_current_config,
)
from src.api.api_catalog_loader import load_alias_dictionary, load_api_catalog
from src.api.api_retriever import APIRetriever
from src.api.slot_extractor import SlotExtractor
from src.document.retriever import DocumentRetriever
from src.selector.selector_model import SelectorModel
from src.selector.routing_features import RoutingFeatureExtractor
from src.preprocess.question_normalizer import normalize_question
from src.utils.io_utils import read_excel_sheet
from src.utils.logging_utils import logger
from configs.constants import FUNC_CALL_DOCUMENT, FUNC_CALL_API
from sklearn.model_selection import cross_val_score


def build_routing_feature_extractor():
    """Build question-only retrieval/schema features for selector training."""
    enabled = os.getenv("VIETTEL_SELECTOR_ROUTE_FEATURES", "1").lower()
    if enabled in {"0", "false", "no"}:
        logger.info("Selector routing features disabled by environment")
        return None

    document_retriever = None
    api_retriever = None
    slot_extractor = None

    try:
        if DOCUMENT_CHUNKS_FILE.exists():
            document_retriever = DocumentRetriever()
            document_retriever.load_index(DOCUMENT_CHUNKS_FILE)
        else:
            logger.warning(f"Document chunks not found: {DOCUMENT_CHUNKS_FILE}")
    except Exception as exc:
        logger.warning(f"Could not load document routing index: {exc}")
        document_retriever = None

    try:
        apis = load_api_catalog(API_CONFIG_FILE)
        alias_dict = load_alias_dictionary(API_CONFIG_FILE)
        api_retriever = APIRetriever()
        api_retriever.build_index(apis)
        slot_extractor = SlotExtractor(alias_dict=alias_dict)
    except Exception as exc:
        logger.warning(f"Could not load API routing index: {exc}")
        api_retriever = None
        slot_extractor = None

    if document_retriever is None and api_retriever is None:
        logger.warning("Selector will train with text features only")
        return None

    logger.info("Selector will train with question-only routing features")
    return RoutingFeatureExtractor(
        document_retriever=document_retriever,
        api_retriever=api_retriever,
        slot_extractor=slot_extractor,
    )


def main():
    ensure_dirs()
    print_current_config()
    # Load example data
    questions_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_question")
    results_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_result")

    # Build question → label mapping
    id_to_question = {}
    for row in questions_data:
        qid = int(row["id"])
        id_to_question[qid] = row["fun_question"]

    questions = []
    labels = []
    for row in results_data:
        qid = int(row["id"])
        func_code = row["func_code"]
        if qid in id_to_question:
            q = normalize_question(id_to_question[qid])
            questions.append(q)
            labels.append(func_code)

    logger.info(f"Training data: {len(questions)} samples")
    logger.info(f"  call_api: {labels.count(FUNC_CALL_API)}")
    logger.info(f"  call_document: {labels.count(FUNC_CALL_DOCUMENT)}")

    # Train selector
    routing_feature_extractor = build_routing_feature_extractor()
    model = SelectorModel(
        routing_feature_extractor=routing_feature_extractor,
        use_routing_features=routing_feature_extractor is not None,
    )

    # Cross-validation
    X = model.feature_builder.fit_transform(questions)
    import numpy as np
    y = np.array(labels)

    cv_scores = cross_val_score(model.classifier, X, y, cv=5, scoring="accuracy")
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train on full data
    metrics = model.train(questions, labels)
    logger.info(f"Training metrics: {metrics}")

    # Save
    model.save(SELECTOR_MODEL_DIR)
    logger.info(f"Selector model saved to {SELECTOR_MODEL_DIR}")


if __name__ == "__main__":
    main()
