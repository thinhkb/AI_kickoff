"""
Batch prediction entry point for test-time inference and submission generation.
Usage: python run_submission.py
       python run_submission.py --version 5   (force specific version)
"""
import sys
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import (
    TEST_DATA_FILE, ensure_dirs, get_versioned_paths, print_current_config,
)
from src.schemas import InputSample
from src.pipeline import Pipeline
from src.output.writer import write_predictions, write_submission
from src.utils.io_utils import read_excel_sheet
from src.utils.logging_utils import logger
from configs import model_config as mc
from configs import constants as c


def _runtime_config_lines() -> list[str]:
    """Build important runtime config lines for the final submission summary."""
    use_reranker = os.getenv("VIETTEL_USE_RERANKER", "0")
    reranker_model = os.getenv("VIETTEL_RERANKER_MODEL") or mc.RERANKER_MODEL_NAME
    reranker_device = os.getenv("VIETTEL_RERANKER_DEVICE") or "auto"

    return [
        "  RUNTIME CONFIG",
        "  --------------",
        f"  embedding_model     : {mc.EMBEDDING_MODEL_NAME}",
        f"  embedding_batch     : {mc.EMBEDDING_BATCH_SIZE}",
        f"  embedding_normalize : {mc.EMBEDDING_NORMALIZE}",
        f"  use_reranker        : {use_reranker}",
        f"  reranker_model      : {reranker_model}",
        f"  reranker_device     : {reranker_device}",
        f"  reranker_batch      : {mc.RERANKER_BATCH_SIZE}",
        f"  llm_model           : {mc.LLM_MODEL_NAME}",
        f"  llm_api_base        : {mc.LLM_API_BASE}",
        f"  ocr_model           : {mc.OCR_MODEL}",
        f"  layout_model        : {mc.LAYOUT_MODEL}",
        f"  ocr_lang            : {mc.OCR_LANG}",
        f"  doc_top_k           : {c.DOC_RETRIEVAL_TOP_K}",
        f"  api_top_k           : {c.API_RETRIEVAL_TOP_K}",
        f"  chunk_size          : {c.CHUNK_SIZE}",
        f"  chunk_overlap       : {c.CHUNK_OVERLAP}",
        f"  selector_type       : {mc.SELECTOR_TYPE}",
    ]


def main():
    # ─── Parse args ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Run submission inference")
    parser.add_argument(
        "--version", "-v", type=int, default=None,
        help="Force output version number (e.g. --version 3). "
             "Default: auto-increment.",
    )
    args = parser.parse_args()

    ensure_dirs()

    # ─── Print configuration ──────────────────────────────────────
    print_current_config()

    # ─── Determine output paths ───────────────────────────────────
    pred_path, sub_path = get_versioned_paths(args.version)
    version_num = int(pred_path.stem.split("_v")[-1])
    logger.info(f"Output version: v{version_num}")
    logger.info(f"  Predictions → {pred_path}")
    logger.info(f"  Submission  → {sub_path}")

    # ─── Load test data ───────────────────────────────────────────
    logger.info("Loading test data...")
    test_data = read_excel_sheet(TEST_DATA_FILE, "question_test")
    samples = [InputSample.from_dict(row) for row in test_data]
    logger.info(f"Loaded {len(samples)} test questions")

    # ─── Initialize and load pipeline ─────────────────────────────
    pipeline = Pipeline()
    pipeline.load()

    # ─── Run inference ────────────────────────────────────────────
    logger.info("Running inference...")
    predictions = pipeline.predict_batch(samples)

    # ─── Write outputs ────────────────────────────────────────────
    write_predictions(predictions, path=pred_path)
    write_submission(predictions, path=sub_path)

    # ─── Summary ──────────────────────────────────────────────────
    api_count = sum(1 for p in predictions if p.function_code == "call_api")
    doc_count = sum(1 for p in predictions if p.function_code == "call_document")
    avg_time = sum(p.time_response for p in predictions) / max(len(predictions), 1)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"  SUBMISSION SUMMARY (v{version_num})")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Total predictions : {len(predictions)}")
    logger.info(f"  call_api          : {api_count}")
    logger.info(f"  call_document     : {doc_count}")
    logger.info(f"  avg time_response : {avg_time:.3f}s")
    logger.info(f"  predictions file  : {pred_path.name}")
    logger.info(f"  submission file   : {sub_path.name}")
    logger.info("")
    for line in _runtime_config_lines():
        logger.info(line)
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()

