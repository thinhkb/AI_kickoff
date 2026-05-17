"""
Writes batch predictions to JSONL / CSV submission file.
"""
from pathlib import Path
from typing import List
from src.schemas import PredictionResult
from src.output.formatter import format_batch
from src.utils.io_utils import write_jsonl, write_csv
from src.utils.logging_utils import logger
from configs.paths import PREDICTIONS_FILE, SUBMISSION_FILE


def write_predictions(preds: List[PredictionResult], path: str | Path = None):
    """Write predictions to JSONL file."""
    path = Path(path or PREDICTIONS_FILE)
    formatted = format_batch(preds)
    write_jsonl(formatted, path)
    logger.info(f"Wrote {len(formatted)} predictions to {path}")


def write_submission(preds: List[PredictionResult], path: str | Path = None):
    """Write predictions to CSV submission file."""
    path = Path(path or SUBMISSION_FILE)
    formatted = format_batch(preds)
    write_csv(
        formatted, path,
        fieldnames=["id", "function_code", "function_answer", "time_response"],
    )
    logger.info(f"Wrote submission to {path}")
