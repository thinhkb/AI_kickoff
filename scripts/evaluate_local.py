"""
Run local evaluation before submission.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import EXAMPLE_DATA_FILE, ensure_dirs, print_current_config
from src.schemas import InputSample
from src.pipeline import Pipeline
from src.utils.io_utils import read_excel_sheet
from src.utils.logging_utils import logger
from configs.constants import FUNC_CALL_DOCUMENT, FUNC_CALL_API


def main():
    ensure_dirs()
    print_current_config()

    # Load example data
    questions_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_question")
    results_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_result")

    # Build ground truth
    gt = {}
    for row in results_data:
        gt[int(row["id"])] = {
            "func_code": row["func_code"],
            "func_param": row["func_param"],
        }

    # Build samples
    samples = []
    for row in questions_data:
        samples.append(InputSample.from_dict(row))

    logger.info(f"Evaluating on {len(samples)} samples...")

    # Load pipeline
    pipeline = Pipeline()
    pipeline.load()

    # Predict
    preds = pipeline.predict_batch(samples)

    # Evaluate routing accuracy
    correct_route = 0
    total = 0
    for pred in preds:
        if pred.id in gt:
            total += 1
            if pred.function_code == gt[pred.id]["func_code"]:
                correct_route += 1
            else:
                logger.warning(
                    f"  Route error: id={pred.id}, "
                    f"pred={pred.function_code}, "
                    f"gt={gt[pred.id]['func_code']}"
                )

    if total > 0:
        logger.info(f"\nRouting accuracy: {correct_route}/{total} = {correct_route/total:.4f}")

    # Evaluate API path accuracy
    api_correct = 0
    api_total = 0
    for pred in preds:
        if pred.id in gt and gt[pred.id]["func_code"] == FUNC_CALL_API:
            api_total += 1
            try:
                pred_json = json.loads(pred.function_answer)
                gt_json = json.loads(gt[pred.id]["func_param"])
                if pred_json.get("path") == gt_json.get("path"):
                    api_correct += 1
            except (json.JSONDecodeError, TypeError):
                pass

    if api_total > 0:
        logger.info(f"API path accuracy: {api_correct}/{api_total} = {api_correct/api_total:.4f}")

    # Average time
    avg_time = sum(p.time_response for p in preds) / max(len(preds), 1)
    logger.info(f"Average time_response: {avg_time:.3f}s")


if __name__ == "__main__":
    main()
