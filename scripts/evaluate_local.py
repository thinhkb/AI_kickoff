"""
Run local evaluation on example_data.

This evaluates the same surface the submission is judged on:
- function routing
- call_document answer letter
- call_api path
- call_api full JSON body
- end-to-end exact correctness
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import EXAMPLE_DATA_FILE, OUTPUTS_DIR, ensure_dirs, print_current_config
from evaluation.metrics import (
    api_body_accuracy,
    api_path_accuracy,
    average_time,
    doc_answer_accuracy,
    exact_accuracy,
    normalize_answer,
    normalize_doc_result,
    routing_accuracy,
)
from src.pipeline import Pipeline
from src.schemas import InputSample
from src.utils.io_utils import read_excel_sheet, write_json, write_jsonl
from src.utils.logging_utils import logger


def build_ground_truth(rows):
    return {
        int(row["id"]): {
            "func_code": row["func_code"],
            "func_param": row["func_param"],
        }
        for row in rows
    }


def build_mismatches(predictions, ground_truth, question_by_id):
    mismatches = []
    for pred in predictions:
        pid = pred["id"]
        gt = ground_truth.get(pid)
        if not gt:
            continue

        issue = None
        pred_norm = None
        gt_norm = None

        if pred["function_code"] != gt["func_code"]:
            issue = "route"
            pred_norm = pred["function_code"]
            gt_norm = gt["func_code"]
        elif gt["func_code"] == "call_document":
            pred_norm = normalize_doc_result(pred["function_answer"])
            gt_norm = normalize_doc_result(gt["func_param"])
            if pred_norm != gt_norm:
                issue = "document_answer"
        else:
            pred_norm = normalize_answer(pred["function_answer"])
            gt_norm = normalize_answer(gt["func_param"])
            if pred_norm != gt_norm:
                issue = "api_answer"

        if issue:
            mismatches.append({
                "id": pid,
                "issue": issue,
                "question": question_by_id.get(pid, ""),
                "pred_function_code": pred["function_code"],
                "gt_function_code": gt["func_code"],
                "pred_answer": pred_norm,
                "gt_answer": gt_norm,
                "time_response": pred.get("time_response", 0),
            })

    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline on example_data.xlsx")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample pipeline logs.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write eval_report.json and eval_mismatches.jsonl to outputs/.",
    )
    args = parser.parse_args()

    ensure_dirs()
    if not args.quiet:
        print_current_config()
    else:
        logger.setLevel(logging.WARNING)
        for handler in logger.handlers:
            handler.setLevel(logging.WARNING)

    questions_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_question")
    results_data = read_excel_sheet(EXAMPLE_DATA_FILE, "example_result")
    ground_truth = build_ground_truth(results_data)
    samples = [InputSample.from_dict(row) for row in questions_data]
    question_by_id = {sample.id: sample.question for sample in samples}

    logger.info(f"Evaluating on {len(samples)} samples...")
    pipeline = Pipeline()
    pipeline.load()
    preds = pipeline.predict_batch(samples)
    pred_dicts = [p.to_dict() for p in preds]

    metrics = {
        "total": len(pred_dicts),
        "routing_accuracy": routing_accuracy(pred_dicts, ground_truth),
        "document_answer_accuracy": doc_answer_accuracy(pred_dicts, ground_truth),
        "api_path_accuracy": api_path_accuracy(pred_dicts, ground_truth),
        "api_body_exact_accuracy": api_body_accuracy(pred_dicts, ground_truth),
        "end_to_end_exact_accuracy": exact_accuracy(pred_dicts, ground_truth),
        "average_time_response": average_time(pred_dicts),
    }

    mismatches = build_mismatches(pred_dicts, ground_truth, question_by_id)
    metrics["mismatch_count"] = len(mismatches)

    print("\n=== Local Evaluation ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if mismatches:
        print("\nTop mismatches:")
        for item in mismatches[:10]:
            print(f"- id={item['id']} issue={item['issue']} q={item['question'][:100]}")

    if args.write_report:
        report_path = OUTPUTS_DIR / "eval_report.json"
        mismatch_path = OUTPUTS_DIR / "eval_mismatches.jsonl"
        write_json(metrics, report_path)
        write_jsonl(mismatches, mismatch_path)
        print(f"\nWrote {report_path}")
        print(f"Wrote {mismatch_path}")


if __name__ == "__main__":
    main()
