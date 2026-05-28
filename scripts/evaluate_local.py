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
    safe_json_loads,
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

    # Detailed section-by-section metrics calculation
    selector_total = len(pred_dicts)
    selector_correct = sum(1 for p in pred_dicts if p["function_code"] == ground_truth[p["id"]]["func_code"])
    routing_acc = selector_correct / max(selector_total, 1)

    doc_total = sum(1 for p in pred_dicts if ground_truth[p["id"]]["func_code"] == "call_document")
    doc_correct = sum(
        1 for p in pred_dicts 
        if ground_truth[p["id"]]["func_code"] == "call_document" 
        and normalize_doc_result(p["function_answer"]) == normalize_doc_result(ground_truth[p["id"]]["func_param"])
    )
    doc_acc = doc_correct / max(doc_total, 1)

    api_total = sum(1 for p in pred_dicts if ground_truth[p["id"]]["func_code"] == "call_api")
    api_path_correct = sum(
        1 for p in pred_dicts 
        if ground_truth[p["id"]]["func_code"] == "call_api" 
        and safe_json_loads(p["function_answer"]) 
        and safe_json_loads(ground_truth[p["id"]]["func_param"]) 
        and safe_json_loads(p["function_answer"]).get("path") == safe_json_loads(ground_truth[p["id"]]["func_param"]).get("path")
    )
    api_body_correct = sum(
        1 for p in pred_dicts 
        if ground_truth[p["id"]]["func_code"] == "call_api" 
        and normalize_answer(p["function_answer"]) == normalize_answer(ground_truth[p["id"]]["func_param"])
    )
    api_path_acc = api_path_correct / max(api_total, 1)
    api_body_acc = api_body_correct / max(api_total, 1)

    end_to_end_correct = sum(
        1 for p in pred_dicts 
        if (ground_truth[p["id"]]["func_code"] == "call_document" and normalize_doc_result(p["function_answer"]) == normalize_doc_result(ground_truth[p["id"]]["func_param"]))
        or (ground_truth[p["id"]]["func_code"] == "call_api" and normalize_answer(p["function_answer"]) == normalize_answer(ground_truth[p["id"]]["func_param"]))
    )
    e2e_acc = end_to_end_correct / max(selector_total, 1)

    avg_time = average_time(pred_dicts)

    metrics = {
        "total": selector_total,
        "routing_accuracy": routing_acc,
        "document_answer_accuracy": doc_acc,
        "api_path_accuracy": api_path_acc,
        "api_body_exact_accuracy": api_body_acc,
        "end_to_end_exact_accuracy": e2e_acc,
        "average_time_response": avg_time,
    }

    mismatches = build_mismatches(pred_dicts, ground_truth, question_by_id)
    metrics["mismatch_count"] = len(mismatches)

    print("\n==================================================")
    print("        KẾT QUẢ ĐÁNH GIÁ CHI TIẾT THEO TỪNG PHẦN")
    print("==================================================")
    print("\n1. BỘ PHÂN LOẠI NHÁNH (SELECTOR)")
    print(f"   - Tổng số câu hỏi   : {selector_total}")
    print(f"   - Phân loại đúng     : {selector_correct} / {selector_total}")
    print(f"   - Độ chính xác (Acc) : {routing_acc * 100:.2f}%")

    print("\n2. NHÁNH TÀI LIỆU (CALL_DOCUMENT)")
    print(f"   - Tổng số câu hỏi tài liệu: {doc_total}")
    print(f"   - Điền đáp án đúng        : {doc_correct} / {doc_total}")
    print(f"   - Độ chính xác (Acc)      : {doc_acc * 100:.2f}%")

    print("\n3. NHÁNH API (CALL_API)")
    print(f"   - Tổng số câu hỏi API     : {api_total}")
    print(f"   - Đúng đường dẫn (Path)   : {api_path_correct} / {api_total} ({api_path_acc * 100:.2f}%)")
    print(f"   - Đúng tham số (Params)   : {api_body_correct} / {api_total} ({api_body_acc * 100:.2f}%)")

    print("\n4. KẾT QUẢ TỔNG HỢP (AGGREGATED)")
    print(f"   - Tổng số câu hỏi (Total) : {selector_total}")
    print(f"   - Số câu đúng hoàn hảo    : {end_to_end_correct} / {selector_total}")
    print(f"   - Độ chính xác toàn diện  : {e2e_acc * 100:.2f}%")
    print(f"   - Thời gian phản hồi TB   : {avg_time:.4f}s")
    print("==================================================")

    if mismatches:
        print("\nCác trường hợp dự đoán lệch (Mismatches):")
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
