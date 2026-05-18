"""
Project-wide evaluation metrics and answer normalization.
"""
import json
import re
from typing import List, Dict, Any, Optional


def safe_json_loads(raw: Any) -> Optional[Any]:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _repair_api_json(text)


def _repair_api_json(text: str) -> Optional[Dict[str, Any]]:
    """Repair the malformed sample rows where body is an unescaped JSON string."""
    path_match = re.search(r'"path"\s*:\s*"([^"]+)"', text)
    if not path_match:
        return None

    body: Dict[str, Any] = {}
    for key, value in re.findall(r'"(fromDate|toDate)"\s*:\s*"([^"]+)"', text):
        body[key] = value

    for key in [
        "projectStatus", "projectList", "projectType",
        "organization", "customerList",
    ]:
        if re.search(rf'"{key}"\s*:\s*\[\s*\]', text):
            body[key] = []

    return {"path": path_match.group(1), "body": body}


def normalize_answer(raw: Any) -> Any:
    parsed = safe_json_loads(raw)
    if isinstance(parsed, dict) and isinstance(parsed.get("body"), str):
        body_text = re.sub(r",\s*}", "}", parsed["body"].strip())
        try:
            parsed = dict(parsed)
            parsed["body"] = json.loads(body_text)
        except json.JSONDecodeError:
            pass
    return canonicalize(parsed)


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    return value


def normalize_doc_result(raw: Any) -> str:
    parsed = safe_json_loads(raw) or {}
    result = str(parsed.get("result", "")).replace(" ", "")
    return ",".join(sorted(x for x in result.split(",") if x))


def routing_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth:
            total += 1
            if pred["function_code"] == ground_truth[pid]["func_code"]:
                correct += 1
    return correct / max(total, 1)


def exact_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid not in ground_truth:
            continue
        total += 1
        gt = ground_truth[pid]
        if pred["function_code"] != gt["func_code"]:
            continue
        if gt["func_code"] == "call_document":
            if normalize_doc_result(pred["function_answer"]) == normalize_doc_result(gt["func_param"]):
                correct += 1
        elif normalize_answer(pred["function_answer"]) == normalize_answer(gt["func_param"]):
            correct += 1
    return correct / max(total, 1)


def api_path_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth and ground_truth[pid]["func_code"] == "call_api":
            total += 1
            pred_j = safe_json_loads(pred["function_answer"])
            gt_j = safe_json_loads(ground_truth[pid]["func_param"])
            if isinstance(pred_j, dict) and isinstance(gt_j, dict):
                if pred_j.get("path") == gt_j.get("path"):
                    correct += 1
    return correct / max(total, 1)


def api_body_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth and ground_truth[pid]["func_code"] == "call_api":
            total += 1
            if normalize_answer(pred["function_answer"]) == normalize_answer(ground_truth[pid]["func_param"]):
                correct += 1
    return correct / max(total, 1)


def doc_answer_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth and ground_truth[pid]["func_code"] == "call_document":
            total += 1
            if normalize_doc_result(pred["function_answer"]) == normalize_doc_result(
                ground_truth[pid]["func_param"]
            ):
                correct += 1
    return correct / max(total, 1)


def average_time(predictions: List[Dict]) -> float:
    times = [p.get("time_response", 0) for p in predictions]
    return sum(times) / max(len(times), 1)
