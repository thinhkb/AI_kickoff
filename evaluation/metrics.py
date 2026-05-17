"""
Project-wide evaluation metrics.
"""
import json
from typing import List, Dict, Any
from src.utils.logging_utils import logger


def routing_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    """Calculate routing (function_code) accuracy."""
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth:
            total += 1
            if pred["function_code"] == ground_truth[pid]["func_code"]:
                correct += 1
    return correct / max(total, 1)


def api_path_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    """Calculate API path selection accuracy."""
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth and ground_truth[pid]["func_code"] == "call_api":
            total += 1
            try:
                pred_j = json.loads(pred["function_answer"])
                gt_j = json.loads(ground_truth[pid]["func_param"])
                if pred_j.get("path") == gt_j.get("path"):
                    correct += 1
            except (json.JSONDecodeError, TypeError):
                pass
    return correct / max(total, 1)


def api_body_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    """Calculate API body parameter accuracy (exact match)."""
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth and ground_truth[pid]["func_code"] == "call_api":
            total += 1
            try:
                pred_j = json.loads(pred["function_answer"])
                gt_j = json.loads(ground_truth[pid]["func_param"])
                if pred_j == gt_j:
                    correct += 1
            except (json.JSONDecodeError, TypeError):
                pass
    return correct / max(total, 1)


def doc_answer_accuracy(predictions: List[Dict], ground_truth: Dict[int, Dict]) -> float:
    """Calculate document answer accuracy."""
    correct = 0
    total = 0
    for pred in predictions:
        pid = pred["id"]
        if pid in ground_truth and ground_truth[pid]["func_code"] == "call_document":
            total += 1
            try:
                pred_j = json.loads(pred["function_answer"])
                gt_j = json.loads(ground_truth[pid]["func_param"])
                if pred_j.get("result") == gt_j.get("result"):
                    correct += 1
            except (json.JSONDecodeError, TypeError):
                pass
    return correct / max(total, 1)


def average_time(predictions: List[Dict]) -> float:
    """Calculate average response time."""
    times = [p.get("time_response", 0) for p in predictions]
    return sum(times) / max(len(times), 1)
