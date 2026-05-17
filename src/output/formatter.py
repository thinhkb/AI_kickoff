"""
Formats final output into challenge-compliant schema.
"""
from src.schemas import PredictionResult


def format_prediction(pred: PredictionResult) -> dict:
    """Format a single prediction for submission."""
    return {
        "id": pred.id,
        "function_code": pred.function_code,
        "function_answer": pred.function_answer,
        "time_response": round(pred.time_response, 4),
    }


def format_batch(preds: list[PredictionResult]) -> list[dict]:
    """Format a batch of predictions."""
    return [format_prediction(p) for p in preds]

