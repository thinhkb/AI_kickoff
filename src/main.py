"""
Main application entry point.
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import InputSample
from src.pipeline import Pipeline
from src.utils.logging_utils import logger


def predict_single(question: str, note: str = None) -> dict:
    """Quick single-question prediction."""
    pipeline = Pipeline()
    pipeline.load()

    sample = InputSample(id=0, question=question, note=note)
    result = pipeline.predict_one(sample)
    return result.to_dict()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = predict_single(question)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Usage: python src/main.py '<question>'")
