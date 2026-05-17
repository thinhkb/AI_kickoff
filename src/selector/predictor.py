"""
High-level function for selector inference.
"""
from pathlib import Path
from typing import Optional

from src.selector.selector_model import SelectorModel
from src.preprocess.question_normalizer import normalize_question
from src.utils.logging_utils import logger


class SelectorPredictor:
    """
    High-level predictor that loads the selector model and
    normalizes questions before prediction.
    """

    def __init__(self, model_dir: Optional[str | Path] = None):
        self.model = SelectorModel()
        self._loaded = False
        if model_dir:
            self.load(model_dir)

    def load(self, model_dir: str | Path = None):
        """Load the pretrained selector model."""
        self.model.load(model_dir)
        self._loaded = True

    def predict(self, question: str) -> str:
        """
        Predict the function code for a question.
        Normalizes the question before prediction.
        """
        if not self._loaded:
            raise RuntimeError("Selector model not loaded. Call load() first.")

        q_norm = normalize_question(question)
        return self.model.predict(q_norm)

    def predict_with_confidence(self, question: str) -> tuple[str, float]:
        """Predict with confidence score."""
        if not self._loaded:
            raise RuntimeError("Selector model not loaded. Call load() first.")

        q_norm = normalize_question(question)
        return self.model.predict_proba(q_norm)

    def predict_batch(self, questions: list[str]) -> list[str]:
        """Predict function codes for a batch of questions."""
        if not self._loaded:
            raise RuntimeError("Selector model not loaded. Call load() first.")

        normalized = [normalize_question(q) for q in questions]
        return self.model.predict_batch(normalized)
