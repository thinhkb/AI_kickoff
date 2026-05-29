"""
Implements or loads the classifier that predicts call_document / call_api.
"""
import joblib
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.selector.feature_builder import FeatureBuilder
from configs.constants import FUNC_CALL_DOCUMENT, FUNC_CALL_API
from configs.paths import SELECTOR_MODEL_DIR
from src.utils.logging_utils import logger


class SelectorModel:
    """
    TF-IDF + LogisticRegression classifier for branch selection.
    Predicts call_document or call_api based on the question only.
    """

    def __init__(
        self,
        routing_feature_extractor=None,
        use_routing_features: bool = False,
    ):
        self.feature_builder = FeatureBuilder(
            routing_feature_extractor=routing_feature_extractor,
            use_routing_features=use_routing_features,
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        self._fitted = False

    def set_routing_feature_extractor(self, extractor):
        """Attach question-only routing feature resources after loading indexes."""
        self.feature_builder.set_routing_feature_extractor(extractor)

    def train(self, questions: list[str], labels: list[str]) -> dict:
        """
        Train the selector on labeled questions.
        labels should be FUNC_CALL_DOCUMENT or FUNC_CALL_API.
        Returns training metrics.
        """
        logger.info(f"Training selector on {len(questions)} samples...")

        # Build features
        X = self.feature_builder.fit_transform(questions)
        y = np.array(labels)

        # Train classifier
        self.classifier.fit(X, y)
        self._fitted = True

        # Training accuracy
        preds = self.classifier.predict(X)
        accuracy = float((preds == y).mean())
        logger.info(f"Selector training accuracy: {accuracy:.4f}")

        return {"accuracy": accuracy, "n_samples": len(questions)}

    def predict(self, question: str) -> str:
        """Predict function code for a single question."""
        X = self.feature_builder.transform([question])
        return self.classifier.predict(X)[0]

    def predict_proba(self, question: str) -> Tuple[str, float]:
        """Predict with confidence score."""
        X = self.feature_builder.transform([question])
        pred = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_
        idx = list(classes).index(pred)
        return pred, float(proba[idx])

    def predict_batch(self, questions: list[str]) -> list[str]:
        """Predict function codes for a batch of questions."""
        X = self.feature_builder.transform(questions)
        return self.classifier.predict(X).tolist()

    def save(self, model_dir: str | Path = None):
        """Save the model to disk."""
        model_dir = Path(model_dir or SELECTOR_MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.feature_builder, model_dir / "feature_builder.joblib")
        joblib.dump(self.classifier, model_dir / "classifier.joblib")
        logger.info(f"Selector model saved to {model_dir}")

    def load(self, model_dir: str | Path = None):
        """Load the model from disk."""
        model_dir = Path(model_dir or SELECTOR_MODEL_DIR)

        self.feature_builder = joblib.load(model_dir / "feature_builder.joblib")
        self.classifier = joblib.load(model_dir / "classifier.joblib")
        self._fitted = True
        logger.info(f"Selector model loaded from {model_dir}")
