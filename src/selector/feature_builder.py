"""
Builds selector features from the question and retrieval signals.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional
import numpy as np

from configs.model_config import (
    SELECTOR_NGRAM_RANGE,
    SELECTOR_MAX_FEATURES,
    SELECTOR_USE_CHAR_NGRAMS,
)
from src.selector.routing_features import ROUTING_FEATURE_NAMES


class FeatureBuilder:
    """
    Builds TF-IDF features for the branch selector.
    Combines word n-grams and optional character n-grams.
    """

    def __init__(
        self,
        routing_feature_extractor=None,
        use_routing_features: bool = False,
    ):
        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=SELECTOR_NGRAM_RANGE,
            max_features=SELECTOR_MAX_FEATURES,
            sublinear_tf=True,
            strip_accents=None,  # Keep Vietnamese accents
        )
        self.char_vectorizer = None
        if SELECTOR_USE_CHAR_NGRAMS:
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                max_features=SELECTOR_MAX_FEATURES // 2,
                sublinear_tf=True,
            )
        self.routing_feature_extractor = routing_feature_extractor
        self.use_routing_features = use_routing_features
        self.routing_feature_names = list(ROUTING_FEATURE_NAMES)
        self._fitted = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Runtime retrievers can be large and are restored by Pipeline.load().
        state["routing_feature_extractor"] = None
        return state

    def set_routing_feature_extractor(self, extractor):
        """Attach runtime resources used to compute question-only route features."""
        self.routing_feature_extractor = extractor

    def fit(self, texts: list[str]):
        """Fit vectorizers on training texts."""
        self.word_vectorizer.fit(texts)
        if self.char_vectorizer:
            self.char_vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str]):
        """Transform texts into feature matrix."""
        from scipy.sparse import csr_matrix, hstack

        word_features = self.word_vectorizer.transform(texts)
        feature_blocks = [word_features]
        if self.char_vectorizer:
            char_features = self.char_vectorizer.transform(texts)
            feature_blocks.append(char_features)

        if getattr(self, "use_routing_features", False):
            route_features = self._transform_routing_features(texts)
            feature_blocks.append(csr_matrix(route_features))

        return hstack(feature_blocks)

    def fit_transform(self, texts: list[str]):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    def _transform_routing_features(self, texts: list[str]) -> np.ndarray:
        extractor = getattr(self, "routing_feature_extractor", None)
        if extractor is None:
            n_features = len(getattr(
                self,
                "routing_feature_names",
                ROUTING_FEATURE_NAMES,
            ))
            return np.zeros((len(texts), n_features), dtype=float)
        return extractor.transform(texts)
