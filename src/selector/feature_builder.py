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


class FeatureBuilder:
    """
    Builds TF-IDF features for the branch selector.
    Combines word n-grams and optional character n-grams.
    """

    def __init__(self):
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
        self._fitted = False

    def fit(self, texts: list[str]):
        """Fit vectorizers on training texts."""
        self.word_vectorizer.fit(texts)
        if self.char_vectorizer:
            self.char_vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str]):
        """Transform texts into feature matrix."""
        from scipy.sparse import hstack

        word_features = self.word_vectorizer.transform(texts)
        if self.char_vectorizer:
            char_features = self.char_vectorizer.transform(texts)
            return hstack([word_features, char_features])
        return word_features

    def fit_transform(self, texts: list[str]):
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)
