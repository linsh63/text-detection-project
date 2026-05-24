"""Frozen-encoder semantic classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..encoders.base import SemanticEncoder


class FrozenEncoderClassifier:
    """Classifier trained on frozen semantic embeddings."""

    def __init__(
        self,
        encoder: SemanticEncoder,
        classifier: LogisticRegression | None = None,
        batch_size: int = 64,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier or LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.batch_size = batch_size

    def fit(self, texts, labels) -> "FrozenEncoderClassifier":
        vectors = self.encoder.encode(texts, batch_size=self.batch_size)
        self.classifier.fit(vectors, labels)
        return self

    def score_texts(self, texts) -> np.ndarray:
        vectors = self.encoder.encode(texts, batch_size=self.batch_size)
        if hasattr(self.classifier, "predict_proba"):
            return np.asarray(self.classifier.predict_proba(vectors)[:, 1], dtype=float)
        if hasattr(self.classifier, "decision_function"):
            return np.asarray(self.classifier.decision_function(vectors), dtype=float)
        raise ValueError("Classifier must expose predict_proba or decision_function.")

