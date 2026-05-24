"""Sentence-transformer based frozen semantic encoder."""

from __future__ import annotations

import numpy as np

from ..preprocessing.normalizer import normalize_for_semantic_model


class SentenceTransformerEncoder:
    """Frozen sentence embedding encoder.

    The dependency is imported lazily so non-semantic experiments keep working
    without installing transformer packages.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Running v8 requires sentence-transformers. Install project "
                "dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        normalized = [normalize_for_semantic_model(text) for text in texts]
        vectors = self.model.encode(
            normalized,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

