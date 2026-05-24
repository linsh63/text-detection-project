"""Base interfaces for semantic encoders."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SemanticEncoder(Protocol):
    """Protocol implemented by frozen semantic encoders."""

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        """Return one dense vector per input text."""
        ...
