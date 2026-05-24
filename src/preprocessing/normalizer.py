"""Generic text normalization for semantic models.

This module intentionally avoids spam-keyword replacement, hand-written
variant maps, or risk dictionaries. It only handles source-agnostic text noise.
"""

from __future__ import annotations

import re
import unicodedata


CONTROL_PATTERN = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
SPACE_PATTERN = re.compile(r"\s+")


def normalize_for_semantic_model(text: str) -> str:
    """Normalize generic unicode and whitespace noise without keyword rules."""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = CONTROL_PATTERN.sub(" ", normalized)
    normalized = SPACE_PATTERN.sub(" ", normalized)
    return normalized.strip().lower()

