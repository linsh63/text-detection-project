"""Adversarial-data helpers without manually curated keyword lists."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


ADVERSARIAL_COLUMNS = ["label", "text", "original_text", "variant"]


def _empty_adversarial_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ADVERSARIAL_COLUMNS)


def replace_with_variant(text: str, rng: random.Random) -> tuple[str, str]:
    """Compatibility hook that no longer replaces known keywords with variants."""
    _ = rng
    return str(text), ""


def generate_adversarial_dataset(
    input_path: str | Path,
    output_path: str | Path,
    max_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create an empty adversarial set because hard keyword variants are removed."""
    _ = input_path, max_samples, random_state
    result = _empty_adversarial_frame()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def generate_keyword_challenge_dataset(output_path: str | Path) -> pd.DataFrame:
    """Create an empty keyword challenge set without handcrafted sensitive terms."""
    result = _empty_adversarial_frame()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def keyword_training_samples() -> pd.DataFrame:
    """Return no manual keyword samples for augmentation."""
    return pd.DataFrame(columns=["label", "text"])
