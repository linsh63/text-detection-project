"""Generate adversarial Chinese spam variants for demonstration and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .char_similarity import KEYWORD_VARIANTS


def replace_with_variant(text: str, rng: random.Random) -> tuple[str, str]:
    """Replace the first matched keyword with one random variant."""
    candidates = [keyword for keyword in KEYWORD_VARIANTS if keyword in text]
    if not candidates:
        return text, ""

    keyword = rng.choice(candidates)
    variant = rng.choice(KEYWORD_VARIANTS[keyword])
    return text.replace(keyword, variant, 1), f"{keyword}->{variant}"


def generate_adversarial_dataset(
    input_path: str | Path,
    output_path: str | Path,
    max_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a spam-only adversarial evaluation set from labeled data."""
    data = pd.read_csv(input_path, sep="\t")
    spam = data[data["label"] == 1].copy()
    rng = random.Random(random_state)

    rows: list[dict[str, str | int]] = []
    for _, row in spam.sample(frac=1.0, random_state=random_state).iterrows():
        original = str(row["text"])
        text, variant = replace_with_variant(original, rng)
        if not variant:
            continue
        rows.append(
            {
                "label": 1,
                "text": text,
                "original_text": original,
                "variant": variant,
            }
        )
        if len(rows) >= max_samples:
            break

    result = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def generate_keyword_challenge_dataset(output_path: str | Path) -> pd.DataFrame:
    """Create a compact adversarial keyword challenge set.

    The samples are intentionally short: they isolate whether a model recognizes
    the sensitive keyword variant itself rather than relying on long spam context.
    """
    templates = (
        "{variant}",
        "加{variant}",
        "{variant}联系",
        "请加{variant}",
        "有事{variant}聊",
        "办理{variant}",
    )

    rows: list[dict[str, str | int]] = []
    for keyword, variants in KEYWORD_VARIANTS.items():
        for variant in variants:
            for template in templates:
                rows.append(
                    {
                        "label": 1,
                        "text": template.format(variant=variant),
                        "original_text": template.format(variant=keyword),
                        "variant": f"{keyword}->{variant}",
                    }
                )

    result = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def keyword_training_samples() -> pd.DataFrame:
    """Return canonical sensitive-keyword samples for CSN-aware augmentation."""
    templates = (
        "{keyword}",
        "加{keyword}",
        "{keyword}联系",
        "请加{keyword}",
        "有事{keyword}聊",
        "办理{keyword}",
    )
    rows = [
        {
            "label": 1,
            "text": template.format(keyword=keyword),
        }
        for keyword in KEYWORD_VARIANTS
        for template in templates
    ]
    return pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
