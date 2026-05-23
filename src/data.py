"""Dataset preparation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_COLUMNS = ("label", "tag", "class", "target", "y")
TEXT_COLUMNS = ("text", "content", "message", "sentence", "sms")


def _clean_labeled_frame(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[data["label"].isin(["0", "1"])]
    data["label"] = data["label"].astype(int)
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data = data.drop_duplicates(subset=["label", "text"])
    return data[["label", "text"]].reset_index(drop=True)


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def read_labeled_tsv(path: str | Path) -> pd.DataFrame:
    """Read a labeled TSV file with columns label and text."""
    data = pd.read_csv(
        path,
        sep="\t",
        names=["label", "text"],
        header=None,
        dtype={"label": "string", "text": "string"},
        on_bad_lines="skip",
    )
    return _clean_labeled_frame(data)


def read_flexible_labeled_dataset(path: str | Path) -> pd.DataFrame:
    """Read common labeled text formats and normalize to label/text columns."""
    path = Path(path)

    for sep in ("\t", ","):
        try:
            data = pd.read_csv(path, sep=sep, dtype="string", on_bad_lines="skip")
        except UnicodeDecodeError:
            data = pd.read_csv(
                path,
                sep=sep,
                dtype="string",
                encoding="gb18030",
                on_bad_lines="skip",
            )
        except pd.errors.ParserError:
            continue

        label_column = _find_column(data.columns, LABEL_COLUMNS)
        text_column = _find_column(data.columns, TEXT_COLUMNS)
        if label_column is not None and text_column is not None:
            normalized = data.rename(
                columns={label_column: "label", text_column: "text"}
            )
            return _clean_labeled_frame(normalized)

    return read_labeled_tsv(path)


def stratified_sample(
    data: pd.DataFrame,
    sample_size: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a class-balanced-ish stratified sample preserving label ratio."""
    if sample_size is None or sample_size >= len(data):
        return data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    pieces = []
    for _, group in data.groupby("label"):
        ratio = len(group) / len(data)
        n = max(1, round(sample_size * ratio))
        n = min(n, len(group))
        pieces.append(group.sample(n=n, random_state=random_state))
    sampled = pd.concat(pieces, ignore_index=True)

    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def prepare_labeled_dataset(
    raw_path: str | Path,
    output_path: str | Path,
    sample_size: int | None = 20000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Normalize and optionally sample a labeled spam dataset."""
    data = read_labeled_tsv(raw_path)
    data = stratified_sample(data, sample_size=sample_size, random_state=random_state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, sep="\t", index=False)
    return data


def prepare_ast_dataset(
    raw_path: str | Path,
    output_path: str | Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Normalize an AST-style dataset to label/text TSV."""
    data = read_flexible_labeled_dataset(raw_path)
    data = stratified_sample(data, sample_size=sample_size, random_state=random_state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, sep="\t", index=False)
    return data


def dataset_summary(data: pd.DataFrame) -> dict[str, int]:
    counts = data["label"].value_counts().to_dict()
    return {
        "rows": int(len(data)),
        "normal": int(counts.get(0, 0)),
        "spam": int(counts.get(1, 0)),
    }
