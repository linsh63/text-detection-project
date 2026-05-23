"""Dataset preparation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_COLUMNS = ("label", "tag", "class", "target", "y")
TEXT_COLUMNS = ("text", "content", "message", "sentence", "sms")
FBS_IGNORED_FILES = {"README.md", ".gitattributes"}


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


def read_fbs_spam_messages(raw_dir: str | Path) -> pd.DataFrame:
    """Read the FBS spam-only corpus released as one plain-text file per category."""
    raw_dir = Path(raw_dir)
    rows: list[dict[str, str | int]] = []
    for path in sorted(raw_dir.iterdir()):
        if path.name.startswith(".") or path.name in FBS_IGNORED_FILES or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(encoding="gb18030").splitlines()
        for line in lines:
            text = line.strip()
            if text:
                rows.append({"label": 1, "text": text, "source_category": path.name})

    data = pd.DataFrame(rows)
    if data.empty:
        return pd.DataFrame(columns=["label", "text", "source_category"])
    return data.drop_duplicates(subset=["text"]).reset_index(drop=True)


def prepare_fbs_mixed_dataset(
    fbs_dir: str | Path,
    normal_raw_path: str | Path,
    output_path: str | Path,
    sample_size: int | None = 10000,
    spam_ratio: float = 0.5,
    exclude_path: str | Path | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a binary cross-source set from FBS spam plus normal messages."""
    if not 0 < spam_ratio < 1:
        raise ValueError("spam_ratio must be between 0 and 1.")

    spam = read_fbs_spam_messages(fbs_dir)[["label", "text"]]
    normal = read_labeled_tsv(normal_raw_path)
    normal = normal[normal["label"] == 0][["label", "text"]]

    if exclude_path is not None:
        excluded = read_flexible_labeled_dataset(exclude_path)
        normal = normal[~normal["text"].isin(set(excluded["text"]))]

    if sample_size is None:
        n_spam = min(len(spam), len(normal))
        n_normal = n_spam
    else:
        n_spam = min(round(sample_size * spam_ratio), len(spam))
        n_normal = min(sample_size - n_spam, len(normal))

    if n_spam == 0 or n_normal == 0:
        raise ValueError("Not enough spam or normal messages to build a mixed dataset.")

    sampled = pd.concat(
        [
            spam.sample(n=n_spam, random_state=random_state),
            normal.sample(n=n_normal, random_state=random_state),
        ],
        ignore_index=True,
    )
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, sep="\t", index=False)
    return sampled


def dataset_summary(data: pd.DataFrame) -> dict[str, int]:
    counts = data["label"].value_counts().to_dict()
    return {
        "rows": int(len(data)),
        "normal": int(counts.get(0, 0)),
        "spam": int(counts.get(1, 0)),
    }
