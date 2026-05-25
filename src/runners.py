"""Experiment runners for the v1 traditional machine-learning model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from .modeling import build_v1_pipeline, evaluate_predictions, read_dataset, score_texts


DISPLAY_COLUMNS = [
    "dataset",
    "name",
    "n_samples",
    "n_normal",
    "n_spam",
    "accuracy",
    "precision_spam",
    "recall_spam",
    "f1_spam",
    "false_positive",
    "false_negative",
]


def _format_metric(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_v1_markdown(results: pd.DataFrame, output_path: str | Path) -> None:
    table = results[[column for column in DISPLAY_COLUMNS if column in results.columns]].copy()
    table = table.map(_format_metric)

    lines = [
        "# v1 字符级 Linear SVM 评测结果",
        "",
        "当前仓库只保留 v1：字符级 TF-IDF(1-3gram) + Linear SVM。",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_v1_on_split(
    dataset_name: str,
    source_path: str,
    texts: pd.Series,
    labels: pd.Series,
    model,
) -> dict[str, object]:
    pred_y = model.predict(texts)
    score_y = score_texts(model, texts)
    counts = labels.value_counts().to_dict()
    return {
        "dataset": dataset_name,
        "source_path": source_path,
        "name": "v1_char_tfidf_linear_svm",
        "description": "字符级 TF-IDF(1-3gram) + Linear SVM",
        "n_samples": int(len(labels)),
        "n_normal": int(counts.get(0, 0)),
        "n_spam": int(counts.get(1, 0)),
        **evaluate_predictions(labels, pred_y, score_y),
    }


def validate_v1(
    train_data_path: str | Path,
    eval_data_paths: Sequence[tuple[str, str | Path]] = (),
    output_csv: str | Path = "docs/experiments/multidataset/v1_validation.csv",
    output_md: str | Path = "docs/experiments/multidataset/v1_validation.md",
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train v1 on the main split and evaluate it on holdout/external data."""
    train_data = read_dataset(train_data_path)
    stratify = train_data["label"] if train_data["label"].nunique() > 1 else None
    train_x, test_x, train_y, test_y = train_test_split(
        train_data["text"],
        train_data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = build_v1_pipeline()
    model.fit(train_x, train_y)

    rows = [
        _evaluate_v1_on_split(
            "main_holdout",
            str(train_data_path),
            test_x.reset_index(drop=True),
            test_y.reset_index(drop=True),
            model,
        )
    ]

    for name, path in eval_data_paths:
        data = read_dataset(path)
        rows.append(
            _evaluate_v1_on_split(
                name,
                str(path),
                data["text"].reset_index(drop=True),
                data["label"].reset_index(drop=True),
                model,
            )
        )

    results = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    write_v1_markdown(results, output_md)
    return results


def compare_baselines(
    data_path: str | Path,
    output_csv: str | Path = "docs/experiments/classic/v1_baseline.csv",
    output_md: str | Path = "docs/experiments/classic/v1_baseline.md",
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compatibility wrapper: run the single retained v1 baseline."""
    return validate_v1(
        train_data_path=data_path,
        output_csv=output_csv,
        output_md=output_md,
        test_size=test_size,
        random_state=random_state,
    )


def compare_all_versions_multidataset_validation(
    train_data_path: str | Path,
    eval_data_paths: Sequence[tuple[str, str | Path]] = (),
    output_csv: str | Path = "docs/experiments/multidataset/v1_validation.csv",
    output_md: str | Path = "docs/experiments/multidataset/v1_validation.md",
    test_size: float = 0.3,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compatibility wrapper: the repository now keeps only v1."""
    _ = validation_size
    return validate_v1(
        train_data_path=train_data_path,
        eval_data_paths=eval_data_paths,
        output_csv=output_csv,
        output_md=output_md,
        test_size=test_size,
        random_state=random_state,
    )
