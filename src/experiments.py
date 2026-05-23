"""Reusable experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from .adversarial import keyword_training_samples
from .modeling import build_pipeline, evaluate_predictions, read_dataset, score_texts


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    analyzer: str
    classifier: str
    description: str
    augment_keywords: bool = False


BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="word_tfidf_logreg",
        analyzer="word",
        classifier="logreg",
        description="词级 TF-IDF(1-2gram) + Logistic Regression",
    ),
    BaselineSpec(
        name="char_tfidf_logreg",
        analyzer="char",
        classifier="logreg",
        description="字符级 TF-IDF(1-3gram) + Logistic Regression",
    ),
    BaselineSpec(
        name="word_tfidf_linear_svm",
        analyzer="word",
        classifier="linear_svm",
        description="词级 TF-IDF(1-2gram) + Linear SVM",
    ),
    BaselineSpec(
        name="char_tfidf_linear_svm",
        analyzer="char",
        classifier="linear_svm",
        description="字符级 TF-IDF(1-3gram) + Linear SVM",
    ),
)


CSN_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="char_tfidf_linear_svm",
        analyzer="char",
        classifier="linear_svm",
        description="字符级 TF-IDF(1-3gram) + Linear SVM",
    ),
    BaselineSpec(
        name="char_csn_tfidf_linear_svm",
        analyzer="char_csn",
        classifier="linear_svm",
        description="字符相似性归一化 + 字符级 TF-IDF(1-3gram) + Linear SVM",
    ),
    BaselineSpec(
        name="char_csn_aug_tfidf_linear_svm",
        analyzer="char_csn",
        classifier="linear_svm",
        description="字符相似性归一化 + 关键词增强 + 字符级 TF-IDF + Linear SVM",
        augment_keywords=True,
    ),
    BaselineSpec(
        name="char_tfidf_logreg",
        analyzer="char",
        classifier="logreg",
        description="字符级 TF-IDF(1-3gram) + Logistic Regression",
    ),
    BaselineSpec(
        name="char_csn_tfidf_logreg",
        analyzer="char_csn",
        classifier="logreg",
        description="字符相似性归一化 + 字符级 TF-IDF(1-3gram) + Logistic Regression",
    ),
    BaselineSpec(
        name="char_csn_aug_tfidf_logreg",
        analyzer="char_csn",
        classifier="logreg",
        description="字符相似性归一化 + 关键词增强 + 字符级 TF-IDF + Logistic Regression",
        augment_keywords=True,
    ),
)


DISPLAY_COLUMNS = [
    "name",
    "description",
    "accuracy",
    "macro_f1",
    "f1_spam",
    "precision_spam",
    "recall_spam",
    "pr_auc",
    "roc_auc",
    "recall_at_precision_90",
    "recall_at_precision_95",
    "false_positive_rate",
]


def _format_metric(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available_columns = [column for column in DISPLAY_COLUMNS if column in results.columns]
    table = results[available_columns].copy()
    table = table.map(_format_metric)

    lines = [
        "# Baseline 对比实验结果",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_baselines(
    data_path: str | Path,
    output_csv: str | Path = "docs/baseline_comparison.csv",
    output_md: str | Path = "docs/baseline_comparison.md",
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train and evaluate all baseline specs on the same split."""
    data = read_dataset(data_path)
    stratify = data["label"] if data["label"].nunique() > 1 else None
    train_x, test_x, train_y, test_y = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    rows: list[dict[str, object]] = []
    for spec in BASELINE_SPECS:
        model = build_pipeline(analyzer=spec.analyzer, classifier=spec.classifier)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        score_y = score_texts(model, test_x)
        metrics = evaluate_predictions(test_y, pred_y, score_y)
        confusion = confusion_matrix(test_y, pred_y, labels=[0, 1]).tolist()
        rows.append(
            {
                "name": spec.name,
                "description": spec.description,
                "analyzer": spec.analyzer,
                "classifier": spec.classifier,
                "confusion_matrix": str(confusion),
                **metrics,
            }
        )

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["f1_spam", "recall_at_precision_95", "pr_auc"],
        ascending=False,
    ).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    write_markdown_table(results, output_md)
    return results


def compare_csn_optimization(
    data_path: str | Path,
    adversarial_path: str | Path,
    output_csv: str | Path = "docs/csn_comparison.csv",
    output_md: str | Path = "docs/csn_comparison.md",
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare plain character models with CSN-normalized variants."""
    data = read_dataset(data_path)
    adversarial = read_dataset(adversarial_path)
    stratify = data["label"] if data["label"].nunique() > 1 else None
    train_x, test_x, train_y, test_y = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    rows: list[dict[str, object]] = []
    for spec in CSN_SPECS:
        model = build_pipeline(analyzer=spec.analyzer, classifier=spec.classifier)
        fit_x = train_x
        fit_y = train_y
        if spec.augment_keywords:
            augment = keyword_training_samples()
            fit_x = pd.concat([train_x, augment["text"]], ignore_index=True)
            fit_y = pd.concat([train_y, augment["label"]], ignore_index=True)
        model.fit(fit_x, fit_y)

        clean_pred = model.predict(test_x)
        clean_score = score_texts(model, test_x)
        clean_metrics = evaluate_predictions(test_y, clean_pred, clean_score)

        adv_pred = model.predict(adversarial["text"])
        adv_score = score_texts(model, adversarial["text"])
        adv_metrics = evaluate_predictions(adversarial["label"], adv_pred, adv_score)

        row = {
            "name": spec.name,
            "description": spec.description,
            "analyzer": spec.analyzer,
            "classifier": spec.classifier,
            "augment_keywords": spec.augment_keywords,
        }
        row.update({f"clean_{key}": value for key, value in clean_metrics.items()})
        row.update({f"adv_{key}": value for key, value in adv_metrics.items()})
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        by=["adv_recall_spam", "clean_f1_spam"],
        ascending=False,
    ).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    table_columns = [
        "name",
        "description",
        "clean_accuracy",
        "clean_f1_spam",
        "clean_recall_spam",
        "clean_recall_at_precision_95",
        "adv_recall_spam",
        "adv_f1_spam",
        "adv_false_negative",
    ]
    table = results[table_columns].copy()
    table = table.map(_format_metric)
    lines = [
        "# 字符相似性网络优化对比",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    return results
