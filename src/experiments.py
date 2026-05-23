"""Reusable experiment runners."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from .adversarial import keyword_training_samples
from .modeling import build_pipeline, evaluate_predictions, read_dataset, score_texts
from .risk_features import spam_risk_scores


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

RISK_BONUS_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
THRESHOLD_GRID = tuple(value / 100 for value in range(-20, 121, 5))


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


def _with_keyword_augmentation(train_x: pd.Series, train_y: pd.Series) -> tuple[pd.Series, pd.Series]:
    augment = keyword_training_samples()
    fit_x = pd.concat([train_x, augment["text"]], ignore_index=True)
    fit_y = pd.concat([train_y, augment["label"]], ignore_index=True)
    return fit_x, fit_y


def _adjusted_scores(model, texts, risk_bonus: float) -> np.ndarray:
    base_scores = score_texts(model, texts)
    if base_scores is None:
        raise ValueError("The model must expose predict_proba or decision_function.")
    risk_scores = np.asarray(spam_risk_scores(texts), dtype=float)
    return np.asarray(base_scores, dtype=float) + risk_bonus * risk_scores


def _evaluate_threshold(y_true, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return evaluate_predictions(y_true, pred, scores)


def _search_risk_threshold(
    model,
    texts,
    labels,
    risk_bonuses: tuple[float, ...] = RISK_BONUS_GRID,
    thresholds: tuple[float, ...] = THRESHOLD_GRID,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for risk_bonus in risk_bonuses:
        scores = _adjusted_scores(model, texts, risk_bonus)
        for threshold in thresholds:
            metrics = _evaluate_threshold(labels, scores, threshold)
            rows.append(
                {
                    "risk_bonus": risk_bonus,
                    "threshold": threshold,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _select_best_threshold(grid: pd.DataFrame) -> pd.Series:
    return grid.sort_values(
        by=["f1_spam", "accuracy", "precision_spam", "recall_spam"],
        ascending=False,
    ).iloc[0]


def _evaluate_risk_config(
    name: str,
    description: str,
    model,
    clean_x,
    clean_y,
    adversarial: pd.DataFrame,
    risk_bonus: float,
    threshold: float,
) -> dict[str, object]:
    clean_scores = _adjusted_scores(model, clean_x, risk_bonus)
    clean_metrics = _evaluate_threshold(clean_y, clean_scores, threshold)

    adv_scores = _adjusted_scores(model, adversarial["text"], risk_bonus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adv_metrics = _evaluate_threshold(adversarial["label"], adv_scores, threshold)

    row: dict[str, object] = {
        "name": name,
        "description": description,
        "risk_bonus": risk_bonus,
        "threshold": threshold,
    }
    row.update({f"clean_{key}": value for key, value in clean_metrics.items()})
    row.update({f"adv_{key}": value for key, value in adv_metrics.items()})
    return row


def write_bad_case_markdown(
    results: pd.DataFrame,
    output_path: str | Path,
    selected_bonus: float,
    selected_threshold: float,
) -> None:
    table_columns = [
        "name",
        "description",
        "risk_bonus",
        "threshold",
        "clean_accuracy",
        "clean_precision_spam",
        "clean_recall_spam",
        "clean_f1_spam",
        "clean_false_positive",
        "clean_false_negative",
        "adv_recall_spam",
        "adv_false_negative",
    ]
    table = results[table_columns].copy()
    table = table.map(_format_metric)

    lines = [
        "# Bad-case 驱动阈值优化对比",
        "",
        f"- 验证集选择参数：risk_bonus={selected_bonus:.2f}, threshold={selected_threshold:.2f}",
        "- `v4_eval_oracle` 是在测试集上扫描得到的上界，只用于分析，不作为严格泛化结果。",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def compare_bad_case_optimization(
    data_path: str | Path,
    adversarial_path: str | Path,
    output_csv: str | Path = "docs/bad_case_optimization.csv",
    output_md: str | Path = "docs/bad_case_optimization.md",
    grid_csv: str | Path = "docs/bad_case_tuning_grid.csv",
    test_size: float = 0.3,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Tune a CSN-aware model with bad-case risk features and thresholds."""
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

    valid_stratify = train_y if train_y.nunique() > 1 else None
    fit_x, valid_x, fit_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=validation_size,
        random_state=random_state,
        stratify=valid_stratify,
    )

    baseline_model = build_pipeline(analyzer="char", classifier="linear_svm")
    baseline_model.fit(train_x, train_y)

    csn_model = build_pipeline(analyzer="char_csn", classifier="linear_svm")
    csn_fit_x, csn_fit_y = _with_keyword_augmentation(train_x, train_y)
    csn_model.fit(csn_fit_x, csn_fit_y)

    tuning_model = build_pipeline(analyzer="char_csn", classifier="linear_svm")
    tune_fit_x, tune_fit_y = _with_keyword_augmentation(fit_x, fit_y)
    tuning_model.fit(tune_fit_x, tune_fit_y)
    validation_grid = _search_risk_threshold(tuning_model, valid_x, valid_y)
    selected = _select_best_threshold(validation_grid)
    selected_bonus = float(selected["risk_bonus"])
    selected_threshold = float(selected["threshold"])

    eval_grid = _search_risk_threshold(csn_model, test_x, test_y)
    eval_oracle = _select_best_threshold(eval_grid)

    grid_output = Path(grid_csv)
    grid_output.parent.mkdir(parents=True, exist_ok=True)
    tuning_grid = pd.concat(
        [
            validation_grid.assign(split="validation"),
            eval_grid.assign(split="evaluation"),
        ],
        ignore_index=True,
    )
    tuning_grid.to_csv(grid_output, index=False)

    rows = [
        _evaluate_risk_config(
            name="v1_strong_baseline_default",
            description="字符级 TF-IDF + Linear SVM，默认阈值",
            model=baseline_model,
            clean_x=test_x,
            clean_y=test_y,
            adversarial=adversarial,
            risk_bonus=0.0,
            threshold=0.0,
        ),
        _evaluate_risk_config(
            name="v3_csn_aug_default",
            description="CSN 归一化 + 关键词增强，默认阈值",
            model=csn_model,
            clean_x=test_x,
            clean_y=test_y,
            adversarial=adversarial,
            risk_bonus=0.0,
            threshold=0.0,
        ),
        _evaluate_risk_config(
            name="v4_threshold_only",
            description="CSN 关键词增强 + 验证集阈值调优",
            model=csn_model,
            clean_x=test_x,
            clean_y=test_y,
            adversarial=adversarial,
            risk_bonus=0.0,
            threshold=selected_threshold,
        ),
        _evaluate_risk_config(
            name="v4_bad_case_valid_tuned",
            description="CSN 关键词增强 + bad-case 风险分数 + 验证集阈值调优",
            model=csn_model,
            clean_x=test_x,
            clean_y=test_y,
            adversarial=adversarial,
            risk_bonus=selected_bonus,
            threshold=selected_threshold,
        ),
        _evaluate_risk_config(
            name="v4_eval_oracle",
            description="CSN 关键词增强 + bad-case 风险分数 + 测试集扫描上界",
            model=csn_model,
            clean_x=test_x,
            clean_y=test_y,
            adversarial=adversarial,
            risk_bonus=float(eval_oracle["risk_bonus"]),
            threshold=float(eval_oracle["threshold"]),
        ),
    ]

    results = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    write_bad_case_markdown(
        results,
        output_md,
        selected_bonus=selected_bonus,
        selected_threshold=selected_threshold,
    )
    return results
