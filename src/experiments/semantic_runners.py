"""Semantic v8 experiment runners."""

from __future__ import annotations

from collections import Counter
from os.path import relpath
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from ..encoders.sentence_transformer import SentenceTransformerEncoder
from ..models.modeling import read_dataset
from ..preprocessing.normalizer import normalize_for_semantic_model
from .runners import (
    _evaluate_threshold,
    _search_score_threshold,
    _select_best_fusion_threshold,
    write_evaluation_protocol_markdown,
)


DEFAULT_ENCODERS: tuple[tuple[str, str], ...] = (
    ("bge_small_zh", "BAAI/bge-small-zh-v1.5"),
    ("multilingual_minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ("m3e_small", "moka-ai/m3e-small"),
)


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value.lower())
    return "_".join(part for part in safe.split("_") if part)


def _is_chinese_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _is_informative_ngram(term: str) -> bool:
    if len(term) < 2:
        return False
    signal_chars = [char for char in term if char.isalnum() or _is_chinese_char(char)]
    if len(signal_chars) < 2:
        return False
    if len(set(signal_chars)) == 1:
        return False
    return len(signal_chars) / max(len(term), 1) >= 0.7


def _char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> set[str]:
    normalized = normalize_for_semantic_model(text)
    compact = "".join(char for char in normalized if not char.isspace())
    terms: set[str] = set()
    for size in range(min_n, max_n + 1):
        if len(compact) < size:
            continue
        for index in range(0, len(compact) - size + 1):
            term = compact[index : index + size]
            if _is_informative_ngram(term):
                terms.add(term)
    return terms


def _hard_positive_texts(texts: pd.Series, labels: pd.Series, vectors: np.ndarray) -> pd.Series:
    if labels.nunique() < 2 or int((labels == 1).sum()) == 0:
        return texts.iloc[0:0]
    classifier = _fit_logreg(vectors, labels)
    scores = _positive_scores(classifier, vectors)
    spam_scores = scores[labels.to_numpy(dtype=int) == 1]
    cutoff = float(np.quantile(spam_scores, 0.35))
    mask = (labels.to_numpy(dtype=int) == 1) & (scores <= cutoff)
    return texts.reset_index(drop=True)[mask]


def _build_pinyin_index(texts: pd.Series, max_alternatives: int = 8) -> dict[str, list[str]]:
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return {}

    char_counts: Counter[str] = Counter()
    for text in texts:
        for char in normalize_for_semantic_model(text):
            if _is_chinese_char(char):
                char_counts[char] += 1

    grouped: dict[str, list[tuple[str, int]]] = {}
    for char, count in char_counts.items():
        pinyin = lazy_pinyin(char, errors="ignore")
        if not pinyin:
            continue
        grouped.setdefault(pinyin[0], []).append((char, count))

    return {
        pinyin: [char for char, _ in sorted(chars, key=lambda item: (-item[1], item[0]))[:max_alternatives]]
        for pinyin, chars in grouped.items()
    }


def _extract_auto_hard_terms(
    texts: pd.Series,
    labels: pd.Series,
    hard_texts: pd.Series,
    scope: str,
    max_terms: int = 120,
    min_spam_df: int = 2,
) -> pd.DataFrame:
    spam_counts: Counter[str] = Counter()
    normal_counts: Counter[str] = Counter()
    hard_counts: Counter[str] = Counter()

    for text, label in zip(texts, labels, strict=True):
        terms = _char_ngrams(text)
        if int(label) == 1:
            spam_counts.update(terms)
        else:
            normal_counts.update(terms)

    for text in hard_texts:
        hard_counts.update(_char_ngrams(text))

    rows: list[dict[str, object]] = []
    for term, spam_df in spam_counts.items():
        if spam_df < min_spam_df:
            continue
        normal_df = normal_counts.get(term, 0)
        hard_df = hard_counts.get(term, 0)
        risk_ratio = (spam_df + 1.0 + 2.0 * hard_df) / (normal_df + 1.0)
        score = risk_ratio * np.log1p(spam_df + hard_df)
        rows.append(
            {
                "scope": scope,
                "term": term,
                "spam_df": int(spam_df),
                "normal_df": int(normal_df),
                "hard_df": int(hard_df),
                "score": float(score),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["scope", "term", "spam_df", "normal_df", "hard_df", "score"])
    data = pd.DataFrame(rows)
    return (
        data.sort_values(
            by=["score", "hard_df", "spam_df", "normal_df", "term"],
            ascending=[False, False, False, True, True],
        )
        .head(max_terms)
        .reset_index(drop=True)
    )


def _auto_variants(term: str, pinyin_index: dict[str, list[str]]) -> list[str]:
    variants = {
        term,
        " ".join(term),
        ".".join(term),
        "/".join(term),
        "_".join(term),
    }
    if len(term) >= 3:
        variants.add(term[0] + " " + term[1:])
        variants.add(term[:-1] + " " + term[-1])
        variants.add(term[:1] + "-" + term[1:])
        variants.add(term[:-1] + "-" + term[-1])
    if len(term) >= 2:
        variants.add(term[0] * 2 + term[1:])
        variants.add(term[0] + term[1] * 2 + term[2:])

    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        lazy_pinyin = None

    if lazy_pinyin is not None:
        chars = list(term)
        for index, char in enumerate(chars):
            if not _is_chinese_char(char):
                continue
            pinyin = lazy_pinyin(char, errors="ignore")
            if not pinyin:
                continue
            added = 0
            for alternative in pinyin_index.get(pinyin[0], []):
                if alternative == char:
                    continue
                replaced = chars.copy()
                replaced[index] = alternative
                variants.add("".join(replaced))
                added += 1
                if added >= 2:
                    break

    return sorted(variant for variant in variants if _is_informative_ngram(variant.replace(" ", "")))


def _make_auto_augmentation(
    texts: pd.Series,
    labels: pd.Series,
    vectors: np.ndarray,
    scope: str,
    max_terms: int = 120,
    max_augmented: int = 800,
    min_spam_df: int = 2,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True).astype(int)
    hard_texts = _hard_positive_texts(texts, labels, vectors)
    terms = _extract_auto_hard_terms(
        texts,
        labels,
        hard_texts=hard_texts,
        scope=scope,
        max_terms=max_terms,
        min_spam_df=min_spam_df,
    )
    pinyin_index = _build_pinyin_index(texts)
    generated_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for _, row in terms.iterrows():
        term = str(row["term"])
        for variant in _auto_variants(term, pinyin_index):
            if variant in seen:
                continue
            seen.add(variant)
            generated_rows.append({"scope": scope, "source_term": term, "text": variant, "label": 1})
            if len(generated_rows) >= max_augmented:
                break
        if len(generated_rows) >= max_augmented:
            break

    examples = pd.DataFrame(generated_rows, columns=["scope", "source_term", "text", "label"])
    if examples.empty:
        return (
            pd.Series(dtype=str),
            pd.Series(dtype=int),
            terms,
            examples,
        )
    return examples["text"].astype(str), examples["label"].astype(int), terms, examples


def _fit_augmented_classifier(
    encoder: SentenceTransformerEncoder,
    texts: pd.Series,
    labels: pd.Series,
    vectors: np.ndarray,
    scope: str,
    batch_size: int,
    max_terms: int,
    max_augmented: int,
    min_spam_df: int,
) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame]:
    aug_texts, aug_labels, terms, examples = _make_auto_augmentation(
        texts=texts,
        labels=labels,
        vectors=vectors,
        scope=scope,
        max_terms=max_terms,
        max_augmented=max_augmented,
        min_spam_df=min_spam_df,
    )
    if len(aug_texts):
        aug_vectors = encoder.encode(aug_texts, batch_size=batch_size)
        fit_vectors = np.vstack([vectors, aug_vectors])
        fit_labels = pd.concat([labels.reset_index(drop=True), aug_labels.reset_index(drop=True)], ignore_index=True)
    else:
        fit_vectors = vectors
        fit_labels = labels.reset_index(drop=True)
    classifier = _fit_logreg(fit_vectors, fit_labels)
    return classifier, terms, examples


def _normal_hard_terms(
    texts: pd.Series,
    labels: pd.Series,
    normal_hard_texts: pd.Series,
    scope: str,
    max_terms: int,
    min_normal_df: int,
) -> pd.DataFrame:
    inverted = 1 - labels.reset_index(drop=True).astype(int)
    return _extract_auto_hard_terms(
        texts=texts,
        labels=inverted,
        hard_texts=normal_hard_texts,
        scope=f"{scope}_negative",
        max_terms=max_terms,
        min_spam_df=min_normal_df,
    )


def _make_filtered_auto_augmentation(
    encoder: SentenceTransformerEncoder,
    texts: pd.Series,
    labels: pd.Series,
    vectors: np.ndarray,
    scope: str,
    batch_size: int,
    max_terms: int = 80,
    max_augmented: int = 200,
    min_spam_df: int = 3,
    max_hard_negatives: int = 200,
    positive_min_score: float = 0.05,
    positive_max_score: float = 0.75,
    hard_negative_min_score: float = 0.25,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    texts = texts.reset_index(drop=True)
    labels = labels.reset_index(drop=True).astype(int)
    base_classifier = _fit_logreg(vectors, labels)
    base_scores = _positive_scores(base_classifier, vectors)

    pos_texts, pos_labels, pos_terms, pos_examples = _make_auto_augmentation(
        texts=texts,
        labels=labels,
        vectors=vectors,
        scope=scope,
        max_terms=max_terms,
        max_augmented=max_augmented,
        min_spam_df=min_spam_df,
    )
    if len(pos_examples):
        pos_vectors = encoder.encode(pos_texts, batch_size=batch_size)
        pos_scores = _positive_scores(base_classifier, pos_vectors)
        term_stats = pos_terms.set_index("term")[["spam_df", "normal_df", "hard_df", "score"]].to_dict("index")
        keep_rows: list[dict[str, object]] = []
        for index, example in pos_examples.reset_index(drop=True).iterrows():
            term = str(example["source_term"])
            stats = term_stats.get(term, {})
            term_length = len(term)
            normal_df = int(stats.get("normal_df", 0))
            hard_df = int(stats.get("hard_df", 0))
            score = float(pos_scores[index])
            is_specific = term_length >= 3 or normal_df == 0 or hard_df >= 2
            if positive_min_score <= score <= positive_max_score and is_specific:
                row = example.to_dict()
                row["augmentation_type"] = "filtered_positive"
                row["base_score"] = score
                keep_rows.append(row)
        pos_examples = pd.DataFrame(keep_rows)
    else:
        pos_examples = pd.DataFrame(columns=["scope", "source_term", "text", "label", "augmentation_type", "base_score"])

    normal_mask = labels.to_numpy(dtype=int) == 0
    hard_normal_candidates = pd.DataFrame(
        {
            "text": texts[normal_mask].reset_index(drop=True),
            "score": base_scores[normal_mask],
        }
    ).sort_values(["score", "text"], ascending=[False, True])
    hard_normal_candidates = hard_normal_candidates[
        hard_normal_candidates["score"] >= hard_negative_min_score
    ].head(max_hard_negatives // 2)
    hard_negative_rows = [
        {
            "scope": scope,
            "source_term": "__full_normal__",
            "text": row["text"],
            "label": 0,
            "augmentation_type": "hard_negative_full",
            "base_score": float(row["score"]),
        }
        for _, row in hard_normal_candidates.iterrows()
    ]

    neg_terms = _normal_hard_terms(
        texts=texts,
        labels=labels,
        normal_hard_texts=hard_normal_candidates["text"],
        scope=scope,
        max_terms=max_terms,
        min_normal_df=min_spam_df,
    )
    pinyin_index = _build_pinyin_index(texts)
    seen_negative_texts = {str(row["text"]) for row in hard_negative_rows}
    candidate_negative_rows: list[dict[str, object]] = []
    for _, row in neg_terms.iterrows():
        term = str(row["term"])
        for variant in _auto_variants(term, pinyin_index):
            if variant in seen_negative_texts:
                continue
            seen_negative_texts.add(variant)
            candidate_negative_rows.append(
                {
                    "scope": scope,
                    "source_term": term,
                    "text": variant,
                    "label": 0,
                    "augmentation_type": "hard_negative_variant",
                }
            )
            if len(candidate_negative_rows) >= max_hard_negatives:
                break
        if len(candidate_negative_rows) >= max_hard_negatives:
            break
    if candidate_negative_rows:
        candidate_negative_examples = pd.DataFrame(candidate_negative_rows)
        candidate_vectors = encoder.encode(candidate_negative_examples["text"].astype(str), batch_size=batch_size)
        candidate_scores = _positive_scores(base_classifier, candidate_vectors)
        for index, score in enumerate(candidate_scores):
            if score >= hard_negative_min_score and len(hard_negative_rows) < max_hard_negatives:
                row = candidate_negative_examples.iloc[index].to_dict()
                row["base_score"] = float(score)
                hard_negative_rows.append(row)

    neg_examples = pd.DataFrame(hard_negative_rows)
    examples = pd.concat([pos_examples, neg_examples], ignore_index=True)
    if examples.empty:
        terms = pd.concat([pos_terms, neg_terms], ignore_index=True)
        return pd.Series(dtype=str), pd.Series(dtype=int), terms, examples
    terms = pd.concat(
        [
            pos_terms.assign(term_type="positive"),
            neg_terms.assign(term_type="hard_negative"),
        ],
        ignore_index=True,
    )
    return examples["text"].astype(str), examples["label"].astype(int), terms, examples


def _fit_filtered_augmented_classifier(
    encoder: SentenceTransformerEncoder,
    texts: pd.Series,
    labels: pd.Series,
    vectors: np.ndarray,
    scope: str,
    batch_size: int,
    max_terms: int,
    max_augmented: int,
    min_spam_df: int,
    max_hard_negatives: int,
    positive_min_score: float,
    positive_max_score: float,
    hard_negative_min_score: float,
) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame]:
    aug_texts, aug_labels, terms, examples = _make_filtered_auto_augmentation(
        encoder=encoder,
        texts=texts,
        labels=labels,
        vectors=vectors,
        scope=scope,
        batch_size=batch_size,
        max_terms=max_terms,
        max_augmented=max_augmented,
        min_spam_df=min_spam_df,
        max_hard_negatives=max_hard_negatives,
        positive_min_score=positive_min_score,
        positive_max_score=positive_max_score,
        hard_negative_min_score=hard_negative_min_score,
    )
    if len(aug_texts):
        aug_vectors = encoder.encode(aug_texts, batch_size=batch_size)
        fit_vectors = np.vstack([vectors, aug_vectors])
        fit_labels = pd.concat([labels.reset_index(drop=True), aug_labels.reset_index(drop=True)], ignore_index=True)
    else:
        fit_vectors = vectors
        fit_labels = labels.reset_index(drop=True)
    classifier = _fit_logreg(fit_vectors, fit_labels)
    return classifier, terms, examples


def _fit_logreg(vectors: np.ndarray, labels) -> LogisticRegression:
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    classifier.fit(vectors, labels)
    return classifier


def _positive_scores(classifier: LogisticRegression, vectors: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return np.asarray(classifier.predict_proba(vectors)[:, 1], dtype=float)
    return np.asarray(classifier.decision_function(vectors), dtype=float)


def _select_threshold(classifier: LogisticRegression, vectors: np.ndarray, labels) -> float:
    scores = _positive_scores(classifier, vectors)
    grid = _search_score_threshold(scores, labels)
    selected = _select_best_fusion_threshold(grid)
    return float(selected["threshold"])


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _error_types(labels: pd.Series, scores: np.ndarray, threshold: float) -> list[str]:
    preds = (scores >= threshold).astype(int)
    errors: list[str] = []
    for label, pred in zip(labels.astype(int), preds, strict=True):
        if label == pred:
            errors.append("correct")
        elif label == 1:
            errors.append("false_negative")
        else:
            errors.append("false_positive")
    return errors


def _score_stats(labels: pd.Series, scores: np.ndarray) -> dict[str, float]:
    label_values = labels.to_numpy(dtype=int)
    normal_scores = scores[label_values == 0]
    spam_scores = scores[label_values == 1]

    def stat(values: np.ndarray, fn, default: float = float("nan")) -> float:
        return float(fn(values)) if len(values) else default

    mean_normal = stat(normal_scores, np.mean)
    mean_spam = stat(spam_scores, np.mean)
    return {
        "score_mean_normal": mean_normal,
        "score_mean_spam": mean_spam,
        "score_gap_spam_minus_normal": mean_spam - mean_normal
        if not np.isnan(mean_normal) and not np.isnan(mean_spam)
        else float("nan"),
        "score_p95_normal": stat(normal_scores, lambda values: np.quantile(values, 0.95)),
        "score_p05_spam": stat(spam_scores, lambda values: np.quantile(values, 0.05)),
    }


def _diagnosis(row: dict[str, object]) -> str:
    if int(row["n_normal"]) == 0 or int(row["n_spam"]) == 0:
        return "spam-only challenge: use as threshold sensitivity check"
    current_f1 = float(row["current_f1_spam"])
    oracle_f1 = float(row["oracle_f1_spam"])
    pr_auc = float(row.get("pr_auc", float("nan")))
    f1_gain = oracle_f1 - current_f1
    if current_f1 >= 0.9:
        return "healthy under current threshold"
    if not np.isnan(pr_auc) and pr_auc < 0.6:
        return "representation/domain mismatch: oracle threshold is degenerate; calibration alone is insufficient"
    if f1_gain >= 0.2:
        return "calibration issue: score ranking is useful but threshold is too conservative"
    return "mixed issue: try encoder comparison plus calibration"


def _evaluation_record(
    protocol_id: str,
    protocol_name: str,
    dataset: str,
    dataset_type: str,
    source_path: str,
    model_version: str,
    training_scope: str,
    labels: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = labels.reset_index(drop=True).astype(int)
    scores = np.asarray(scores, dtype=float)
    current_metrics = _evaluate_threshold(labels, scores, threshold)
    grid = _search_score_threshold(scores, labels)
    oracle = _select_best_fusion_threshold(grid)
    oracle_threshold = float(oracle["threshold"])
    oracle_metrics = _evaluate_threshold(labels, scores, oracle_threshold)
    counts = labels.value_counts().to_dict()

    record: dict[str, object] = {
        "protocol_id": protocol_id,
        "protocol_name": protocol_name,
        "dataset": dataset,
        "dataset_type": dataset_type,
        "source_path": source_path,
        "model_version": model_version,
        "training_scope": training_scope,
        "threshold_current": threshold,
        "threshold_oracle": oracle_threshold,
        "n_samples": int(len(labels)),
        "n_normal": int(counts.get(0, 0)),
        "n_spam": int(counts.get(1, 0)),
        "current_accuracy": current_metrics["accuracy"],
        "current_precision_spam": current_metrics["precision_spam"],
        "current_recall_spam": current_metrics["recall_spam"],
        "current_f1_spam": current_metrics["f1_spam"],
        "current_false_positive": current_metrics["false_positive"],
        "current_false_negative": current_metrics["false_negative"],
        "oracle_accuracy": oracle_metrics["accuracy"],
        "oracle_precision_spam": oracle_metrics["precision_spam"],
        "oracle_recall_spam": oracle_metrics["recall_spam"],
        "oracle_f1_spam": oracle_metrics["f1_spam"],
        "oracle_false_positive": oracle_metrics["false_positive"],
        "oracle_false_negative": oracle_metrics["false_negative"],
        "oracle_f1_gain": oracle_metrics["f1_spam"] - current_metrics["f1_spam"],
        "oracle_recall_gain": oracle_metrics["recall_spam"] - current_metrics["recall_spam"],
        "roc_auc": current_metrics.get("roc_auc", float("nan")),
        "pr_auc": current_metrics.get("pr_auc", float("nan")),
        "recall_at_precision_90": current_metrics.get("recall_at_precision_90", float("nan")),
        "recall_at_precision_95": current_metrics.get("recall_at_precision_95", float("nan")),
        **_score_stats(labels, scores),
    }
    record["diagnosis"] = _diagnosis(record)

    grid = grid.copy()
    grid.insert(0, "training_scope", training_scope)
    grid.insert(0, "model_version", model_version)
    grid.insert(0, "dataset", dataset)
    grid.insert(0, "protocol_id", protocol_id)

    score_samples = pd.DataFrame(
        {
            "protocol_id": protocol_id,
            "dataset": dataset,
            "dataset_type": dataset_type,
            "model_version": model_version,
            "training_scope": training_scope,
            "sample_index": np.arange(len(labels)),
            "label": labels,
            "score": scores,
            "threshold_current": threshold,
            "pred_current": (scores >= threshold).astype(int),
            "error_type": _error_types(labels, scores, threshold),
        }
    )

    if labels.nunique() == 2:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        threshold_values = np.append(thresholds, np.nan)
        pr_curve = pd.DataFrame(
            {
                "protocol_id": protocol_id,
                "dataset": dataset,
                "model_version": model_version,
                "training_scope": training_scope,
                "precision": precision,
                "recall": recall,
                "threshold": threshold_values,
            }
        )
        if len(pr_curve) > 600:
            keep = np.unique(np.linspace(0, len(pr_curve) - 1, 600).astype(int))
            pr_curve = pr_curve.iloc[keep].reset_index(drop=True)
    else:
        pr_curve = pd.DataFrame(
            columns=[
                "protocol_id",
                "dataset",
                "model_version",
                "training_scope",
                "precision",
                "recall",
                "threshold",
            ]
        )

    return record, grid, score_samples, pr_curve


def _svg_text(
    x: float,
    y: float,
    value: str,
    size: int = 13,
    weight: str = "400",
    fill: str = "#253044",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, '
        f"'PingFang SC', 'Microsoft YaHei', sans-serif\" font-size=\"{size}\" "
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{value}</text>'
    )


def _write_threshold_gain_svg(diagnostics: pd.DataFrame, output_path: str | Path) -> Path:
    binary = diagnostics[diagnostics["dataset_type"] == "binary"].copy()
    binary = binary[
        binary["model_version"].isin(["v8_semantic_main", "v8_semantic_multisource"])
    ].reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel_width = 360
    panel_height = 210
    cols = 2
    rows = int(np.ceil(len(binary) / cols))
    width = 820
    height = 120 + rows * panel_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="v8 threshold calibration gain">',
        f'<rect width="{width}" height="{height}" fill="#F6F8FB"/>',
        _svg_text(42, 52, "v8.1 阈值校准诊断", size=26, weight="800", fill="#172033"),
        _svg_text(42, 80, "对比当前阈值与每个评测集 oracle 阈值下的 Spam F1", size=14, fill="#526071"),
    ]

    for index, row in binary.iterrows():
        col = index % cols
        grid_row = index // cols
        x = 42 + col * 390
        y = 112 + grid_row * panel_height
        current = float(row["current_f1_spam"])
        oracle = float(row["oracle_f1_spam"])
        title = f"{row['dataset']} / {row['model_version'].replace('v8_semantic_', '')}"
        lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="{panel_width}" height="170" rx="6" fill="#FFFFFF" stroke="#D9DEE8"/>',
                _svg_text(x + 16, y + 28, title, size=13, weight="700"),
            ]
        )
        axis_y = y + 132
        bar_width = 62
        for bar_index, (label, value, color) in enumerate(
            [("current", current, "#2F6FED"), ("oracle", oracle, "#2E9D68")]
        ):
            bar_x = x + 92 + bar_index * 112
            bar_h = 96 * max(0.0, min(value, 1.0))
            bar_y = axis_y - bar_h
            lines.extend(
                [
                    f'<rect x="{bar_x}" y="{bar_y:.1f}" width="{bar_width}" height="{bar_h:.1f}" '
                    f'rx="4" fill="{color}"/>',
                    _svg_text(bar_x + bar_width / 2, bar_y - 8, f"{value:.3f}", size=12, weight="700", anchor="middle"),
                    _svg_text(bar_x + bar_width / 2, axis_y + 22, label, size=11, fill="#526071", anchor="middle"),
                ]
            )
        lines.append(
            f'<line x1="{x + 54}" y1="{axis_y}" x2="{x + panel_width - 36}" y2="{axis_y}" stroke="#B8C0CC"/>'
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_score_distribution_svg(score_samples: pd.DataFrame, output_path: str | Path) -> Path:
    data = score_samples[
        (score_samples["model_version"] == "v8_semantic_main")
        & (score_samples["dataset_type"] == "binary")
    ].copy()
    datasets = data["dataset"].drop_duplicates().tolist()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel_width = 360
    panel_height = 220
    cols = 2
    rows = int(np.ceil(len(datasets) / cols))
    width = 820
    height = 122 + rows * panel_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="v8 score distribution">',
        f'<rect width="{width}" height="{height}" fill="#F6F8FB"/>',
        _svg_text(42, 52, "v8.1 main-only 分数分布", size=26, weight="800", fill="#172033"),
        _svg_text(42, 80, "蓝色为正常文本，红色为垃圾文本；重叠越多，zero-shot 越难校准", size=14, fill="#526071"),
        '<rect x="630" y="38" width="14" height="14" rx="3" fill="#2F6FED"/>',
        _svg_text(650, 51, "normal", size=12, fill="#526071"),
        '<rect x="630" y="60" width="14" height="14" rx="3" fill="#E05D44"/>',
        _svg_text(650, 73, "spam", size=12, fill="#526071"),
    ]
    bins = np.linspace(0.0, 1.0, 21)
    for index, dataset in enumerate(datasets):
        subset = data[data["dataset"] == dataset]
        col = index % cols
        grid_row = index // cols
        x = 42 + col * 390
        y = 112 + grid_row * panel_height
        chart_x = x + 36
        chart_y = y + 50
        chart_w = panel_width - 66
        chart_h = 112
        axis_y = chart_y + chart_h
        lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="{panel_width}" height="180" rx="6" fill="#FFFFFF" stroke="#D9DEE8"/>',
                _svg_text(x + 16, y + 28, dataset, size=13, weight="700"),
                f'<line x1="{chart_x}" y1="{axis_y}" x2="{chart_x + chart_w}" y2="{axis_y}" stroke="#B8C0CC"/>',
            ]
        )
        max_count = 1
        histograms: dict[int, np.ndarray] = {}
        for label in (0, 1):
            values = subset.loc[subset["label"] == label, "score"].to_numpy(dtype=float)
            counts, _ = np.histogram(values, bins=bins)
            histograms[label] = counts
            max_count = max(max_count, int(counts.max()) if len(counts) else 1)
        bar_group_w = chart_w / (len(bins) - 1)
        for bin_index in range(len(bins) - 1):
            for label, color, offset in ((0, "#2F6FED", 0.0), (1, "#E05D44", 0.5)):
                count = histograms[label][bin_index]
                bar_h = chart_h * count / max_count
                bar_w = bar_group_w * 0.42
                bar_x = chart_x + bin_index * bar_group_w + offset * bar_group_w
                bar_y = axis_y - bar_h
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                    f'fill="{color}" opacity="0.78"/>'
                )
        lines.extend(
            [
                _svg_text(chart_x, axis_y + 20, "0", size=10, fill="#6B7280", anchor="middle"),
                _svg_text(chart_x + chart_w, axis_y + 20, "1", size=10, fill="#6B7280", anchor="middle"),
                _svg_text(chart_x + chart_w / 2, axis_y + 20, "score", size=10, fill="#6B7280", anchor="middle"),
            ]
        )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_calibration_markdown(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
    threshold_grid_csv: str | Path,
    score_samples_csv: str | Path,
    pr_curve_csv: str | Path,
    threshold_figure: str | Path,
    score_figure: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_cols = [
        "protocol_id",
        "dataset",
        "model_version",
        "threshold_current",
        "threshold_oracle",
        "current_f1_spam",
        "oracle_f1_spam",
        "oracle_f1_gain",
        "pr_auc",
        "diagnosis",
    ]
    table = diagnostics[display_cols].copy().map(_format_metric)
    lines = [
        "# v8.1 评测诊断与校准",
        "",
        "本实验在 v8.0 冻结语义编码器结果上增加校准诊断：比较当前阈值与每个评测集 oracle 阈值，判断问题来自阈值校准还是语义表示本身。",
        "",
        "## 诊断结论",
        "",
        "- FBS zero-shot 的 PR-AUC 较高，说明排序能力存在，主要问题是主数据阈值迁移后过于保守。",
        "- `hf_chinese_spam_10000` zero-shot 的 PR-AUC 接近随机，说明单靠阈值无法修复，需要换编码器或做监督适配。",
        "- 多来源适配后的 v8 表现明显提升，说明语义模型需要少量目标域标注来校准边界。",
        "- keyword challenge 仍是 v8.0 的短板，后续应做自动 hard-case 增强或监督微调。",
        "",
        f"![v8.1 阈值校准诊断]({Path(relpath(threshold_figure, start=output_path.parent)).as_posix()})",
        "",
        f"![v8.1 分数分布]({Path(relpath(score_figure, start=output_path.parent)).as_posix()})",
        "",
        "## 核心表",
        "",
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- `{Path(output_path).as_posix()}`：本页说明和核心诊断表",
            f"- `{Path(threshold_grid_csv).as_posix()}`：所有阈值扫描结果",
            f"- `{Path(score_samples_csv).as_posix()}`：逐样本分数、当前预测和错误类型",
            f"- `{Path(pr_curve_csv).as_posix()}`：PR 曲线采样点",
            f"- `{Path(threshold_figure).as_posix()}`：当前阈值 vs oracle 阈值 F1 对比图",
            f"- `{Path(score_figure).as_posix()}`：main-only 分数分布图",
            "",
            "## 下一步",
            "",
            "v8.2 应优先做编码器对比。对于 FBS 和 conversation 数据集，可同步尝试无目标域标签的阈值校准；对于 `hf_chinese_spam_10000`，需要优先解决语义表示分不开的问题。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_encoder_comparison_svg(results: pd.DataFrame, output_path: str | Path) -> Path:
    main_results = results[
        (results["dataset_type"] == "binary")
        & (results["model_version"] == "v8_semantic_main")
    ].copy()
    datasets = main_results["dataset"].drop_duplicates().tolist()
    encoders = main_results["encoder_name"].drop_duplicates().tolist()
    colors = ["#2F6FED", "#E05D44", "#2E9D68", "#8B5CF6", "#D97706", "#0891B2"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_width = 430
    panel_height = 220
    cols = 2
    rows = int(np.ceil(len(datasets) / cols))
    width = 980
    height = 140 + rows * panel_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="v8 encoder comparison">',
        f'<rect width="{width}" height="{height}" fill="#F6F8FB"/>',
        _svg_text(46, 54, "v8.2 Encoder 对比", size=26, weight="800", fill="#172033"),
        _svg_text(46, 82, "同一训练/评测协议下比较不同冻结语义编码器的 main-only Spam F1", size=14, fill="#526071"),
    ]
    legend_x = 650
    for index, encoder in enumerate(encoders):
        color = colors[index % len(colors)]
        y = 38 + index * 24
        lines.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" rx="3" fill="{color}"/>')
        lines.append(_svg_text(legend_x + 22, y + 12, encoder, size=12, fill="#526071"))

    for index, dataset in enumerate(datasets):
        subset = main_results[main_results["dataset"] == dataset]
        col = index % cols
        grid_row = index // cols
        x = 46 + col * 466
        y = 120 + grid_row * panel_height
        chart_x = x + 58
        chart_y = y + 52
        chart_w = panel_width - 92
        chart_h = 104
        axis_y = chart_y + chart_h
        lines.extend(
            [
                f'<rect x="{x}" y="{y}" width="{panel_width}" height="178" rx="6" fill="#FFFFFF" stroke="#D9DEE8"/>',
                _svg_text(x + 16, y + 28, dataset, size=13, weight="700"),
                f'<line x1="{chart_x}" y1="{axis_y}" x2="{chart_x + chart_w}" y2="{axis_y}" stroke="#B8C0CC"/>',
            ]
        )
        if encoders:
            bar_gap = 12
            bar_width = min(58, (chart_w - bar_gap * (len(encoders) + 1)) / max(len(encoders), 1))
        else:
            bar_gap = 12
            bar_width = 58
        for enc_index, encoder in enumerate(encoders):
            row = subset[subset["encoder_name"] == encoder]
            value = float(row["f1_spam"].iloc[0]) if len(row) else 0.0
            color = colors[enc_index % len(colors)]
            bar_h = chart_h * max(0.0, min(value, 1.0))
            bar_x = chart_x + bar_gap + enc_index * (bar_width + bar_gap)
            bar_y = axis_y - bar_h
            lines.extend(
                [
                    f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" '
                    f'rx="4" fill="{color}"/>',
                    _svg_text(bar_x + bar_width / 2, max(bar_y - 8, chart_y + 10), f"{value:.3f}", size=11, weight="700", anchor="middle"),
                    _svg_text(bar_x + bar_width / 2, axis_y + 20, encoder.split("_")[0], size=10, fill="#526071", anchor="middle"),
                ]
            )

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_encoder_comparison_markdown(
    results: pd.DataFrame,
    output_path: str | Path,
    figure_path: str | Path,
    errors: pd.DataFrame,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_rows = (
        results.sort_values(
            by=["protocol_id", "dataset", "model_version", "f1_spam", "pr_auc", "accuracy"],
            ascending=[True, True, True, False, False, False],
        )
        .groupby(["protocol_id", "dataset", "model_version"], sort=False)
        .head(1)
    )
    best_cols = [
        "protocol_id",
        "dataset",
        "model_version",
        "encoder_name",
        "accuracy",
        "precision_spam",
        "recall_spam",
        "f1_spam",
        "pr_auc",
        "false_negative",
    ]
    full_cols = [
        "protocol_id",
        "dataset",
        "model_version",
        "encoder_name",
        "training_scope",
        "threshold",
        "accuracy",
        "precision_spam",
        "recall_spam",
        "f1_spam",
        "pr_auc",
        "false_positive",
        "false_negative",
    ]
    best_table = best_rows[best_cols].copy().map(_format_metric)
    full_table = results[full_cols].copy().map(_format_metric)

    lines = [
        "# v8.2 Encoder 对比",
        "",
        "本实验固定 v8 的训练方式和 A/B/C/D 评测协议，仅替换冻结语义编码器，比较不同 encoder 对主数据集、外部 zero-shot、外部少量适配和 challenge 数据的影响。",
        "",
        f"![v8.2 Encoder 对比]({Path(relpath(figure_path, start=output_path.parent)).as_posix()})",
        "",
        "## 每个协议/数据集的最优 Encoder",
        "",
        "| " + " | ".join(best_table.columns) + " |",
        "| " + " | ".join(["---"] * len(best_table.columns)) + " |",
    ]
    for _, row in best_table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    lines.extend(
        [
            "",
            "## 完整对比结果",
            "",
            "| " + " | ".join(full_table.columns) + " |",
            "| " + " | ".join(["---"] * len(full_table.columns)) + " |",
        ]
    )
    for _, row in full_table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    if not errors.empty:
        error_table = errors.copy().map(_format_metric)
        lines.extend(
            [
                "",
                "## 加载失败的 Encoder",
                "",
                "| " + " | ".join(error_table.columns) + " |",
                "| " + " | ".join(["---"] * len(error_table.columns)) + " |",
            ]
        )
        for _, row in error_table.iterrows():
            lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    lines.extend(
        [
            "",
            "## 结论口径",
            "",
            "- Protocol A 关注主数据集最终表现。",
            "- Protocol B 关注 main-only zero-shot 泛化。",
            "- Protocol C 关注少量外部标注适配后的泛化。",
            "- Protocol D 关注 challenge 数据的鲁棒性。",
            "",
            "后续 v8.3 应基于本表选择一个默认 encoder，再做自动 hard-case 增强或监督微调。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_autoaug_delta_svg(
    results: pd.DataFrame,
    output_path: str | Path,
    aug_main_model: str = "v8_semantic_autoaug_main",
    aug_multisource_model: str = "v8_semantic_autoaug_multisource",
    title: str = "v8.3a 自动增强效果",
    subtitle: str = "对比同一 encoder 下 baseline 与自动 hard-case 增强后的 Spam F1",
) -> Path:
    pairs = [
        ("A", "main_holdout", "v8_semantic_main", aug_main_model),
        ("B", "fbs_mixed_holdout", "v8_semantic_main", aug_main_model),
        ("B", "hf_chinese_spam_10000_holdout", "v8_semantic_main", aug_main_model),
        ("B", "hf_chinese_conversation_spam_holdout", "v8_semantic_main", aug_main_model),
        ("D", "keyword_challenge", "v8_semantic_main", aug_main_model),
        ("D", "keyword_challenge", "v8_semantic_multisource", aug_multisource_model),
    ]
    model_labels = {
        aug_main_model: "main",
        aug_multisource_model: "multisource",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 920
    height = 130 + len(pairs) * 74
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="v8 auto augmentation delta">',
        f'<rect width="{width}" height="{height}" fill="#F6F8FB"/>',
        _svg_text(42, 52, title, size=26, weight="800", fill="#172033"),
        _svg_text(42, 80, subtitle, size=14, fill="#526071"),
    ]
    chart_x = 390
    chart_w = 440
    for index, (protocol, dataset, base_model, aug_model) in enumerate(pairs):
        y = 122 + index * 74
        base = results[
            (results["protocol_id"] == protocol)
            & (results["dataset"] == dataset)
            & (results["model_version"] == base_model)
        ]
        aug = results[
            (results["protocol_id"] == protocol)
            & (results["dataset"] == dataset)
            & (results["model_version"] == aug_model)
        ]
        if base.empty or aug.empty:
            continue
        base_f1 = float(base["f1_spam"].iloc[0])
        aug_f1 = float(aug["f1_spam"].iloc[0])
        delta = aug_f1 - base_f1
        label = f"{protocol} {dataset} / {model_labels.get(aug_model, aug_model)}"
        lines.append(_svg_text(42, y + 18, label, size=12, weight="700"))
        lines.append(_svg_text(42, y + 40, f"{base_f1:.3f} -> {aug_f1:.3f} ({delta:+.3f})", size=12, fill="#526071"))
        zero_x = chart_x + chart_w / 2
        lines.append(f'<line x1="{zero_x:.1f}" y1="{y}" x2="{zero_x:.1f}" y2="{y + 44}" stroke="#B8C0CC"/>')
        bar_w = min(abs(delta) / 0.5, 1.0) * (chart_w / 2 - 20)
        color = "#2E9D68" if delta >= 0 else "#E05D44"
        if delta >= 0:
            bar_x = zero_x
        else:
            bar_x = zero_x - bar_w
        lines.append(
            f'<rect x="{bar_x:.1f}" y="{y + 10}" width="{bar_w:.1f}" height="20" rx="4" fill="{color}"/>'
        )
        lines.append(_svg_text(bar_x + (bar_w if delta >= 0 else 0), y + 48, f"{delta:+.3f}", size=11, fill=color, anchor="middle"))
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _write_autoaug_markdown(
    results: pd.DataFrame,
    terms: pd.DataFrame,
    examples: pd.DataFrame,
    output_path: str | Path,
    figure_path: str | Path,
    aug_main_model: str = "v8_semantic_autoaug_main",
    aug_multisource_model: str = "v8_semantic_autoaug_multisource",
    title: str = "v8.3a 轻量自动 Hard-case 增强",
    intro: str = (
        "本实验不使用人工垃圾词表，从训练 split 自动挖掘 spam 相关短 n-gram，"
        "并基于分隔符、重复字符和训练语料中的同音字符生成 hard positive 样本。"
        "编码器和分类器仍沿用 v8：冻结语义编码器 + Logistic Regression。"
    ),
    figure_alt: str = "v8.3a 自动增强效果",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_labels = {
        aug_main_model: "main",
        aug_multisource_model: "multisource",
    }

    pair_rows: list[dict[str, object]] = []
    for protocol_id, dataset, base_model, aug_model in [
        ("A", "main_holdout", "v8_semantic_main", aug_main_model),
        ("A", "main_holdout", "v8_semantic_multisource", aug_multisource_model),
        ("B", "fbs_mixed_holdout", "v8_semantic_main", aug_main_model),
        ("B", "hf_chinese_spam_10000_holdout", "v8_semantic_main", aug_main_model),
        ("B", "hf_chinese_conversation_spam_holdout", "v8_semantic_main", aug_main_model),
        ("C", "fbs_mixed_holdout", "v8_semantic_multisource", aug_multisource_model),
        ("C", "hf_chinese_spam_10000_holdout", "v8_semantic_multisource", aug_multisource_model),
        ("C", "hf_chinese_conversation_spam_holdout", "v8_semantic_multisource", aug_multisource_model),
        ("D", "keyword_challenge", "v8_semantic_main", aug_main_model),
        ("D", "keyword_challenge", "v8_semantic_multisource", aug_multisource_model),
    ]:
        base = results[
            (results["protocol_id"] == protocol_id)
            & (results["dataset"] == dataset)
            & (results["model_version"] == base_model)
        ]
        aug = results[
            (results["protocol_id"] == protocol_id)
            & (results["dataset"] == dataset)
            & (results["model_version"] == aug_model)
        ]
        if base.empty or aug.empty:
            continue
        base_row = base.iloc[0]
        aug_row = aug.iloc[0]
        pair_rows.append(
            {
                "protocol_id": protocol_id,
                "dataset": dataset,
                "scope": model_labels.get(aug_model, aug_model),
                "baseline_f1": float(base_row["f1_spam"]),
                "autoaug_f1": float(aug_row["f1_spam"]),
                "f1_delta": float(aug_row["f1_spam"] - base_row["f1_spam"]),
                "baseline_recall": float(base_row["recall_spam"]),
                "autoaug_recall": float(aug_row["recall_spam"]),
                "fn_delta": float(aug_row["false_negative"] - base_row["false_negative"]),
            }
        )

    comparison = pd.DataFrame(pair_rows)
    comparison_table = comparison.copy().map(_format_metric)
    full_columns = [
        "protocol_id",
        "dataset",
        "model_version",
        "training_scope",
        "threshold",
        "accuracy",
        "precision_spam",
        "recall_spam",
        "f1_spam",
        "false_positive",
        "false_negative",
    ]
    full_table = results[full_columns].copy().map(_format_metric)
    term_preview = terms.head(20).copy().map(_format_metric)
    example_preview = examples.head(20).copy().map(_format_metric)

    lines = [
        f"# {title}",
        "",
        intro,
        "",
        f"![{figure_alt}]({Path(relpath(figure_path, start=output_path.parent)).as_posix()})",
        "",
        "## Baseline vs AutoAug",
        "",
        "| " + " | ".join(comparison_table.columns) + " |",
        "| " + " | ".join(["---"] * len(comparison_table.columns)) + " |",
    ]
    for _, row in comparison_table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    lines.extend(
        [
            "",
            "## 完整指标",
            "",
            "| " + " | ".join(full_table.columns) + " |",
            "| " + " | ".join(["---"] * len(full_table.columns)) + " |",
        ]
    )
    for _, row in full_table.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    lines.extend(
        [
            "",
            "## 自动挖掘片段示例",
            "",
            "| " + " | ".join(term_preview.columns) + " |",
            "| " + " | ".join(["---"] * len(term_preview.columns)) + " |",
        ]
    )
    for _, row in term_preview.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    lines.extend(
        [
            "",
            "## 自动生成样本示例",
            "",
            "| " + " | ".join(example_preview.columns) + " |",
            "| " + " | ".join(["---"] * len(example_preview.columns)) + " |",
        ]
    )
    for _, row in example_preview.iterrows():
        lines.append("| " + " | ".join(row.astype(str).tolist()) + " |")

    lines.extend(
        [
            "",
            "## 结论口径",
            "",
            "- 若 keyword challenge 提升且主数据不明显下降，说明自动 hard-case 增强有效。",
            "- 若主数据或外部数据明显退化，需要降低 `--max-augmented` 或提高 `--min-spam-df`。",
            "- 本实验仍不等价于人工 CSN 词表；所有片段来自训练 split 的统计挖掘。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _row(
    protocol_id: str,
    protocol_name: str,
    dataset: str,
    dataset_type: str,
    source_path: str,
    model_version: str,
    training_scope: str,
    description: str,
    labels: pd.Series,
    scores: np.ndarray,
    threshold: float,
    model_name: str,
) -> dict[str, object]:
    metrics = _evaluate_threshold(labels, scores, threshold)
    counts = labels.value_counts().to_dict()
    return {
        "protocol_id": protocol_id,
        "protocol_name": protocol_name,
        "dataset": dataset,
        "dataset_type": dataset_type,
        "source_path": source_path,
        "model_version": model_version,
        "training_scope": training_scope,
        "description": description,
        "encoder": model_name,
        "risk_bonus": 0.0,
        "threshold": threshold,
        "n_samples": int(len(labels)),
        "n_normal": int(counts.get(0, 0)),
        "n_spam": int(counts.get(1, 0)),
        **metrics,
    }


def compare_semantic_v8_protocols(
    train_data_path: str | Path,
    external_data_paths: Sequence[tuple[str, str | Path]],
    challenge_data_paths: Sequence[tuple[str, str | Path]] = (),
    output_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_protocol_results.csv",
    output_md: str | Path = "docs/experiments/semantic_v8/semantic_v8_protocol_results.md",
    split_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_protocol_splits.csv",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 64,
    device: str | None = None,
    test_size: float = 0.3,
    validation_size: float = 0.2,
    adapt_train_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run v8.0 frozen semantic embeddings through protocols A/B/C/D."""
    if not 0 < adapt_train_size < 1:
        raise ValueError("adapt_train_size must be between 0 and 1.")

    main_data = read_dataset(train_data_path)
    main_stratify = main_data["label"] if main_data["label"].nunique() > 1 else None
    main_train_x, main_holdout_x, main_train_y, main_holdout_y = train_test_split(
        main_data["text"],
        main_data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=main_stratify,
    )
    valid_stratify = main_train_y if main_train_y.nunique() > 1 else None
    main_fit_x, main_valid_x, main_fit_y, main_valid_y = train_test_split(
        main_train_x,
        main_train_y,
        test_size=validation_size,
        random_state=random_state,
        stratify=valid_stratify,
    )

    external_splits: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for name, path in external_data_paths:
        data = read_dataset(path)
        stratify = data["label"] if data["label"].nunique() > 1 else None
        adapt_train, holdout = train_test_split(
            data,
            train_size=adapt_train_size,
            random_state=random_state,
            stratify=stratify,
        )
        adapt_stratify = adapt_train["label"] if adapt_train["label"].nunique() > 1 else None
        adapt_fit, adapt_valid = train_test_split(
            adapt_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=adapt_stratify,
        )
        external_splits.append(
            {
                "name": name,
                "path": str(path),
                "adapt_train": adapt_train.reset_index(drop=True),
                "adapt_fit": adapt_fit.reset_index(drop=True),
                "adapt_valid": adapt_valid.reset_index(drop=True),
                "holdout": holdout.reset_index(drop=True),
            }
        )
        split_rows.append(
            {
                "dataset": name,
                "total_rows": int(len(data)),
                "adapt_train_rows": int(len(adapt_train)),
                "adapt_fit_rows": int(len(adapt_fit)),
                "adapt_valid_rows": int(len(adapt_valid)),
                "holdout_rows": int(len(holdout)),
                "holdout_spam": int((holdout["label"] == 1).sum()),
                "holdout_normal": int((holdout["label"] == 0).sum()),
            }
        )

    challenge_sets: list[dict[str, object]] = []
    for name, path in challenge_data_paths:
        data = read_dataset(path).reset_index(drop=True)
        challenge_sets.append({"name": name, "path": str(path), "data": data})

    encoder = SentenceTransformerEncoder(model_name=model_name, device=device)

    main_fit_vec = encoder.encode(main_fit_x, batch_size=batch_size)
    main_valid_vec = encoder.encode(main_valid_x, batch_size=batch_size)
    main_train_vec = encoder.encode(main_train_x, batch_size=batch_size)
    main_holdout_vec = encoder.encode(main_holdout_x, batch_size=batch_size)

    adapt_train_vecs: list[np.ndarray] = []
    adapt_fit_vecs: list[np.ndarray] = []
    adapt_valid_vecs: list[np.ndarray] = []
    for split in external_splits:
        adapt_train_vecs.append(encoder.encode(split["adapt_train"]["text"], batch_size=batch_size))
        adapt_fit_vecs.append(encoder.encode(split["adapt_fit"]["text"], batch_size=batch_size))
        adapt_valid_vecs.append(encoder.encode(split["adapt_valid"]["text"], batch_size=batch_size))
        split["holdout_vec"] = encoder.encode(split["holdout"]["text"], batch_size=batch_size)

    for split in challenge_sets:
        split["vectors"] = encoder.encode(split["data"]["text"], batch_size=batch_size)

    main_tuning = _fit_logreg(main_fit_vec, main_fit_y)
    main_threshold = _select_threshold(main_tuning, main_valid_vec, main_valid_y)
    main_model = _fit_logreg(main_train_vec, main_train_y)

    multisource_fit_vec = np.vstack([main_fit_vec, *adapt_fit_vecs])
    multisource_fit_y = pd.concat(
        [main_fit_y.reset_index(drop=True), *[split["adapt_fit"]["label"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_valid_vec = np.vstack([main_valid_vec, *adapt_valid_vecs])
    multisource_valid_y = pd.concat(
        [main_valid_y.reset_index(drop=True), *[split["adapt_valid"]["label"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_train_vec = np.vstack([main_train_vec, *adapt_train_vecs])
    multisource_train_y = pd.concat(
        [main_train_y.reset_index(drop=True), *[split["adapt_train"]["label"] for split in external_splits]],
        ignore_index=True,
    )

    multisource_tuning = _fit_logreg(multisource_fit_vec, multisource_fit_y)
    multisource_threshold = _select_threshold(
        multisource_tuning,
        multisource_valid_vec,
        multisource_valid_y,
    )
    multisource_model = _fit_logreg(multisource_train_vec, multisource_train_y)

    rows: list[dict[str, object]] = []
    main_holdout_y = main_holdout_y.reset_index(drop=True)
    rows.append(
        _row(
            "A",
            "In-domain main holdout",
            "main_holdout",
            "binary",
            str(train_data_path),
            "v8_semantic_main",
            "main_only",
            "Frozen sentence encoder + Logistic Regression",
            main_holdout_y,
            _positive_scores(main_model, main_holdout_vec),
            main_threshold,
            model_name,
        )
    )
    rows.append(
        _row(
            "A",
            "In-domain main holdout",
            "main_holdout",
            "binary",
            str(train_data_path),
            "v8_semantic_multisource",
            "main_plus_external_adapt",
            "Frozen sentence encoder + multi-source Logistic Regression",
            main_holdout_y,
            _positive_scores(multisource_model, main_holdout_vec),
            multisource_threshold,
            model_name,
        )
    )

    for split in external_splits:
        labels = split["holdout"]["label"].reset_index(drop=True)
        vectors = split["holdout_vec"]
        rows.append(
            _row(
                "B",
                "Zero-shot cross-domain",
                f"{split['name']}_holdout",
                "binary",
                split["path"],
                "v8_semantic_main",
                "main_only",
                "Frozen sentence encoder + Logistic Regression",
                labels,
                _positive_scores(main_model, vectors),
                main_threshold,
                model_name,
            )
        )
        rows.append(
            _row(
                "C",
                "Few-shot domain adaptation",
                f"{split['name']}_holdout",
                "binary",
                split["path"],
                "v8_semantic_multisource",
                "main_plus_external_adapt",
                "Frozen sentence encoder + multi-source Logistic Regression",
                labels,
                _positive_scores(multisource_model, vectors),
                multisource_threshold,
                model_name,
            )
        )

    for split in challenge_sets:
        labels = split["data"]["label"].reset_index(drop=True)
        vectors = split["vectors"]
        rows.append(
            _row(
                "D",
                "Adversarial robustness",
                split["name"],
                "spam_only",
                split["path"],
                "v8_semantic_main",
                "main_only",
                "Frozen sentence encoder + Logistic Regression",
                labels,
                _positive_scores(main_model, vectors),
                main_threshold,
                model_name,
            )
        )
        rows.append(
            _row(
                "D",
                "Adversarial robustness",
                split["name"],
                "spam_only",
                split["path"],
                "v8_semantic_multisource",
                "main_plus_external_adapt",
                "Frozen sentence encoder + multi-source Logistic Regression",
                labels,
                _positive_scores(multisource_model, vectors),
                multisource_threshold,
                model_name,
            )
        )

    results = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    split_summary = pd.DataFrame(split_rows)
    split_output = Path(split_csv)
    split_output.parent.mkdir(parents=True, exist_ok=True)
    split_summary.to_csv(split_output, index=False)

    write_evaluation_protocol_markdown(
        results,
        split_summary,
        output_md,
        title="v8 语义编码统一评测结果",
        intro=(
            "本实验把 v8.0 冻结语义编码器放入与 v0-v7 相同的 A/B/C/D "
            "统一评测协议，用于判断纯模型语义推断方案是否能提升跨来源泛化。"
        ),
    )
    return results


def compare_semantic_v8_encoders(
    train_data_path: str | Path,
    external_data_paths: Sequence[tuple[str, str | Path]],
    challenge_data_paths: Sequence[tuple[str, str | Path]] = (),
    encoders: Sequence[tuple[str, str]] = DEFAULT_ENCODERS,
    output_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_encoder_comparison.csv",
    output_md: str | Path = "docs/experiments/semantic_v8/semantic_v8_encoder_comparison.md",
    error_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_encoder_errors.csv",
    figure_path: str | Path = "docs/figures/semantic_v8/semantic_v8_encoder_comparison.svg",
    batch_size: int = 64,
    device: str | None = None,
    test_size: float = 0.3,
    validation_size: float = 0.2,
    adapt_train_size: float = 0.3,
    random_state: int = 42,
    continue_on_error: bool = True,
) -> pd.DataFrame:
    """Compare multiple frozen semantic encoders under the same v8 protocol."""
    if not encoders:
        raise ValueError("At least one encoder must be provided.")

    all_results: list[pd.DataFrame] = []
    error_rows: list[dict[str, str]] = []
    tmp_root = Path("/private/tmp/text_detection_v8_encoder_compare")
    tmp_root.mkdir(parents=True, exist_ok=True)

    for encoder_name, model_name in encoders:
        safe_encoder = _safe_name(encoder_name)
        try:
            results = compare_semantic_v8_protocols(
                train_data_path=train_data_path,
                external_data_paths=external_data_paths,
                challenge_data_paths=challenge_data_paths,
                output_csv=tmp_root / f"{safe_encoder}_results.csv",
                output_md=tmp_root / f"{safe_encoder}_results.md",
                split_csv=tmp_root / f"{safe_encoder}_splits.csv",
                model_name=model_name,
                batch_size=batch_size,
                device=device,
                test_size=test_size,
                validation_size=validation_size,
                adapt_train_size=adapt_train_size,
                random_state=random_state,
            )
            results = results.copy()
            results.insert(0, "encoder_name", encoder_name)
            results["encoder"] = model_name
            all_results.append(results)
        except Exception as exc:
            if not continue_on_error:
                raise
            error_rows.append(
                {
                    "encoder_name": encoder_name,
                    "encoder": model_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc).replace("\n", " ")[:500],
                }
            )

    errors = pd.DataFrame(error_rows, columns=["encoder_name", "encoder", "error_type", "error"])
    error_csv = Path(error_csv)
    error_csv.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(error_csv, index=False)

    if not all_results:
        raise RuntimeError(f"All encoders failed. See {error_csv}.")

    combined = pd.concat(all_results, ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    _write_encoder_comparison_svg(combined, figure_path)
    _write_encoder_comparison_markdown(combined, output_md, figure_path, errors)
    return combined


def compare_semantic_v8_auto_augmentation(
    train_data_path: str | Path,
    external_data_paths: Sequence[tuple[str, str | Path]],
    challenge_data_paths: Sequence[tuple[str, str | Path]] = (),
    output_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_results.csv",
    output_md: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_results.md",
    terms_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_terms.csv",
    examples_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_examples.csv",
    figure_path: str | Path = "docs/figures/semantic_v8/semantic_v8_autoaug_delta.svg",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 64,
    device: str | None = None,
    test_size: float = 0.3,
    validation_size: float = 0.2,
    adapt_train_size: float = 0.3,
    random_state: int = 42,
    max_terms: int = 80,
    max_augmented: int = 200,
    min_spam_df: int = 3,
    filtered: bool = False,
    max_hard_negatives: int = 200,
    positive_min_score: float = 0.05,
    positive_max_score: float = 0.75,
    hard_negative_min_score: float = 0.25,
    augmented_main_version: str = "v8_semantic_autoaug_main",
    augmented_multisource_version: str = "v8_semantic_autoaug_multisource",
    augmented_main_scope: str = "main_only_autoaug",
    augmented_multisource_scope: str = "main_plus_external_adapt_autoaug",
    augmented_description: str = "automatic hard-case positive augmentation",
    markdown_title: str = "v8.3a 轻量自动 Hard-case 增强",
    markdown_intro: str = (
        "本实验不使用人工垃圾词表，从训练 split 自动挖掘 spam 相关短 n-gram，"
        "并基于分隔符、重复字符和训练语料中的同音字符生成 hard positive 样本。"
        "编码器和分类器仍沿用 v8：冻结语义编码器 + Logistic Regression。"
    ),
    figure_title: str = "v8.3a 自动增强效果",
    figure_subtitle: str = "对比同一 encoder 下 baseline 与自动 hard-case 增强后的 Spam F1",
) -> pd.DataFrame:
    """Compare v8 baseline with automatic hard-case augmentation."""
    if not 0 < adapt_train_size < 1:
        raise ValueError("adapt_train_size must be between 0 and 1.")
    if positive_min_score > positive_max_score:
        raise ValueError("positive_min_score must be less than or equal to positive_max_score.")

    def fit_augmented(
        texts: pd.Series,
        labels: pd.Series,
        vectors: np.ndarray,
        scope: str,
    ) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame]:
        if filtered:
            return _fit_filtered_augmented_classifier(
                encoder=encoder,
                texts=texts,
                labels=labels,
                vectors=vectors,
                scope=scope,
                batch_size=batch_size,
                max_terms=max_terms,
                max_augmented=max_augmented,
                min_spam_df=min_spam_df,
                max_hard_negatives=max_hard_negatives,
                positive_min_score=positive_min_score,
                positive_max_score=positive_max_score,
                hard_negative_min_score=hard_negative_min_score,
            )
        return _fit_augmented_classifier(
            encoder=encoder,
            texts=texts,
            labels=labels,
            vectors=vectors,
            scope=scope,
            batch_size=batch_size,
            max_terms=max_terms,
            max_augmented=max_augmented,
            min_spam_df=min_spam_df,
        )

    main_data = read_dataset(train_data_path)
    main_stratify = main_data["label"] if main_data["label"].nunique() > 1 else None
    main_train_x, main_holdout_x, main_train_y, main_holdout_y = train_test_split(
        main_data["text"],
        main_data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=main_stratify,
    )
    valid_stratify = main_train_y if main_train_y.nunique() > 1 else None
    main_fit_x, main_valid_x, main_fit_y, main_valid_y = train_test_split(
        main_train_x,
        main_train_y,
        test_size=validation_size,
        random_state=random_state,
        stratify=valid_stratify,
    )

    external_splits: list[dict[str, object]] = []
    for name, path in external_data_paths:
        data = read_dataset(path)
        stratify = data["label"] if data["label"].nunique() > 1 else None
        adapt_train, holdout = train_test_split(
            data,
            train_size=adapt_train_size,
            random_state=random_state,
            stratify=stratify,
        )
        adapt_stratify = adapt_train["label"] if adapt_train["label"].nunique() > 1 else None
        adapt_fit, adapt_valid = train_test_split(
            adapt_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=adapt_stratify,
        )
        external_splits.append(
            {
                "name": name,
                "path": str(path),
                "adapt_train": adapt_train.reset_index(drop=True),
                "adapt_fit": adapt_fit.reset_index(drop=True),
                "adapt_valid": adapt_valid.reset_index(drop=True),
                "holdout": holdout.reset_index(drop=True),
            }
        )

    challenge_sets: list[dict[str, object]] = []
    for name, path in challenge_data_paths:
        data = read_dataset(path).reset_index(drop=True)
        challenge_sets.append({"name": name, "path": str(path), "data": data})

    encoder = SentenceTransformerEncoder(model_name=model_name, device=device)

    main_fit_x = main_fit_x.reset_index(drop=True)
    main_fit_y = main_fit_y.reset_index(drop=True)
    main_valid_x = main_valid_x.reset_index(drop=True)
    main_valid_y = main_valid_y.reset_index(drop=True)
    main_train_x = main_train_x.reset_index(drop=True)
    main_train_y = main_train_y.reset_index(drop=True)
    main_holdout_x = main_holdout_x.reset_index(drop=True)
    main_holdout_y = main_holdout_y.reset_index(drop=True)

    main_fit_vec = encoder.encode(main_fit_x, batch_size=batch_size)
    main_valid_vec = encoder.encode(main_valid_x, batch_size=batch_size)
    main_train_vec = encoder.encode(main_train_x, batch_size=batch_size)
    main_holdout_vec = encoder.encode(main_holdout_x, batch_size=batch_size)

    adapt_train_vecs: list[np.ndarray] = []
    adapt_fit_vecs: list[np.ndarray] = []
    adapt_valid_vecs: list[np.ndarray] = []
    for split in external_splits:
        adapt_train_vecs.append(encoder.encode(split["adapt_train"]["text"], batch_size=batch_size))
        adapt_fit_vecs.append(encoder.encode(split["adapt_fit"]["text"], batch_size=batch_size))
        adapt_valid_vecs.append(encoder.encode(split["adapt_valid"]["text"], batch_size=batch_size))
        split["holdout_vec"] = encoder.encode(split["holdout"]["text"], batch_size=batch_size)

    for split in challenge_sets:
        split["vectors"] = encoder.encode(split["data"]["text"], batch_size=batch_size)

    main_tuning = _fit_logreg(main_fit_vec, main_fit_y)
    main_threshold = _select_threshold(main_tuning, main_valid_vec, main_valid_y)
    main_model = _fit_logreg(main_train_vec, main_train_y)

    main_auto_tuning, main_fit_terms, main_fit_examples = fit_augmented(
        main_fit_x,
        main_fit_y,
        main_fit_vec,
        "main_fit",
    )
    main_auto_threshold = _select_threshold(main_auto_tuning, main_valid_vec, main_valid_y)
    main_auto_model, main_train_terms, main_train_examples = fit_augmented(
        main_train_x,
        main_train_y,
        main_train_vec,
        "main_train",
    )

    multisource_fit_x = pd.concat(
        [main_fit_x, *[split["adapt_fit"]["text"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_fit_y = pd.concat(
        [main_fit_y, *[split["adapt_fit"]["label"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_valid_x = pd.concat(
        [main_valid_x, *[split["adapt_valid"]["text"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_valid_y = pd.concat(
        [main_valid_y, *[split["adapt_valid"]["label"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_train_x = pd.concat(
        [main_train_x, *[split["adapt_train"]["text"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_train_y = pd.concat(
        [main_train_y, *[split["adapt_train"]["label"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_fit_vec = np.vstack([main_fit_vec, *adapt_fit_vecs])
    multisource_valid_vec = np.vstack([main_valid_vec, *adapt_valid_vecs])
    multisource_train_vec = np.vstack([main_train_vec, *adapt_train_vecs])

    multisource_tuning = _fit_logreg(multisource_fit_vec, multisource_fit_y)
    multisource_threshold = _select_threshold(multisource_tuning, multisource_valid_vec, multisource_valid_y)
    multisource_model = _fit_logreg(multisource_train_vec, multisource_train_y)

    multisource_auto_tuning, multisource_fit_terms, multisource_fit_examples = fit_augmented(
        multisource_fit_x,
        multisource_fit_y,
        multisource_fit_vec,
        "multisource_fit",
    )
    multisource_auto_threshold = _select_threshold(
        multisource_auto_tuning,
        multisource_valid_vec,
        multisource_valid_y,
    )
    multisource_auto_model, multisource_train_terms, multisource_train_examples = fit_augmented(
        multisource_train_x,
        multisource_train_y,
        multisource_train_vec,
        "multisource_train",
    )

    rows: list[dict[str, object]] = []

    def add_row(
        protocol_id: str,
        protocol_name: str,
        dataset: str,
        dataset_type: str,
        source_path: str,
        model_version: str,
        training_scope: str,
        description: str,
        labels: pd.Series,
        scores: np.ndarray,
        threshold: float,
    ) -> None:
        rows.append(
            _row(
                protocol_id,
                protocol_name,
                dataset,
                dataset_type,
                source_path,
                model_version,
                training_scope,
                description,
                labels,
                scores,
                threshold,
                model_name,
            )
        )

    main_models = (
        (
            "v8_semantic_main",
            "main_only",
            "Frozen sentence encoder + Logistic Regression",
            main_model,
            main_threshold,
        ),
        (
            augmented_main_version,
            augmented_main_scope,
            f"v8 main + {augmented_description}",
            main_auto_model,
            main_auto_threshold,
        ),
    )
    multisource_models = (
        (
            "v8_semantic_multisource",
            "main_plus_external_adapt",
            "Frozen sentence encoder + multi-source Logistic Regression",
            multisource_model,
            multisource_threshold,
        ),
        (
            augmented_multisource_version,
            augmented_multisource_scope,
            f"v8 multisource + {augmented_description}",
            multisource_auto_model,
            multisource_auto_threshold,
        ),
    )

    for version, scope, description, model, threshold in (*main_models, *multisource_models):
        add_row(
            "A",
            "In-domain main holdout",
            "main_holdout",
            "binary",
            str(train_data_path),
            version,
            scope,
            description,
            main_holdout_y,
            _positive_scores(model, main_holdout_vec),
            threshold,
        )

    for split in external_splits:
        labels = split["holdout"]["label"].reset_index(drop=True)
        vectors = split["holdout_vec"]
        for version, scope, description, model, threshold in main_models:
            add_row(
                "B",
                "Zero-shot cross-domain",
                f"{split['name']}_holdout",
                "binary",
                split["path"],
                version,
                scope,
                description,
                labels,
                _positive_scores(model, vectors),
                threshold,
            )
        for version, scope, description, model, threshold in multisource_models:
            add_row(
                "C",
                "Few-shot domain adaptation",
                f"{split['name']}_holdout",
                "binary",
                split["path"],
                version,
                scope,
                description,
                labels,
                _positive_scores(model, vectors),
                threshold,
            )

    for split in challenge_sets:
        labels = split["data"]["label"].reset_index(drop=True)
        vectors = split["vectors"]
        for version, scope, description, model, threshold in (*main_models, *multisource_models):
            add_row(
                "D",
                "Adversarial robustness",
                split["name"],
                "spam_only",
                split["path"],
                version,
                scope,
                description,
                labels,
                _positive_scores(model, vectors),
                threshold,
            )

    results = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    term_frames = [main_fit_terms, main_train_terms, multisource_fit_terms, multisource_train_terms]
    terms = pd.concat(term_frames, ignore_index=True) if term_frames else pd.DataFrame()
    terms_csv = Path(terms_csv)
    terms_csv.parent.mkdir(parents=True, exist_ok=True)
    terms.to_csv(terms_csv, index=False)

    example_frames = [main_fit_examples, main_train_examples, multisource_fit_examples, multisource_train_examples]
    examples = pd.concat(example_frames, ignore_index=True) if example_frames else pd.DataFrame()
    examples_csv = Path(examples_csv)
    examples_csv.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(examples_csv, index=False)

    _write_autoaug_delta_svg(
        results,
        figure_path,
        aug_main_model=augmented_main_version,
        aug_multisource_model=augmented_multisource_version,
        title=figure_title,
        subtitle=figure_subtitle,
    )
    _write_autoaug_markdown(
        results,
        terms,
        examples,
        output_md,
        figure_path,
        aug_main_model=augmented_main_version,
        aug_multisource_model=augmented_multisource_version,
        title=markdown_title,
        intro=markdown_intro,
        figure_alt=figure_title,
    )
    return results


def compare_semantic_v8_filtered_auto_augmentation(
    train_data_path: str | Path,
    external_data_paths: Sequence[tuple[str, str | Path]],
    challenge_data_paths: Sequence[tuple[str, str | Path]] = (),
    output_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_results.csv",
    output_md: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_results.md",
    terms_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_terms.csv",
    examples_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_examples.csv",
    figure_path: str | Path = "docs/figures/semantic_v8/semantic_v8_autoaug_filtered_delta.svg",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 64,
    device: str | None = None,
    test_size: float = 0.3,
    validation_size: float = 0.2,
    adapt_train_size: float = 0.3,
    random_state: int = 42,
    max_terms: int = 80,
    max_augmented: int = 200,
    min_spam_df: int = 3,
    max_hard_negatives: int = 200,
    positive_min_score: float = 0.05,
    positive_max_score: float = 0.75,
    hard_negative_min_score: float = 0.25,
) -> pd.DataFrame:
    """Compare v8.3b filtered augmentation and hard negatives against semantic baselines."""
    return compare_semantic_v8_auto_augmentation(
        train_data_path=train_data_path,
        external_data_paths=external_data_paths,
        challenge_data_paths=challenge_data_paths,
        output_csv=output_csv,
        output_md=output_md,
        terms_csv=terms_csv,
        examples_csv=examples_csv,
        figure_path=figure_path,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        test_size=test_size,
        validation_size=validation_size,
        adapt_train_size=adapt_train_size,
        random_state=random_state,
        max_terms=max_terms,
        max_augmented=max_augmented,
        min_spam_df=min_spam_df,
        filtered=True,
        max_hard_negatives=max_hard_negatives,
        positive_min_score=positive_min_score,
        positive_max_score=positive_max_score,
        hard_negative_min_score=hard_negative_min_score,
        augmented_main_version="v8_semantic_autoaug_filtered_main",
        augmented_multisource_version="v8_semantic_autoaug_filtered_multisource",
        augmented_main_scope="main_only_autoaug_filtered",
        augmented_multisource_scope="main_plus_external_adapt_autoaug_filtered",
        augmented_description="filtered automatic hard-case augmentation + hard negatives",
        markdown_title="v8.3b 增强样本筛选 + Hard Negative",
        markdown_intro=(
            "本实验保留 v8.3a 的自动 hard positive 生成，但用原始语义模型分数过滤过易或过激进的增强样本，"
            "同时把高风险正常样本和其自动变体作为 hard negative 加回训练。目标是在保持 keyword challenge "
            "收益的同时，减少 HF conversation 等跨域正常文本被误判为 spam 的副作用。当前正式默认参数为 "
            "`positive_min_score=0.05`、`positive_max_score=0.75`、`max_hard_negatives=200`。"
        ),
        figure_title="v8.3b 筛选增强 + Hard Negative",
        figure_subtitle="对比同一 encoder 下 baseline 与 v8.3b 后的 Spam F1",
    )


def diagnose_semantic_v8_calibration(
    train_data_path: str | Path,
    external_data_paths: Sequence[tuple[str, str | Path]],
    challenge_data_paths: Sequence[tuple[str, str | Path]] = (),
    diagnostics_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_calibration_diagnostics.csv",
    threshold_grid_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_threshold_grid.csv",
    score_samples_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_score_samples.csv",
    pr_curve_csv: str | Path = "docs/experiments/semantic_v8/semantic_v8_pr_curve.csv",
    output_md: str | Path = "docs/experiments/semantic_v8/semantic_v8_calibration_diagnostics.md",
    threshold_figure: str | Path = "docs/figures/semantic_v8/semantic_v8_threshold_gain.svg",
    score_figure: str | Path = "docs/figures/semantic_v8/semantic_v8_score_distribution.svg",
    model_name: str = "BAAI/bge-small-zh-v1.5",
    batch_size: int = 64,
    device: str | None = None,
    test_size: float = 0.3,
    validation_size: float = 0.2,
    adapt_train_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run v8.1 threshold and score-distribution diagnostics."""
    if not 0 < adapt_train_size < 1:
        raise ValueError("adapt_train_size must be between 0 and 1.")

    main_data = read_dataset(train_data_path)
    main_stratify = main_data["label"] if main_data["label"].nunique() > 1 else None
    main_train_x, main_holdout_x, main_train_y, main_holdout_y = train_test_split(
        main_data["text"],
        main_data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=main_stratify,
    )
    valid_stratify = main_train_y if main_train_y.nunique() > 1 else None
    main_fit_x, main_valid_x, main_fit_y, main_valid_y = train_test_split(
        main_train_x,
        main_train_y,
        test_size=validation_size,
        random_state=random_state,
        stratify=valid_stratify,
    )

    external_splits: list[dict[str, object]] = []
    for name, path in external_data_paths:
        data = read_dataset(path)
        stratify = data["label"] if data["label"].nunique() > 1 else None
        adapt_train, holdout = train_test_split(
            data,
            train_size=adapt_train_size,
            random_state=random_state,
            stratify=stratify,
        )
        adapt_stratify = adapt_train["label"] if adapt_train["label"].nunique() > 1 else None
        adapt_fit, adapt_valid = train_test_split(
            adapt_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=adapt_stratify,
        )
        external_splits.append(
            {
                "name": name,
                "path": str(path),
                "adapt_train": adapt_train.reset_index(drop=True),
                "adapt_fit": adapt_fit.reset_index(drop=True),
                "adapt_valid": adapt_valid.reset_index(drop=True),
                "holdout": holdout.reset_index(drop=True),
            }
        )

    challenge_sets: list[dict[str, object]] = []
    for name, path in challenge_data_paths:
        data = read_dataset(path).reset_index(drop=True)
        challenge_sets.append({"name": name, "path": str(path), "data": data})

    encoder = SentenceTransformerEncoder(model_name=model_name, device=device)

    main_fit_vec = encoder.encode(main_fit_x, batch_size=batch_size)
    main_valid_vec = encoder.encode(main_valid_x, batch_size=batch_size)
    main_train_vec = encoder.encode(main_train_x, batch_size=batch_size)
    main_holdout_vec = encoder.encode(main_holdout_x, batch_size=batch_size)

    adapt_train_vecs: list[np.ndarray] = []
    adapt_fit_vecs: list[np.ndarray] = []
    adapt_valid_vecs: list[np.ndarray] = []
    for split in external_splits:
        adapt_train_vecs.append(encoder.encode(split["adapt_train"]["text"], batch_size=batch_size))
        adapt_fit_vecs.append(encoder.encode(split["adapt_fit"]["text"], batch_size=batch_size))
        adapt_valid_vecs.append(encoder.encode(split["adapt_valid"]["text"], batch_size=batch_size))
        split["holdout_vec"] = encoder.encode(split["holdout"]["text"], batch_size=batch_size)

    for split in challenge_sets:
        split["vectors"] = encoder.encode(split["data"]["text"], batch_size=batch_size)

    main_tuning = _fit_logreg(main_fit_vec, main_fit_y)
    main_threshold = _select_threshold(main_tuning, main_valid_vec, main_valid_y)
    main_model = _fit_logreg(main_train_vec, main_train_y)

    multisource_fit_vec = np.vstack([main_fit_vec, *adapt_fit_vecs])
    multisource_fit_y = pd.concat(
        [main_fit_y.reset_index(drop=True), *[split["adapt_fit"]["label"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_valid_vec = np.vstack([main_valid_vec, *adapt_valid_vecs])
    multisource_valid_y = pd.concat(
        [main_valid_y.reset_index(drop=True), *[split["adapt_valid"]["label"] for split in external_splits]],
        ignore_index=True,
    )
    multisource_train_vec = np.vstack([main_train_vec, *adapt_train_vecs])
    multisource_train_y = pd.concat(
        [main_train_y.reset_index(drop=True), *[split["adapt_train"]["label"] for split in external_splits]],
        ignore_index=True,
    )

    multisource_tuning = _fit_logreg(multisource_fit_vec, multisource_fit_y)
    multisource_threshold = _select_threshold(
        multisource_tuning,
        multisource_valid_vec,
        multisource_valid_y,
    )
    multisource_model = _fit_logreg(multisource_train_vec, multisource_train_y)

    diagnostics: list[dict[str, object]] = []
    threshold_grids: list[pd.DataFrame] = []
    score_samples: list[pd.DataFrame] = []
    pr_curves: list[pd.DataFrame] = []

    def add_record(
        protocol_id: str,
        protocol_name: str,
        dataset: str,
        dataset_type: str,
        source_path: str,
        model_version: str,
        training_scope: str,
        labels: pd.Series,
        scores: np.ndarray,
        threshold: float,
    ) -> None:
        record, grid, samples, curve = _evaluation_record(
            protocol_id,
            protocol_name,
            dataset,
            dataset_type,
            source_path,
            model_version,
            training_scope,
            labels,
            scores,
            threshold,
        )
        record["encoder"] = model_name
        diagnostics.append(record)
        threshold_grids.append(grid)
        score_samples.append(samples)
        pr_curves.append(curve)

    main_holdout_y = main_holdout_y.reset_index(drop=True)
    add_record(
        "A",
        "In-domain main holdout",
        "main_holdout",
        "binary",
        str(train_data_path),
        "v8_semantic_main",
        "main_only",
        main_holdout_y,
        _positive_scores(main_model, main_holdout_vec),
        main_threshold,
    )
    add_record(
        "A",
        "In-domain main holdout",
        "main_holdout",
        "binary",
        str(train_data_path),
        "v8_semantic_multisource",
        "main_plus_external_adapt",
        main_holdout_y,
        _positive_scores(multisource_model, main_holdout_vec),
        multisource_threshold,
    )

    for split in external_splits:
        labels = split["holdout"]["label"].reset_index(drop=True)
        vectors = split["holdout_vec"]
        add_record(
            "B",
            "Zero-shot cross-domain",
            f"{split['name']}_holdout",
            "binary",
            split["path"],
            "v8_semantic_main",
            "main_only",
            labels,
            _positive_scores(main_model, vectors),
            main_threshold,
        )
        add_record(
            "C",
            "Few-shot domain adaptation",
            f"{split['name']}_holdout",
            "binary",
            split["path"],
            "v8_semantic_multisource",
            "main_plus_external_adapt",
            labels,
            _positive_scores(multisource_model, vectors),
            multisource_threshold,
        )

    for split in challenge_sets:
        labels = split["data"]["label"].reset_index(drop=True)
        vectors = split["vectors"]
        add_record(
            "D",
            "Adversarial robustness",
            split["name"],
            "spam_only",
            split["path"],
            "v8_semantic_main",
            "main_only",
            labels,
            _positive_scores(main_model, vectors),
            main_threshold,
        )
        add_record(
            "D",
            "Adversarial robustness",
            split["name"],
            "spam_only",
            split["path"],
            "v8_semantic_multisource",
            "main_plus_external_adapt",
            labels,
            _positive_scores(multisource_model, vectors),
            multisource_threshold,
        )

    diagnostics_df = pd.DataFrame(diagnostics)
    threshold_grid_df = pd.concat(threshold_grids, ignore_index=True)
    score_samples_df = pd.concat(score_samples, ignore_index=True)
    pr_curve_df = pd.concat(pr_curves, ignore_index=True)

    diagnostics_csv = Path(diagnostics_csv)
    diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_df.to_csv(diagnostics_csv, index=False)

    threshold_grid_csv = Path(threshold_grid_csv)
    threshold_grid_csv.parent.mkdir(parents=True, exist_ok=True)
    threshold_grid_df.to_csv(threshold_grid_csv, index=False)

    score_samples_csv = Path(score_samples_csv)
    score_samples_csv.parent.mkdir(parents=True, exist_ok=True)
    score_samples_df.to_csv(score_samples_csv, index=False)

    pr_curve_csv = Path(pr_curve_csv)
    pr_curve_csv.parent.mkdir(parents=True, exist_ok=True)
    pr_curve_df.to_csv(pr_curve_csv, index=False)

    _write_threshold_gain_svg(diagnostics_df, threshold_figure)
    _write_score_distribution_svg(score_samples_df, score_figure)
    _write_calibration_markdown(
        diagnostics_df,
        output_md,
        threshold_grid_csv,
        score_samples_csv,
        pr_curve_csv,
        threshold_figure,
        score_figure,
    )
    return diagnostics_df
