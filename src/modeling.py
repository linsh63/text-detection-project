"""Training and inference helpers for the v1 spam text detector."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import char_tokenize, join_tokens


@dataclass(frozen=True)
class TrainResult:
    report: str
    confusion: list[list[int]]
    metrics: dict[str, float]
    model_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    report: str
    confusion: list[list[int]]
    metrics: dict[str, float]


def preprocess_char_text(text: str) -> str:
    return join_tokens(char_tokenize(text))


def read_dataset(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t")
    required = {"label", "text"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data.dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].astype(int)
    data["text"] = data["text"].astype(str)
    return data


def build_v1_pipeline() -> Pipeline:
    """Build v1: character TF-IDF(1-3gram) + Linear SVM."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_char_text,
                    tokenizer=str.split,
                    token_pattern=None,
                    ngram_range=(1, 3),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def score_texts(model: Pipeline, texts) -> list[float] | None:
    """Return positive-class scores for classifiers that expose scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(texts)
    return None


def _recall_at_precision(y_true, y_score, threshold: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    valid = recall[precision >= threshold]
    if len(valid) == 0:
        return 0.0
    return float(valid.max())


def evaluate_predictions(y_true, y_pred, y_score=None) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
    }

    if y_score is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["recall_at_precision_90"] = _recall_at_precision(y_true, y_score, 0.90)
        metrics["recall_at_precision_95"] = _recall_at_precision(y_true, y_score, 0.95)

    return metrics


def train_baseline(
    data_path: str | Path,
    model_path: str | Path = "models/v1_char_svm.joblib",
) -> TrainResult:
    data = read_dataset(data_path)

    stratify = data["label"] if data["label"].nunique() > 1 else None
    train_x, test_x, train_y, test_y = train_test_split(
        data["text"],
        data["label"],
        test_size=0.3,
        random_state=42,
        stratify=stratify,
    )

    pipeline = build_v1_pipeline()
    pipeline.fit(train_x, train_y)

    pred_y = pipeline.predict(test_x)
    score_y = score_texts(pipeline, test_x)
    report = classification_report(test_y, pred_y, digits=4, zero_division=0)
    confusion = confusion_matrix(test_y, pred_y).tolist()
    metrics = evaluate_predictions(test_y, pred_y, score_y)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return TrainResult(
        report=report,
        confusion=confusion,
        metrics=metrics,
        model_path=model_path,
    )


def load_model(model_path: str | Path = "models/v1_char_svm.joblib") -> Pipeline:
    return joblib.load(model_path)


def predict_text(text: str, model_path: str | Path = "models/v1_char_svm.joblib") -> int:
    model = load_model(model_path)
    return int(model.predict([text])[0])


def evaluate_model(
    data_path: str | Path,
    model_path: str | Path = "models/v1_char_svm.joblib",
) -> EvaluationResult:
    data = read_dataset(data_path)
    model = load_model(model_path)
    pred_y = model.predict(data["text"])
    score_y = score_texts(model, data["text"])
    report = classification_report(data["label"], pred_y, digits=4, zero_division=0)
    confusion = confusion_matrix(data["label"], pred_y).tolist()
    metrics = evaluate_predictions(data["label"], pred_y, score_y)
    return EvaluationResult(report=report, confusion=confusion, metrics=metrics)
