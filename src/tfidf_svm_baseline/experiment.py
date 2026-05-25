"""TF-IDF + Linear SVM baseline experiment."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


KEEP_PATTERN = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]")
SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    value = "" if text is None else str(text)
    value = KEEP_PATTERN.sub(" ", value.strip().lower())
    return SPACE_PATTERN.sub(" ", value).strip()


def char_tokenize(text: str) -> list[str]:
    text = clean_text(text)
    tokens: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            tokens.append("".join(buffer))
            buffer.clear()

    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            flush_buffer()
            tokens.append(char)
        elif char.isspace():
            flush_buffer()
        else:
            buffer.append(char)
    flush_buffer()
    return tokens


def preprocess(text: str) -> str:
    return " ".join(char_tokenize(text))


def normalize_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(subset=["label", "text"]).copy()
    data["label"] = data["label"].astype(int)
    data["text"] = data["text"].astype(str)
    return data[["label", "text"]]


def read_loose_tsv(path: str | Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if line_number == 0 and parts[0].strip().lower() == "label":
                continue
            text = "\t".join(parts[1:-1]) if len(parts) >= 3 else parts[1]
            records.append({"label": parts[0], "text": text})
    return normalize_dataset(pd.DataFrame(records))


def read_dataset(path: str | Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(path, sep="\t")
    except pd.errors.ParserError:
        return read_loose_tsv(path)
    if "label" not in data.columns or "text" not in data.columns:
        return read_loose_tsv(path)
    return normalize_dataset(data)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess,
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


def score_texts(model: Pipeline, texts: pd.Series) -> list[float]:
    return model.decision_function(texts)


def evaluate_predictions(y_true, y_pred, y_score) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
    }


def run_experiment(
    data_path: str | Path,
    output_csv: str | Path | None = None,
    model_path: str | Path | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    data = read_dataset(data_path)
    train_x, test_x, train_y, test_y = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )

    model = build_model()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    score_y = score_texts(model, test_x)
    counts = test_y.value_counts().to_dict()
    metrics = evaluate_predictions(test_y, pred_y, score_y)
    result = pd.DataFrame(
        [
            {
                "model": "TF-IDF+SVM",
                "description": "Character TF-IDF(1-3gram) + Linear SVM",
                "n_samples": int(len(test_y)),
                "n_normal": int(counts.get(0, 0)),
                "n_spam": int(counts.get(1, 0)),
                **metrics,
            }
        ]
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF+SVM baseline.")
    parser.add_argument("--data", default="data/processed/ast_dataset.tsv")
    parser.add_argument("--out-csv")
    parser.add_argument("--model")
    args = parser.parse_args()
    result = run_experiment(args.data, output_csv=args.out_csv, model_path=args.model)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
