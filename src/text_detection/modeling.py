"""Training and inference helpers for spam text detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import char_tokenize, join_tokens, word_tokenize


@dataclass(frozen=True)
class TrainResult:
    report: str
    confusion: list[list[int]]
    model_path: Path


def preprocess_char_text(text: str) -> str:
    return join_tokens(char_tokenize(text))


def preprocess_word_text(text: str) -> str:
    return join_tokens(word_tokenize(text))


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


def build_pipeline(analyzer: str = "char") -> Pipeline:
    if analyzer == "word":
        preprocessor = preprocess_word_text
        ngram_range = (1, 2)
    elif analyzer == "char":
        preprocessor = preprocess_char_text
        ngram_range = (1, 3)
    else:
        raise ValueError("analyzer must be 'char' or 'word'")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocessor,
                    tokenizer=str.split,
                    token_pattern=None,
                    ngram_range=ngram_range,
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline(
    data_path: str | Path,
    model_path: str | Path = "models/baseline.joblib",
    analyzer: str = "char",
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

    pipeline = build_pipeline(analyzer=analyzer)
    pipeline.fit(train_x, train_y)

    pred_y = pipeline.predict(test_x)
    report = classification_report(test_y, pred_y, digits=4, zero_division=0)
    confusion = confusion_matrix(test_y, pred_y).tolist()

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return TrainResult(report=report, confusion=confusion, model_path=model_path)


def load_model(model_path: str | Path = "models/baseline.joblib") -> Pipeline:
    return joblib.load(model_path)


def predict_text(text: str, model_path: str | Path = "models/baseline.joblib") -> int:
    model = load_model(model_path)
    return int(model.predict([text])[0])
