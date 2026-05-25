"""Course-style rule-free character similarity network experiment.

The pipeline follows the course slides more closely than a TF-IDF feature
union:

1. build generic glyph/phonetic character codes;
2. construct a sparse character similarity network;
3. learn corpus-based character embeddings;
4. aggregate similar characters into CSN character embeddings;
5. generate sentence embeddings;
6. train a Logistic Regression classifier.

No sensitive-word list, variant dictionary, or handwritten spam pattern is used.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pypinyin import Style, lazy_pinyin, pinyin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


KEEP_PATTERN = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]")
SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    value = "" if text is None else str(text)
    value = KEEP_PATTERN.sub(" ", value.strip().lower())
    return SPACE_PATTERN.sub(" ", value).strip()


def char_tokenize(text: str) -> list[str]:
    text = clean_text(text)
    return [char for char in text if not char.isspace()]


def is_chinese_char(token: str) -> bool:
    return len(token) == 1 and "\u4e00" <= token <= "\u9fff"


@lru_cache(maxsize=8192)
def pinyin_part(char: str, style: Style) -> str:
    result = pinyin(char, style=style, strict=False, errors="ignore")
    return result[0][0] if result and result[0] else ""


@lru_cache(maxsize=8192)
def pinyin_full(char: str) -> str:
    result = lazy_pinyin(char, style=Style.NORMAL, errors="ignore")
    return result[0] if result else ""


def character_code_features(token: str) -> dict[str, float]:
    """Encode a character without manual sensitive-word or variant tables."""
    if is_chinese_char(token):
        features: dict[str, float] = {
            "kind=chinese": 1.0,
            f"unicode_bucket={ord(token) // 32}": 0.6,
            f"unicode_block={ord(token) // 256}": 0.4,
        }
        full = pinyin_full(token)
        initial = pinyin_part(token, Style.INITIALS)
        final = pinyin_part(token, Style.FINALS)
        tone = pinyin_part(token, Style.TONE3)
        if full:
            features[f"pinyin={full}"] = 1.0
        if initial:
            features[f"initial={initial}"] = 0.8
        if final:
            features[f"final={final}"] = 0.8
        if tone and tone[-1].isdigit():
            features[f"tone={tone[-1]}"] = 0.3
        return features

    if token.isdigit():
        return {"kind=digit": 1.0, f"digit={token}": 0.7}
    if token.isalpha():
        return {"kind=latin": 1.0, f"latin={token}": 0.7}
    return {"kind=other": 1.0, f"char={token}": 0.7}


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


def texts_to_token_strings(texts: pd.Series | list[str]) -> list[str]:
    return [" ".join(char_tokenize(text)) for text in texts]


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


@dataclass
class RuleFreeCSNConfig:
    embedding_dim: int = 64
    similarity_threshold: float = 0.52
    max_neighbors: int = 24
    max_chars: int = 120
    random_state: int = 42


class CourseStyleCSNClassifier:
    """Course-style CSN sentence embedding plus Logistic Regression."""

    def __init__(self, config: RuleFreeCSNConfig | None = None) -> None:
        self.config = config or RuleFreeCSNConfig()
        self.char_vectorizer = TfidfVectorizer(
            preprocessor=None,
            tokenizer=str.split,
            token_pattern=None,
            ngram_range=(1, 1),
            min_df=1,
        )
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.config.random_state,
        )

    def fit(self, texts: pd.Series, labels: pd.Series) -> "CourseStyleCSNClassifier":
        token_strings = texts_to_token_strings(texts)
        doc_char_matrix = self.char_vectorizer.fit_transform(token_strings)
        self.characters_ = np.array(self.char_vectorizer.get_feature_names_out())
        self.char_to_index_ = {char: index for index, char in enumerate(self.characters_)}
        self.char_counts_ = np.asarray(doc_char_matrix.sum(axis=0)).ravel()

        self.base_char_vectors_ = self._learn_base_char_vectors(doc_char_matrix)
        self.csn_char_vectors_ = self._build_csn_vectors()
        sentence_vectors = self.transform(texts)
        self.classifier.fit(sentence_vectors, labels)
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        return self.classifier.predict(self.transform(texts))

    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(texts))

    def transform(self, texts: pd.Series | list[str]) -> np.ndarray:
        vectors = [self._sentence_vector(text) for text in texts]
        return np.vstack(vectors)

    def _learn_base_char_vectors(self, doc_char_matrix) -> np.ndarray:
        char_doc_matrix = doc_char_matrix.T
        dim = min(
            self.config.embedding_dim,
            max(2, char_doc_matrix.shape[0] - 1),
            max(2, char_doc_matrix.shape[1] - 1),
        )
        svd = TruncatedSVD(n_components=dim, random_state=self.config.random_state)
        vectors = svd.fit_transform(char_doc_matrix)
        return normalize(vectors).astype(np.float32)

    def _build_csn_vectors(self) -> np.ndarray:
        code_vectors = self._build_code_vectors()
        n_neighbors = min(self.config.max_neighbors, len(self.characters_))
        nearest = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        nearest.fit(code_vectors)
        distances, indices = nearest.kneighbors(code_vectors)

        csn_vectors = np.zeros_like(self.base_char_vectors_, dtype=np.float32)
        for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            similarities = 1.0 - row_distances
            keep = similarities >= self.config.similarity_threshold
            if not np.any(keep):
                keep = row_indices == row_index
            neighbor_indices = row_indices[keep]
            neighbor_similarities = similarities[keep]
            weights = self.char_counts_[neighbor_indices] * np.maximum(neighbor_similarities, 1e-3)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                csn_vectors[row_index] = self.base_char_vectors_[row_index]
            else:
                csn_vectors[row_index] = (self.base_char_vectors_[neighbor_indices] * weights[:, None]).sum(axis=0) / weight_sum
        return normalize(csn_vectors).astype(np.float32)

    def _build_code_vectors(self):
        feature_dicts = [character_code_features(char) for char in self.characters_]
        vectorizer = DictVectorizer(sparse=True)
        return normalize(vectorizer.fit_transform(feature_dicts))

    def _sentence_vector(self, text: str) -> np.ndarray:
        indices = [
            self.char_to_index_[token]
            for token in char_tokenize(text)[: self.config.max_chars]
            if token in self.char_to_index_
        ]
        if not indices:
            return np.zeros(self.csn_char_vectors_.shape[1], dtype=np.float32)

        token_vectors = self.csn_char_vectors_[indices]
        mean_vector = token_vectors.mean(axis=0)
        scores = token_vectors @ mean_vector / math.sqrt(token_vectors.shape[1])
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = weights / weights.sum()
        attention_vector = weights @ token_vectors
        sentence_vector = 0.5 * mean_vector + 0.5 * attention_vector
        norm = np.linalg.norm(sentence_vector)
        if norm > 0:
            sentence_vector = sentence_vector / norm
        return sentence_vector.astype(np.float32)


def run_experiment(
    data_path: str | Path,
    output_csv: str | Path | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
    embedding_dim: int = 64,
    similarity_threshold: float = 0.52,
) -> pd.DataFrame:
    data = read_dataset(data_path)
    train_x, test_x, train_y, test_y = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )

    config = RuleFreeCSNConfig(
        embedding_dim=embedding_dim,
        similarity_threshold=similarity_threshold,
        random_state=random_state,
    )
    model = CourseStyleCSNClassifier(config)
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    score_y = model.predict_proba(test_x)[:, 1]
    counts = test_y.value_counts().to_dict()
    metrics = evaluate_predictions(test_y, pred_y, score_y)
    result = pd.DataFrame(
        [
            {
                "model": "Rule-free CSN+LR",
                "description": "Similarity network char embeddings + sentence embeddings + Logistic Regression",
                "n_samples": int(len(test_y)),
                "n_normal": int(counts.get(0, 0)),
                "n_spam": int(counts.get(1, 0)),
                "embedding_dim": int(embedding_dim),
                "similarity_threshold": float(similarity_threshold),
                **metrics,
            }
        ]
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run course-style rule-free CSN experiment.")
    parser.add_argument("--data", default="data/raw/dataset.txt")
    parser.add_argument("--out-csv")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--similarity-threshold", type=float, default=0.52)
    args = parser.parse_args()
    result = run_experiment(
        args.data,
        output_csv=args.out_csv,
        embedding_dim=args.embedding_dim,
        similarity_threshold=args.similarity_threshold,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
