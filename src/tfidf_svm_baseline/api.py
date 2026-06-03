"""HTTP API for TF-IDF+SVM text violation detection."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .experiment import build_model, read_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.txt"

app = FastAPI(
    title="Text Violation Detection API",
    version="1.0.0",
    description="TF-IDF+SVM API that returns whether an input text is违规.",
)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待检测文本")


def get_data_path() -> Path:
    value = os.environ.get("TEXT_DETECTION_DATA")
    if value:
        return Path(value).expanduser().resolve()
    return DEFAULT_DATA_PATH


def get_model_path() -> Path | None:
    value = os.environ.get("TEXT_DETECTION_MODEL")
    if not value:
        return None
    return Path(value).expanduser().resolve()


@lru_cache(maxsize=1)
def get_model():
    model_path = get_model_path()
    if model_path is not None and model_path.exists():
        return joblib.load(model_path)

    data = read_dataset(get_data_path())
    model = build_model()
    model.fit(data["text"], data["label"])
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    return model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "TF-IDF+SVM"}


@app.post("/predict", response_model=bool)
def predict(request: TextRequest) -> bool:
    text = request.text.strip()
    if not text:
        return False
    prediction = get_model().predict([text])[0]
    return bool(int(prediction) == 1)
