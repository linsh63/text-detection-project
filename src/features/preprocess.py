"""Text cleaning and tokenization helpers."""

from __future__ import annotations

import re
from typing import Iterable

import jieba


_KEEP_PATTERN = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]")
_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize a raw text string for Chinese spam detection."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = _KEEP_PATTERN.sub(" ", text)
    return _SPACE_PATTERN.sub(" ", text).strip()


def char_tokenize(text: str) -> list[str]:
    """Tokenize by Chinese character and non-space ASCII chunk."""
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


def word_tokenize(text: str) -> list[str]:
    """Tokenize Chinese text with jieba after cleaning."""
    text = clean_text(text)
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def join_tokens(tokens: Iterable[str]) -> str:
    """Join tokens for scikit-learn vectorizers."""
    return " ".join(tokens)

