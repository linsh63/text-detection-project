"""Bad-case driven risk features for spam text detection."""

from __future__ import annotations

import re
from collections.abc import Iterable


GAMBLING_TERMS = (
    "六合彩",
    "特码",
    "一肖",
    "六肖",
    "连码",
    "平码",
    "中六",
    "挂牌",
    "取码",
    "博彩",
    "开奖",
)

OBFUSCATED_BUSINESS_TERMS = (
    "白手起家",
    "京喜",
    "祛斑",
    "tel",
    "加qq",
    "qq",
)

ENTERTAINMENT_TERMS = (
    "娱乐城",
    "娱乐场",
    "会所",
    "太阳城",
)

RISK_TERMS = GAMBLING_TERMS + ENTERTAINMENT_TERMS + OBFUSCATED_BUSINESS_TERMS

OBFUSCATION_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"娱\W*乐",
        r"太\W*阳\W*城",
        r"保\W*障",
        r"会\W*所",
        r"加\W*q\W*q",
    )
)

URLISH_PATTERN = re.compile(
    r"(https?://|www\.|wap\.|[a-z0-9-]+\.(com|cn|net|pw|top|vip)|"
    r"点\s*(com|cn|net|pw|top|vip))",
    flags=re.IGNORECASE,
)

MASKED_CONTACT_PATTERN = re.compile(
    r"(tel|电话)\W*[:：]?\W*x{5,}",
    flags=re.IGNORECASE,
)


def spam_risk_score(text: str) -> float:
    """Return a compact risk score from recurring false-negative patterns."""
    value = str(text)
    lowered = value.lower()

    score = 0.0
    score += sum(1.0 for term in RISK_TERMS if term.lower() in lowered)
    score += sum(2.0 for pattern in OBFUSCATION_PATTERNS if pattern.search(value))

    if URLISH_PATTERN.search(value):
        score += 1.5
    if MASKED_CONTACT_PATTERN.search(value):
        score += 1.0

    return score


def spam_risk_scores(texts: Iterable[str]) -> list[float]:
    """Vector-friendly wrapper around spam_risk_score."""
    return [spam_risk_score(text) for text in texts]
