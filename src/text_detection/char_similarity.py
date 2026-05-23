"""Character similarity utilities.

The full course algorithm can later plug in four-corner codes, stroke-count
codes, and richer glyph dictionaries. This module starts with a practical
phonetic and hand-built confusable-character baseline.
"""

from __future__ import annotations

from functools import lru_cache

from pypinyin import Style, lazy_pinyin


CONFUSABLE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("微", "薇", "维", "唯", "胃", "卫", "魏", "韦", "喂"),
    ("信", "新", "欣", "薪", "芯", "辛", "心"),
    ("钱", "前", "钳", "签"),
    ("赚", "转", "砖", "专"),
    ("加", "佳", "家"),
    ("聊", "辽", "撩"),
)


CONFUSABLE_MAP: dict[str, set[str]] = {}
for group in CONFUSABLE_GROUPS:
    chars = set(group)
    for char in chars:
        CONFUSABLE_MAP.setdefault(char, set()).update(chars - {char})


@lru_cache(maxsize=4096)
def _pinyin_code(char: str) -> str:
    if not char:
        return ""
    return lazy_pinyin(char[0], style=Style.TONE3, errors="ignore")[0]


def phonetic_similarity(char_a: str, char_b: str) -> float:
    """Return a coarse phonetic similarity score in [0, 1]."""
    if char_a == char_b:
        return 1.0

    code_a = _pinyin_code(char_a)
    code_b = _pinyin_code(char_b)
    if not code_a or not code_b:
        return 0.0
    if code_a == code_b:
        return 0.9
    if code_a.rstrip("12345") == code_b.rstrip("12345"):
        return 0.75
    if code_a[0] == code_b[0]:
        return 0.35
    return 0.0


def glyph_similarity(char_a: str, char_b: str) -> float:
    """Return a small dictionary-based glyph/confusable similarity score."""
    if char_a == char_b:
        return 1.0
    if char_b in CONFUSABLE_MAP.get(char_a, set()):
        return 0.85
    return 0.0


def character_similarity(char_a: str, char_b: str) -> float:
    """Fuse glyph and phonetic similarity using max pooling."""
    return max(glyph_similarity(char_a, char_b), phonetic_similarity(char_a, char_b))


def normalize_variant_word(word: str, threshold: float = 0.75) -> str:
    """Map common adversarial variants of keywords back to canonical forms."""
    canonical = "微信"
    if len(word) != len(canonical):
        return word

    scores = [
        character_similarity(source, target)
        for source, target in zip(word, canonical, strict=True)
    ]
    if min(scores) >= threshold:
        return canonical
    return word

