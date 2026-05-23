"""Character similarity utilities.

The full course algorithm can later plug in four-corner codes, stroke-count
codes, and richer glyph dictionaries. This module starts with a practical
phonetic and hand-built confusable-character baseline.
"""

from __future__ import annotations

from functools import lru_cache

from pypinyin import Style, lazy_pinyin


KEYWORD_VARIANTS: dict[str, tuple[str, ...]] = {
    "微信": ("薇信", "维信", "胃信", "卫信", "微辛", "薇辛", "胃星", "卫星", "韦欣", "喂新"),
    "赚钱": ("赚前", "转钱", "赚钳", "赚签", "挣钱", "掙钱", "掙錢"),
    "返利": ("反利", "返荔", "返立"),
    "兼职": ("兼直", "蒹职", "兼只", "兼職"),
    "刷单": ("唰单", "刷旦", "刷箪", "刷單"),
    "贷款": ("贷款讯", "代款", "贷歀", "貸款"),
    "信用卡": ("信佣卡", "芯用卡", "信用咔", "信用咭"),
    "红包": ("红苞", "紅包", "虹包"),
    "私聊": ("丝聊", "私撩", "私辽"),
    "加我": ("佳我", "家我", "茄我"),
}


CONFUSABLE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("微", "薇", "维", "唯", "胃", "卫", "魏", "韦", "喂"),
    ("信", "新", "欣", "薪", "芯", "辛", "心"),
    ("钱", "前", "钳", "签"),
    ("赚", "转", "砖", "专"),
    ("加", "佳", "家"),
    ("聊", "辽", "撩"),
)


CONFUSABLE_MAP: dict[str, set[str]] = {}
CANONICAL_CHAR_MAP: dict[str, str] = {}
for group in CONFUSABLE_GROUPS:
    chars = set(group)
    canonical = group[0]
    for char in chars:
        CONFUSABLE_MAP.setdefault(char, set()).update(chars - {char})
        CANONICAL_CHAR_MAP.setdefault(char, canonical)


VARIANT_TO_KEYWORD: dict[str, str] = {}
for keyword, variants in KEYWORD_VARIANTS.items():
    VARIANT_TO_KEYWORD[keyword] = keyword
    for variant in variants:
        VARIANT_TO_KEYWORD[variant] = keyword


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


def normalize_text_variants(text: str) -> str:
    """Normalize known adversarial words to canonical sensitive keywords."""
    normalized = str(text)
    for variant, keyword in sorted(
        VARIANT_TO_KEYWORD.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if variant != keyword:
            normalized = normalized.replace(variant, keyword)
    return normalized


def canonicalize_confusable_chars(text: str) -> str:
    """Canonicalize individual confusable characters.

    This is intentionally exposed separately because phrase-level normalization
    is safer for production use, while character-level canonicalization is useful
    in controlled ablation experiments.
    """
    return "".join(CANONICAL_CHAR_MAP.get(char, char) for char in str(text))
