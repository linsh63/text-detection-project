from text_detection.char_similarity import normalize_variant_word
from text_detection.preprocess import char_tokenize, clean_text


def test_clean_text_removes_punctuation():
    assert clean_text("加我薇信！！！") == "加我薇信"


def test_char_tokenize_keeps_ascii_chunks():
    assert char_tokenize("加V联系 abc") == ["加", "v", "联", "系", "abc"]


def test_normalize_variant_word():
    assert normalize_variant_word("薇信") == "微信"

