from src.features import char_tokenize, clean_text


def test_clean_text_removes_punctuation():
    assert clean_text("测试文本！！！") == "测试文本"


def test_char_tokenize_keeps_ascii_chunks():
    assert char_tokenize("加V联系 abc") == ["加", "v", "联", "系", "abc"]
