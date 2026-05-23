from src.features.risk_features import spam_risk_score


def test_spam_risk_score_detects_gambling_terms():
    assert spam_risk_score("六合彩公司为你提供一码中特") > 0


def test_spam_risk_score_detects_obfuscated_entertainment_terms():
    assert spam_risk_score("娱/乐会/所这么多，首选太/阳/城") >= 4


def test_spam_risk_score_keeps_plain_text_low():
    assert spam_risk_score("今天下午三点开组会") == 0
