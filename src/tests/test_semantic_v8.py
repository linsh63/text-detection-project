import numpy as np
import pandas as pd

from src.experiments.semantic_runners import _auto_variants, _extract_auto_hard_terms, _make_auto_augmentation
from src.preprocessing.normalizer import normalize_for_semantic_model
from src.semantic_models.classifier import FrozenEncoderClassifier


class DummyEncoder:
    def encode(self, texts, batch_size=64):
        return np.asarray(
            [[1.0, 0.0] if "spam" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def test_semantic_normalizer_is_rule_free_noise_cleanup():
    assert normalize_for_semantic_model("\uff21\uff22\tHi\n") == "ab hi"


def test_frozen_encoder_classifier_scores_positive_text_higher():
    classifier = FrozenEncoderClassifier(DummyEncoder(), batch_size=2)
    classifier.fit(["normal message", "spam offer", "normal chat", "spam loan"], [0, 1, 0, 1])

    scores = classifier.score_texts(["normal message", "spam offer"])

    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_auto_hard_terms_are_mined_from_label_statistics():
    texts = pd.Series(["优惠活动", "优惠办理", "普通通知", "普通提醒"])
    labels = pd.Series([1, 1, 0, 0])

    terms = _extract_auto_hard_terms(texts, labels, hard_texts=pd.Series(["优惠活动"]), scope="unit", min_spam_df=1)

    assert "优惠" in set(terms["term"])
    assert terms.loc[terms["term"] == "优惠", "spam_df"].iloc[0] == 2


def test_auto_augmentation_generates_positive_variants_without_keyword_table():
    texts = pd.Series(["优惠活动", "优惠办理", "普通通知", "普通提醒"])
    labels = pd.Series([1, 1, 0, 0])
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)

    aug_texts, aug_labels, terms, examples = _make_auto_augmentation(
        texts,
        labels,
        vectors,
        scope="unit",
        max_terms=3,
        max_augmented=5,
        min_spam_df=1,
    )

    assert len(terms) > 0
    assert len(examples) == len(aug_texts)
    assert set(aug_labels) == {1}


def test_auto_variants_have_stable_order():
    variants = _auto_variants("优惠", {})

    assert variants == sorted(variants)
