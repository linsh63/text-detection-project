# 统一评测协议结果

本实验把旧模型 v0-v7 放入统一评测协议，后续 v8 语义模型也应接入同一协议。

## 协议定义

| 协议 | 名称 | 训练数据 | 测试数据 | 目的 |
|---|---|---|---|---|
| A | In-domain | 主数据 train | 主数据 holdout | 衡量课程主任务表现 |
| B | Zero-shot cross-domain | 只用主数据 train | 外部 holdout | 衡量裸泛化能力 |
| C | Few-shot domain adaptation | 主数据 + 外部 adapt train | 外部 holdout | 衡量少量外部标注后的泛化 |
| D | Adversarial robustness | 对应模型训练数据 | adversarial / keyword challenge | 衡量变体鲁棒性 |

## 外部数据切分

| dataset | total_rows | adapt_train_rows | adapt_fit_rows | adapt_valid_rows | holdout_rows | holdout_spam | holdout_normal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fbs_mixed | 10000 | 3000 | 2400 | 600 | 7000 | 3500 | 3500 |
| hf_chinese_spam_10000 | 9941 | 2982 | 2385 | 597 | 6959 | 3488 | 3471 |
| hf_chinese_conversation_spam | 7731 | 2319 | 1855 | 464 | 5412 | 2154 | 3258 |

## 每个协议与数据集的最优结果

| protocol_id | dataset | model_version | training_scope | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | main_holdout | v5_main_fusion | main_only | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |
| B | fbs_mixed_holdout | v1_char_svm | main_only | 0.9149 | 0.9863 | 0.8414 | 0.9081 | 41.0000 | 555.0000 |
| B | hf_chinese_conversation_spam_holdout | v3_csn_aug | main_only | 0.7441 | 0.9776 | 0.3654 | 0.5319 | 18.0000 | 1367.0000 |
| B | hf_chinese_spam_10000_holdout | v3_csn_aug | main_only | 0.5334 | 0.8283 | 0.0872 | 0.1577 | 63.0000 | 3184.0000 |
| C | fbs_mixed_holdout | v6_csn_multisource | main_plus_external_adapt | 0.9833 | 0.9807 | 0.9860 | 0.9833 | 68.0000 | 49.0000 |
| C | hf_chinese_conversation_spam_holdout | v6_csn_multisource | main_plus_external_adapt | 0.9595 | 0.9532 | 0.9448 | 0.9489 | 100.0000 | 119.0000 |
| C | hf_chinese_spam_10000_holdout | v6_fusion_multisource | main_plus_external_adapt | 0.8921 | 0.9289 | 0.8498 | 0.8876 | 227.0000 | 524.0000 |
| D | adversarial | v1_char_svm | main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | keyword_challenge | v3_csn_aug | main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## 完整结果

| protocol_id | protocol_name | dataset | model_version | training_scope | n_samples | n_normal | n_spam | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | In-domain main holdout | main_holdout | v0_char_logreg | main_only | 6000 | 5396 | 604 | 0.9773 | 0.8324 | 0.9702 | 0.8960 | 118.0000 | 18.0000 |
| A | In-domain main holdout | main_holdout | v1_char_svm | main_only | 6000 | 5396 | 604 | 0.9870 | 0.9021 | 0.9768 | 0.9380 | 64.0000 | 14.0000 |
| A | In-domain main holdout | main_holdout | v2_csn_svm | main_only | 6000 | 5396 | 604 | 0.9870 | 0.9021 | 0.9768 | 0.9380 | 64.0000 | 14.0000 |
| A | In-domain main holdout | main_holdout | v3_csn_aug | main_only | 6000 | 5396 | 604 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84.0000 | 15.0000 |
| A | In-domain main holdout | main_holdout | v4_bad_case | main_only | 6000 | 5396 | 604 | 0.9902 | 0.9534 | 0.9487 | 0.9510 | 28.0000 | 31.0000 |
| A | In-domain main holdout | main_holdout | v5_main_fusion | main_only | 6000 | 5396 | 604 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |
| A | In-domain main holdout | main_holdout | v6_csn_multisource | main_plus_external_adapt | 6000 | 5396 | 604 | 0.9790 | 0.8454 | 0.9685 | 0.9028 | 107.0000 | 19.0000 |
| A | In-domain main holdout | main_holdout | v6_fusion_multisource | main_plus_external_adapt | 6000 | 5396 | 604 | 0.9843 | 0.8840 | 0.9719 | 0.9259 | 77.0000 | 17.0000 |
| A | In-domain main holdout | main_holdout | v7_bridge | main_plus_external_adapt | 6000 | 5396 | 604 | 0.9890 | 0.9256 | 0.9685 | 0.9466 | 47.0000 | 19.0000 |
| B | Zero-shot cross-domain | fbs_mixed_holdout | v0_char_logreg | main_only | 7000 | 3500 | 3500 | 0.9113 | 0.9759 | 0.8434 | 0.9048 | 73.0000 | 548.0000 |
| B | Zero-shot cross-domain | fbs_mixed_holdout | v1_char_svm | main_only | 7000 | 3500 | 3500 | 0.9149 | 0.9863 | 0.8414 | 0.9081 | 41.0000 | 555.0000 |
| B | Zero-shot cross-domain | fbs_mixed_holdout | v2_csn_svm | main_only | 7000 | 3500 | 3500 | 0.9147 | 0.9863 | 0.8411 | 0.9079 | 41.0000 | 556.0000 |
| B | Zero-shot cross-domain | fbs_mixed_holdout | v3_csn_aug | main_only | 7000 | 3500 | 3500 | 0.9131 | 0.9843 | 0.8397 | 0.9063 | 47.0000 | 561.0000 |
| B | Zero-shot cross-domain | fbs_mixed_holdout | v4_bad_case | main_only | 7000 | 3500 | 3500 | 0.8391 | 0.9950 | 0.6817 | 0.8091 | 12.0000 | 1114.0000 |
| B | Zero-shot cross-domain | fbs_mixed_holdout | v5_main_fusion | main_only | 7000 | 3500 | 3500 | 0.8390 | 0.9938 | 0.6823 | 0.8091 | 15.0000 | 1112.0000 |
| C | Few-shot domain adaptation | fbs_mixed_holdout | v3_csn_aug | main_only | 7000 | 3500 | 3500 | 0.9131 | 0.9843 | 0.8397 | 0.9063 | 47.0000 | 561.0000 |
| C | Few-shot domain adaptation | fbs_mixed_holdout | v5_main_fusion | main_only | 7000 | 3500 | 3500 | 0.8390 | 0.9938 | 0.6823 | 0.8091 | 15.0000 | 1112.0000 |
| C | Few-shot domain adaptation | fbs_mixed_holdout | v6_csn_multisource | main_plus_external_adapt | 7000 | 3500 | 3500 | 0.9833 | 0.9807 | 0.9860 | 0.9833 | 68.0000 | 49.0000 |
| C | Few-shot domain adaptation | fbs_mixed_holdout | v6_fusion_multisource | main_plus_external_adapt | 7000 | 3500 | 3500 | 0.9831 | 0.9865 | 0.9797 | 0.9831 | 47.0000 | 71.0000 |
| C | Few-shot domain adaptation | fbs_mixed_holdout | v7_bridge | main_plus_external_adapt | 7000 | 3500 | 3500 | 0.9789 | 0.9901 | 0.9674 | 0.9786 | 34.0000 | 114.0000 |
| B | Zero-shot cross-domain | hf_chinese_spam_10000_holdout | v0_char_logreg | main_only | 6959 | 3471 | 3488 | 0.4963 | 0.3396 | 0.0052 | 0.0102 | 35.0000 | 3470.0000 |
| B | Zero-shot cross-domain | hf_chinese_spam_10000_holdout | v1_char_svm | main_only | 6959 | 3471 | 3488 | 0.4986 | 0.4894 | 0.0066 | 0.0130 | 24.0000 | 3465.0000 |
| B | Zero-shot cross-domain | hf_chinese_spam_10000_holdout | v2_csn_svm | main_only | 6959 | 3471 | 3488 | 0.4986 | 0.4894 | 0.0066 | 0.0130 | 24.0000 | 3465.0000 |
| B | Zero-shot cross-domain | hf_chinese_spam_10000_holdout | v3_csn_aug | main_only | 6959 | 3471 | 3488 | 0.5334 | 0.8283 | 0.0872 | 0.1577 | 63.0000 | 3184.0000 |
| B | Zero-shot cross-domain | hf_chinese_spam_10000_holdout | v4_bad_case | main_only | 6959 | 3471 | 3488 | 0.5139 | 0.8387 | 0.0373 | 0.0714 | 25.0000 | 3358.0000 |
| B | Zero-shot cross-domain | hf_chinese_spam_10000_holdout | v5_main_fusion | main_only | 6959 | 3471 | 3488 | 0.5157 | 0.8512 | 0.0410 | 0.0782 | 25.0000 | 3345.0000 |
| C | Few-shot domain adaptation | hf_chinese_spam_10000_holdout | v3_csn_aug | main_only | 6959 | 3471 | 3488 | 0.5334 | 0.8283 | 0.0872 | 0.1577 | 63.0000 | 3184.0000 |
| C | Few-shot domain adaptation | hf_chinese_spam_10000_holdout | v5_main_fusion | main_only | 6959 | 3471 | 3488 | 0.5157 | 0.8512 | 0.0410 | 0.0782 | 25.0000 | 3345.0000 |
| C | Few-shot domain adaptation | hf_chinese_spam_10000_holdout | v6_csn_multisource | main_plus_external_adapt | 6959 | 3471 | 3488 | 0.8881 | 0.9030 | 0.8701 | 0.8863 | 326.0000 | 453.0000 |
| C | Few-shot domain adaptation | hf_chinese_spam_10000_holdout | v6_fusion_multisource | main_plus_external_adapt | 6959 | 3471 | 3488 | 0.8921 | 0.9289 | 0.8498 | 0.8876 | 227.0000 | 524.0000 |
| C | Few-shot domain adaptation | hf_chinese_spam_10000_holdout | v7_bridge | main_plus_external_adapt | 6959 | 3471 | 3488 | 0.8846 | 0.9485 | 0.8139 | 0.8761 | 154.0000 | 649.0000 |
| B | Zero-shot cross-domain | hf_chinese_conversation_spam_holdout | v0_char_logreg | main_only | 5412 | 3258 | 2154 | 0.7373 | 0.9841 | 0.3454 | 0.5113 | 12.0000 | 1410.0000 |
| B | Zero-shot cross-domain | hf_chinese_conversation_spam_holdout | v1_char_svm | main_only | 5412 | 3258 | 2154 | 0.7310 | 0.9834 | 0.3296 | 0.4937 | 12.0000 | 1444.0000 |
| B | Zero-shot cross-domain | hf_chinese_conversation_spam_holdout | v2_csn_svm | main_only | 5412 | 3258 | 2154 | 0.7291 | 0.9831 | 0.3250 | 0.4885 | 12.0000 | 1454.0000 |
| B | Zero-shot cross-domain | hf_chinese_conversation_spam_holdout | v3_csn_aug | main_only | 5412 | 3258 | 2154 | 0.7441 | 0.9776 | 0.3654 | 0.5319 | 18.0000 | 1367.0000 |
| B | Zero-shot cross-domain | hf_chinese_conversation_spam_holdout | v4_bad_case | main_only | 5412 | 3258 | 2154 | 0.6842 | 0.9890 | 0.2089 | 0.3450 | 5.0000 | 1704.0000 |
| B | Zero-shot cross-domain | hf_chinese_conversation_spam_holdout | v5_main_fusion | main_only | 5412 | 3258 | 2154 | 0.7151 | 0.9935 | 0.2860 | 0.4441 | 4.0000 | 1538.0000 |
| C | Few-shot domain adaptation | hf_chinese_conversation_spam_holdout | v3_csn_aug | main_only | 5412 | 3258 | 2154 | 0.7441 | 0.9776 | 0.3654 | 0.5319 | 18.0000 | 1367.0000 |
| C | Few-shot domain adaptation | hf_chinese_conversation_spam_holdout | v5_main_fusion | main_only | 5412 | 3258 | 2154 | 0.7151 | 0.9935 | 0.2860 | 0.4441 | 4.0000 | 1538.0000 |
| C | Few-shot domain adaptation | hf_chinese_conversation_spam_holdout | v6_csn_multisource | main_plus_external_adapt | 5412 | 3258 | 2154 | 0.9595 | 0.9532 | 0.9448 | 0.9489 | 100.0000 | 119.0000 |
| C | Few-shot domain adaptation | hf_chinese_conversation_spam_holdout | v6_fusion_multisource | main_plus_external_adapt | 5412 | 3258 | 2154 | 0.9586 | 0.9662 | 0.9285 | 0.9470 | 70.0000 | 154.0000 |
| C | Few-shot domain adaptation | hf_chinese_conversation_spam_holdout | v7_bridge | main_plus_external_adapt | 5412 | 3258 | 2154 | 0.9479 | 0.9761 | 0.8909 | 0.9316 | 47.0000 | 235.0000 |
| D | Adversarial robustness | adversarial | v0_char_logreg | main_only | 131 | 0 | 131 | 0.9924 | 1.0000 | 0.9924 | 0.9962 | 0.0000 | 1.0000 |
| D | Adversarial robustness | adversarial | v1_char_svm | main_only | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v2_csn_svm | main_only | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v3_csn_aug | main_only | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v4_bad_case | main_only | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v5_main_fusion | main_only | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v6_csn_multisource | main_plus_external_adapt | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v6_fusion_multisource | main_plus_external_adapt | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v7_bridge | main_plus_external_adapt | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | keyword_challenge | v0_char_logreg | main_only | 270 | 0 | 270 | 0.1444 | 1.0000 | 0.1444 | 0.2524 | 0.0000 | 231.0000 |
| D | Adversarial robustness | keyword_challenge | v1_char_svm | main_only | 270 | 0 | 270 | 0.0667 | 1.0000 | 0.0667 | 0.1250 | 0.0000 | 252.0000 |
| D | Adversarial robustness | keyword_challenge | v2_csn_svm | main_only | 270 | 0 | 270 | 0.1037 | 1.0000 | 0.1037 | 0.1879 | 0.0000 | 242.0000 |
| D | Adversarial robustness | keyword_challenge | v3_csn_aug | main_only | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | keyword_challenge | v4_bad_case | main_only | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | keyword_challenge | v5_main_fusion | main_only | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | keyword_challenge | v6_csn_multisource | main_plus_external_adapt | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | keyword_challenge | v6_fusion_multisource | main_plus_external_adapt | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | keyword_challenge | v7_bridge | main_plus_external_adapt | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
