# v8 语义编码统一评测结果

本实验把 v8.0 冻结语义编码器放入与 v0-v7 相同的 A/B/C/D 统一评测协议，用于判断纯模型语义推断方案是否能提升跨来源泛化。

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
| A | main_holdout | v8_semantic_main | main_only | 0.9923 | 0.9619 | 0.9619 | 0.9619 | 23.0000 | 23.0000 |
| B | fbs_mixed_holdout | v8_semantic_main | main_only | 0.6449 | 0.9875 | 0.2934 | 0.4524 | 13.0000 | 2473.0000 |
| B | hf_chinese_conversation_spam_holdout | v8_semantic_main | main_only | 0.7539 | 0.9801 | 0.3895 | 0.5575 | 17.0000 | 1315.0000 |
| B | hf_chinese_spam_10000_holdout | v8_semantic_main | main_only | 0.4988 | 0.5000 | 0.0032 | 0.0063 | 11.0000 | 3477.0000 |
| C | fbs_mixed_holdout | v8_semantic_multisource | main_plus_external_adapt | 0.9709 | 0.9771 | 0.9643 | 0.9707 | 79.0000 | 125.0000 |
| C | hf_chinese_conversation_spam_holdout | v8_semantic_multisource | main_plus_external_adapt | 0.9200 | 0.8971 | 0.9025 | 0.8998 | 223.0000 | 210.0000 |
| C | hf_chinese_spam_10000_holdout | v8_semantic_multisource | main_plus_external_adapt | 0.8526 | 0.8852 | 0.8111 | 0.8465 | 367.0000 | 659.0000 |
| D | adversarial | v8_semantic_main | main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | keyword_challenge | v8_semantic_multisource | main_plus_external_adapt | 0.5889 | 1.0000 | 0.5889 | 0.7413 | 0.0000 | 111.0000 |

## 完整结果

| protocol_id | protocol_name | dataset | model_version | training_scope | n_samples | n_normal | n_spam | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | In-domain main holdout | main_holdout | v8_semantic_main | main_only | 6000 | 5396 | 604 | 0.9923 | 0.9619 | 0.9619 | 0.9619 | 23.0000 | 23.0000 |
| A | In-domain main holdout | main_holdout | v8_semantic_multisource | main_plus_external_adapt | 6000 | 5396 | 604 | 0.9717 | 0.8014 | 0.9553 | 0.8716 | 143.0000 | 27.0000 |
| B | Zero-shot cross-domain | fbs_mixed_holdout | v8_semantic_main | main_only | 7000 | 3500 | 3500 | 0.6449 | 0.9875 | 0.2934 | 0.4524 | 13.0000 | 2473.0000 |
| C | Few-shot domain adaptation | fbs_mixed_holdout | v8_semantic_multisource | main_plus_external_adapt | 7000 | 3500 | 3500 | 0.9709 | 0.9771 | 0.9643 | 0.9707 | 79.0000 | 125.0000 |
| B | Zero-shot cross-domain | hf_chinese_spam_10000_holdout | v8_semantic_main | main_only | 6959 | 3471 | 3488 | 0.4988 | 0.5000 | 0.0032 | 0.0063 | 11.0000 | 3477.0000 |
| C | Few-shot domain adaptation | hf_chinese_spam_10000_holdout | v8_semantic_multisource | main_plus_external_adapt | 6959 | 3471 | 3488 | 0.8526 | 0.8852 | 0.8111 | 0.8465 | 367.0000 | 659.0000 |
| B | Zero-shot cross-domain | hf_chinese_conversation_spam_holdout | v8_semantic_main | main_only | 5412 | 3258 | 2154 | 0.7539 | 0.9801 | 0.3895 | 0.5575 | 17.0000 | 1315.0000 |
| C | Few-shot domain adaptation | hf_chinese_conversation_spam_holdout | v8_semantic_multisource | main_plus_external_adapt | 5412 | 3258 | 2154 | 0.9200 | 0.8971 | 0.9025 | 0.8998 | 223.0000 | 210.0000 |
| D | Adversarial robustness | adversarial | v8_semantic_main | main_only | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| D | Adversarial robustness | adversarial | v8_semantic_multisource | main_plus_external_adapt | 131 | 0 | 131 | 0.9924 | 1.0000 | 0.9924 | 0.9962 | 0.0000 | 1.0000 |
| D | Adversarial robustness | keyword_challenge | v8_semantic_main | main_only | 270 | 0 | 270 | 0.0481 | 1.0000 | 0.0481 | 0.0919 | 0.0000 | 257.0000 |
| D | Adversarial robustness | keyword_challenge | v8_semantic_multisource | main_plus_external_adapt | 270 | 0 | 270 | 0.5889 | 1.0000 | 0.5889 | 0.7413 | 0.0000 | 111.0000 |
