# v6/v7 跨来源适配实验

本实验从外部二分类数据集中切出一部分作为适配训练数据，剩余部分作为外部保留测试集，用于验证是否能提升跨来源泛化。

- v6 融合参数：risk_bonus=0.30, threshold=0.15
- v7 桥接阈值：threshold=0.35
- v6 训练数据：主训练集 + FBS/HF 外部适配训练集 + 关键词增强样本。
- v7 桥接策略：`max(v5_main_only_score, v6_multisource_score)`，用于兼顾主数据集和外部数据集。
- 外部 holdout 没有进入训练或阈值选择。

## 外部数据切分

| dataset | total_rows | adapt_train_rows | adapt_fit_rows | adapt_valid_rows | holdout_rows | holdout_spam | holdout_normal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fbs_mixed | 10000 | 3000 | 2400 | 600 | 7000 | 3500 | 3500 |
| hf_chinese_spam_10000 | 9941 | 2982 | 2385 | 597 | 6959 | 3488 | 3471 |
| hf_chinese_conversation_spam | 7731 | 2319 | 1855 | 464 | 5412 | 2154 | 3258 |

## 每个数据集最优结果

| dataset | name | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adversarial | v3_main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| fbs_mixed_holdout | v6_csn_aug_multisource | 0.9833 | 0.9807 | 0.9860 | 0.9833 | 68.0000 | 49.0000 |
| hf_chinese_conversation_spam_holdout | v6_csn_aug_multisource | 0.9595 | 0.9532 | 0.9448 | 0.9489 | 100.0000 | 119.0000 |
| hf_chinese_spam_10000_holdout | v6_fusion_multisource | 0.8921 | 0.9289 | 0.8498 | 0.8876 | 227.0000 | 524.0000 |
| keyword_challenge | v3_main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| main_holdout | v5_main_only_fusion | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |

## 完整指标

| dataset | name | n_samples | n_normal | n_spam | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_holdout | v3_main_only | 6000 | 5396 | 604 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84.0000 | 15.0000 |
| main_holdout | v5_main_only_fusion | 6000 | 5396 | 604 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |
| main_holdout | v6_csn_aug_multisource | 6000 | 5396 | 604 | 0.9790 | 0.8454 | 0.9685 | 0.9028 | 107.0000 | 19.0000 |
| main_holdout | v6_fusion_multisource | 6000 | 5396 | 604 | 0.9843 | 0.8840 | 0.9719 | 0.9259 | 77.0000 | 17.0000 |
| main_holdout | v7_bridge_main_multisource | 6000 | 5396 | 604 | 0.9890 | 0.9256 | 0.9685 | 0.9466 | 47.0000 | 19.0000 |
| fbs_mixed_holdout | v3_main_only | 7000 | 3500 | 3500 | 0.9131 | 0.9843 | 0.8397 | 0.9063 | 47.0000 | 561.0000 |
| fbs_mixed_holdout | v5_main_only_fusion | 7000 | 3500 | 3500 | 0.8390 | 0.9938 | 0.6823 | 0.8091 | 15.0000 | 1112.0000 |
| fbs_mixed_holdout | v6_csn_aug_multisource | 7000 | 3500 | 3500 | 0.9833 | 0.9807 | 0.9860 | 0.9833 | 68.0000 | 49.0000 |
| fbs_mixed_holdout | v6_fusion_multisource | 7000 | 3500 | 3500 | 0.9831 | 0.9865 | 0.9797 | 0.9831 | 47.0000 | 71.0000 |
| fbs_mixed_holdout | v7_bridge_main_multisource | 7000 | 3500 | 3500 | 0.9789 | 0.9901 | 0.9674 | 0.9786 | 34.0000 | 114.0000 |
| hf_chinese_spam_10000_holdout | v3_main_only | 6959 | 3471 | 3488 | 0.5334 | 0.8283 | 0.0872 | 0.1577 | 63.0000 | 3184.0000 |
| hf_chinese_spam_10000_holdout | v5_main_only_fusion | 6959 | 3471 | 3488 | 0.5157 | 0.8512 | 0.0410 | 0.0782 | 25.0000 | 3345.0000 |
| hf_chinese_spam_10000_holdout | v6_csn_aug_multisource | 6959 | 3471 | 3488 | 0.8881 | 0.9030 | 0.8701 | 0.8863 | 326.0000 | 453.0000 |
| hf_chinese_spam_10000_holdout | v6_fusion_multisource | 6959 | 3471 | 3488 | 0.8921 | 0.9289 | 0.8498 | 0.8876 | 227.0000 | 524.0000 |
| hf_chinese_spam_10000_holdout | v7_bridge_main_multisource | 6959 | 3471 | 3488 | 0.8846 | 0.9485 | 0.8139 | 0.8761 | 154.0000 | 649.0000 |
| hf_chinese_conversation_spam_holdout | v3_main_only | 5412 | 3258 | 2154 | 0.7441 | 0.9776 | 0.3654 | 0.5319 | 18.0000 | 1367.0000 |
| hf_chinese_conversation_spam_holdout | v5_main_only_fusion | 5412 | 3258 | 2154 | 0.7151 | 0.9935 | 0.2860 | 0.4441 | 4.0000 | 1538.0000 |
| hf_chinese_conversation_spam_holdout | v6_csn_aug_multisource | 5412 | 3258 | 2154 | 0.9595 | 0.9532 | 0.9448 | 0.9489 | 100.0000 | 119.0000 |
| hf_chinese_conversation_spam_holdout | v6_fusion_multisource | 5412 | 3258 | 2154 | 0.9586 | 0.9662 | 0.9285 | 0.9470 | 70.0000 | 154.0000 |
| hf_chinese_conversation_spam_holdout | v7_bridge_main_multisource | 5412 | 3258 | 2154 | 0.9479 | 0.9761 | 0.8909 | 0.9316 | 47.0000 | 235.0000 |
| adversarial | v3_main_only | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v5_main_only_fusion | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v6_csn_aug_multisource | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v6_fusion_multisource | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v7_bridge_main_multisource | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v3_main_only | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v5_main_only_fusion | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v6_csn_aug_multisource | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v6_fusion_multisource | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v7_bridge_main_multisource | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
