# v6 跨来源适配实验

本实验从外部二分类数据集中切出一部分作为适配训练数据，剩余部分作为外部保留测试集，用于验证是否能提升跨来源泛化。

- v6 融合参数：risk_bonus=0.30, threshold=0.25
- v7 桥接阈值：threshold=0.25
- v6 训练数据：主训练集 + FBS/HF 外部适配训练集 + 关键词增强样本。
- v7 桥接策略：`max(v5_main_only_score, v6_multisource_score)`，用于兼顾主数据集和外部数据集。
- 外部 holdout 没有进入训练或阈值选择。

## 外部数据切分

| dataset | total_rows | adapt_train_rows | adapt_fit_rows | adapt_valid_rows | holdout_rows | holdout_spam | holdout_normal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fbs_mixed | 10000 | 1000 | 800 | 200 | 9000 | 4500 | 4500 |
| hf_chinese_spam_10000 | 9941 | 994 | 795 | 199 | 8947 | 4484 | 4463 |
| hf_chinese_conversation_spam | 7731 | 773 | 618 | 155 | 6958 | 2769 | 4189 |

## 每个数据集最优结果

| dataset | name | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adversarial | v3_main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| fbs_mixed_holdout | v6_csn_aug_multisource | 0.9788 | 0.9834 | 0.9740 | 0.9787 | 74.0000 | 117.0000 |
| hf_chinese_conversation_spam_holdout | v6_csn_aug_multisource | 0.9411 | 0.9493 | 0.9000 | 0.9240 | 133.0000 | 277.0000 |
| hf_chinese_spam_10000_holdout | v6_csn_aug_multisource | 0.8582 | 0.8992 | 0.8075 | 0.8509 | 406.0000 | 863.0000 |
| keyword_challenge | v3_main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| main_holdout | v5_main_only_fusion | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |

## 完整指标

| dataset | name | n_samples | n_normal | n_spam | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_holdout | v3_main_only | 6000 | 5396 | 604 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84.0000 | 15.0000 |
| main_holdout | v5_main_only_fusion | 6000 | 5396 | 604 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |
| main_holdout | v6_csn_aug_multisource | 6000 | 5396 | 604 | 0.9810 | 0.8540 | 0.9785 | 0.9120 | 101.0000 | 13.0000 |
| main_holdout | v6_fusion_multisource | 6000 | 5396 | 604 | 0.9873 | 0.9138 | 0.9652 | 0.9388 | 55.0000 | 21.0000 |
| main_holdout | v7_bridge_main_multisource | 6000 | 5396 | 604 | 0.9872 | 0.9035 | 0.9768 | 0.9387 | 63.0000 | 14.0000 |
| fbs_mixed_holdout | v3_main_only | 9000 | 4500 | 4500 | 0.9144 | 0.9839 | 0.8427 | 0.9078 | 62.0000 | 708.0000 |
| fbs_mixed_holdout | v5_main_only_fusion | 9000 | 4500 | 4500 | 0.8406 | 0.9942 | 0.6851 | 0.8112 | 18.0000 | 1417.0000 |
| fbs_mixed_holdout | v6_csn_aug_multisource | 9000 | 4500 | 4500 | 0.9788 | 0.9834 | 0.9740 | 0.9787 | 74.0000 | 117.0000 |
| fbs_mixed_holdout | v6_fusion_multisource | 9000 | 4500 | 4500 | 0.9767 | 0.9924 | 0.9607 | 0.9763 | 33.0000 | 177.0000 |
| fbs_mixed_holdout | v7_bridge_main_multisource | 9000 | 4500 | 4500 | 0.9756 | 0.9897 | 0.9611 | 0.9752 | 45.0000 | 175.0000 |
| hf_chinese_spam_10000_holdout | v3_main_only | 8947 | 4463 | 4484 | 0.5338 | 0.8281 | 0.0881 | 0.1592 | 82.0000 | 4089.0000 |
| hf_chinese_spam_10000_holdout | v5_main_only_fusion | 8947 | 4463 | 4484 | 0.5159 | 0.8592 | 0.0408 | 0.0779 | 30.0000 | 4301.0000 |
| hf_chinese_spam_10000_holdout | v6_csn_aug_multisource | 8947 | 4463 | 4484 | 0.8582 | 0.8992 | 0.8075 | 0.8509 | 406.0000 | 863.0000 |
| hf_chinese_spam_10000_holdout | v6_fusion_multisource | 8947 | 4463 | 4484 | 0.8516 | 0.9357 | 0.7558 | 0.8362 | 233.0000 | 1095.0000 |
| hf_chinese_spam_10000_holdout | v7_bridge_main_multisource | 8947 | 4463 | 4484 | 0.8511 | 0.9339 | 0.7565 | 0.8359 | 240.0000 | 1092.0000 |
| hf_chinese_conversation_spam_holdout | v3_main_only | 6958 | 4189 | 2769 | 0.7463 | 0.9763 | 0.3716 | 0.5383 | 25.0000 | 1740.0000 |
| hf_chinese_conversation_spam_holdout | v5_main_only_fusion | 6958 | 4189 | 2769 | 0.7151 | 0.9925 | 0.2864 | 0.4445 | 6.0000 | 1976.0000 |
| hf_chinese_conversation_spam_holdout | v6_csn_aug_multisource | 6958 | 4189 | 2769 | 0.9411 | 0.9493 | 0.9000 | 0.9240 | 133.0000 | 277.0000 |
| hf_chinese_conversation_spam_holdout | v6_fusion_multisource | 6958 | 4189 | 2769 | 0.9326 | 0.9748 | 0.8527 | 0.9097 | 61.0000 | 408.0000 |
| hf_chinese_conversation_spam_holdout | v7_bridge_main_multisource | 6958 | 4189 | 2769 | 0.9325 | 0.9736 | 0.8534 | 0.9095 | 64.0000 | 406.0000 |
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
