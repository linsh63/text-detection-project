# v6 跨来源适配实验

本实验从外部二分类数据集中切出一部分作为适配训练数据，剩余部分作为外部保留测试集，用于验证是否能提升跨来源泛化。

- v6 融合参数：risk_bonus=0.30, threshold=0.30
- v7 桥接阈值：threshold=0.30
- v6 训练数据：主训练集 + FBS/HF 外部适配训练集 + 关键词增强样本。
- v7 桥接策略：`max(v5_main_only_score, v6_multisource_score)`，用于兼顾主数据集和外部数据集。
- 外部 holdout 没有进入训练或阈值选择。

## 外部数据切分

| dataset | total_rows | adapt_train_rows | adapt_fit_rows | adapt_valid_rows | holdout_rows | holdout_spam | holdout_normal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fbs_mixed | 10000 | 2000 | 1600 | 400 | 8000 | 4000 | 4000 |
| hf_chinese_spam_10000 | 9941 | 1988 | 1590 | 398 | 7953 | 3986 | 3967 |
| hf_chinese_conversation_spam | 7731 | 1546 | 1236 | 310 | 6185 | 2462 | 3723 |

## 每个数据集最优结果

| dataset | name | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adversarial | v3_main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| fbs_mixed_holdout | v6_csn_aug_multisource | 0.9829 | 0.9842 | 0.9815 | 0.9829 | 63.0000 | 74.0000 |
| hf_chinese_conversation_spam_holdout | v6_csn_aug_multisource | 0.9521 | 0.9513 | 0.9273 | 0.9391 | 117.0000 | 179.0000 |
| hf_chinese_spam_10000_holdout | v6_csn_aug_multisource | 0.8779 | 0.8970 | 0.8545 | 0.8752 | 391.0000 | 580.0000 |
| keyword_challenge | v3_main_only | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| main_holdout | v5_main_only_fusion | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |

## 完整指标

| dataset | name | n_samples | n_normal | n_spam | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_holdout | v3_main_only | 6000 | 5396 | 604 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84.0000 | 15.0000 |
| main_holdout | v5_main_only_fusion | 6000 | 5396 | 604 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |
| main_holdout | v6_csn_aug_multisource | 6000 | 5396 | 604 | 0.9795 | 0.8470 | 0.9719 | 0.9052 | 106.0000 | 17.0000 |
| main_holdout | v6_fusion_multisource | 6000 | 5396 | 604 | 0.9865 | 0.9197 | 0.9487 | 0.9340 | 50.0000 | 31.0000 |
| main_holdout | v7_bridge_main_multisource | 6000 | 5396 | 604 | 0.9877 | 0.9128 | 0.9702 | 0.9406 | 56.0000 | 18.0000 |
| fbs_mixed_holdout | v3_main_only | 8000 | 4000 | 4000 | 0.9131 | 0.9847 | 0.8393 | 0.9062 | 52.0000 | 643.0000 |
| fbs_mixed_holdout | v5_main_only_fusion | 8000 | 4000 | 4000 | 0.8385 | 0.9938 | 0.6813 | 0.8084 | 17.0000 | 1275.0000 |
| fbs_mixed_holdout | v6_csn_aug_multisource | 8000 | 4000 | 4000 | 0.9829 | 0.9842 | 0.9815 | 0.9829 | 63.0000 | 74.0000 |
| fbs_mixed_holdout | v6_fusion_multisource | 8000 | 4000 | 4000 | 0.9790 | 0.9925 | 0.9653 | 0.9787 | 29.0000 | 139.0000 |
| fbs_mixed_holdout | v7_bridge_main_multisource | 8000 | 4000 | 4000 | 0.9782 | 0.9905 | 0.9657 | 0.9780 | 37.0000 | 137.0000 |
| hf_chinese_spam_10000_holdout | v3_main_only | 7953 | 3967 | 3986 | 0.5340 | 0.8286 | 0.0886 | 0.1600 | 73.0000 | 3633.0000 |
| hf_chinese_spam_10000_holdout | v5_main_only_fusion | 7953 | 3967 | 3986 | 0.5162 | 0.8557 | 0.0416 | 0.0794 | 28.0000 | 3820.0000 |
| hf_chinese_spam_10000_holdout | v6_csn_aug_multisource | 7953 | 3967 | 3986 | 0.8779 | 0.8970 | 0.8545 | 0.8752 | 391.0000 | 580.0000 |
| hf_chinese_spam_10000_holdout | v6_fusion_multisource | 7953 | 3967 | 3986 | 0.8749 | 0.9410 | 0.8006 | 0.8651 | 200.0000 | 795.0000 |
| hf_chinese_spam_10000_holdout | v7_bridge_main_multisource | 7953 | 3967 | 3986 | 0.8739 | 0.9380 | 0.8013 | 0.8643 | 211.0000 | 792.0000 |
| hf_chinese_conversation_spam_holdout | v3_main_only | 6185 | 3723 | 2462 | 0.7452 | 0.9784 | 0.3680 | 0.5348 | 20.0000 | 1556.0000 |
| hf_chinese_conversation_spam_holdout | v5_main_only_fusion | 6185 | 3723 | 2462 | 0.7154 | 0.9944 | 0.2868 | 0.4451 | 4.0000 | 1756.0000 |
| hf_chinese_conversation_spam_holdout | v6_csn_aug_multisource | 6185 | 3723 | 2462 | 0.9521 | 0.9513 | 0.9273 | 0.9391 | 117.0000 | 179.0000 |
| hf_chinese_conversation_spam_holdout | v6_fusion_multisource | 6185 | 3723 | 2462 | 0.9415 | 0.9773 | 0.8733 | 0.9224 | 50.0000 | 312.0000 |
| hf_chinese_conversation_spam_holdout | v7_bridge_main_multisource | 6185 | 3723 | 2462 | 0.9416 | 0.9760 | 0.8749 | 0.9227 | 53.0000 | 308.0000 |
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
