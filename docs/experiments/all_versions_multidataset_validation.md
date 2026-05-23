# 全版本多数据集验证

本实验用主数据集训练 v0-v5，再在主测试集、外部数据集和对抗挑战集上统一评估。

- v4 固定参数：risk_bonus=0.10, threshold=0.35
- v5 验证集选择参数：risk_bonus=0.40, threshold=0.40
- v5 在 3/6 个评测集上达到该评测集最高 Spam F1。

## 结果解读

- 主测试集最优：`v5_max_score_fusion`，Spam F1=0.9587，Recall=0.9619，FN=23。
- 跨来源二分类集最优：
  - `fbs_mixed`：`v1_strong_baseline_default`，Spam F1=0.9085，Recall=0.8426，FN=787。
  - `hf_chinese_conversation_spam`：`v3_csn_aug_default`，Spam F1=0.5405，Recall=0.3737，FN=1927。
  - `hf_chinese_spam_10000`：`v3_csn_aug_default`，Spam F1=0.1593，Recall=0.0881，FN=4543。
- 对抗/关键词挑战集是单类垃圾样本，重点看 Recall 和 FN；v3 之后的模型已经能覆盖短关键词变体。
- 结论：v5 适合作为主数据集上的最终版本；如果强调跨来源泛化，v3 目前更稳，v4/v5 的高阈值会带来明显漏检。

## 每个数据集最优结果

| dataset | name | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adversarial | v1_strong_baseline_default | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| fbs_mixed | v1_strong_baseline_default | 0.9151 | 0.9855 | 0.8426 | 0.9085 | 62.0000 | 787.0000 |
| hf_chinese_conversation_spam | v3_csn_aug_default | 0.7471 | 0.9762 | 0.3737 | 0.5405 | 28.0000 | 1927.0000 |
| hf_chinese_spam_10000 | v3_csn_aug_default | 0.5341 | 0.8314 | 0.0881 | 0.1593 | 89.0000 | 4543.0000 |
| keyword_challenge | v3_csn_aug_default | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| main_holdout | v5_max_score_fusion | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |

## 完整指标

| dataset | name | n_samples | n_normal | n_spam | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_holdout | v0_char_logreg_default | 6000 | 5396 | 604 | 0.9773 | 0.8324 | 0.9702 | 0.8960 | 118.0000 | 18.0000 |
| main_holdout | v1_strong_baseline_default | 6000 | 5396 | 604 | 0.9870 | 0.9021 | 0.9768 | 0.9380 | 64.0000 | 14.0000 |
| main_holdout | v2_csn_default | 6000 | 5396 | 604 | 0.9870 | 0.9021 | 0.9768 | 0.9380 | 64.0000 | 14.0000 |
| main_holdout | v3_csn_aug_default | 6000 | 5396 | 604 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84.0000 | 15.0000 |
| main_holdout | v4_bad_case_valid_tuned | 6000 | 5396 | 604 | 0.9902 | 0.9534 | 0.9487 | 0.9510 | 28.0000 | 31.0000 |
| main_holdout | v5_max_score_fusion | 6000 | 5396 | 604 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |
| fbs_mixed | v0_char_logreg_default | 10000 | 5000 | 5000 | 0.9111 | 0.9737 | 0.8450 | 0.9048 | 114.0000 | 775.0000 |
| fbs_mixed | v1_strong_baseline_default | 10000 | 5000 | 5000 | 0.9151 | 0.9855 | 0.8426 | 0.9085 | 62.0000 | 787.0000 |
| fbs_mixed | v2_csn_default | 10000 | 5000 | 5000 | 0.9149 | 0.9857 | 0.8420 | 0.9082 | 61.0000 | 790.0000 |
| fbs_mixed | v3_csn_aug_default | 10000 | 5000 | 5000 | 0.9146 | 0.9834 | 0.8434 | 0.9081 | 71.0000 | 783.0000 |
| fbs_mixed | v4_bad_case_valid_tuned | 10000 | 5000 | 5000 | 0.8402 | 0.9948 | 0.6840 | 0.8106 | 18.0000 | 1580.0000 |
| fbs_mixed | v5_max_score_fusion | 10000 | 5000 | 5000 | 0.8409 | 0.9939 | 0.6860 | 0.8117 | 21.0000 | 1570.0000 |
| hf_chinese_spam_10000 | v0_char_logreg_default | 9941 | 4959 | 4982 | 0.4969 | 0.3797 | 0.0060 | 0.0119 | 49.0000 | 4952.0000 |
| hf_chinese_spam_10000 | v1_strong_baseline_default | 9941 | 4959 | 4982 | 0.4990 | 0.5139 | 0.0074 | 0.0146 | 35.0000 | 4945.0000 |
| hf_chinese_spam_10000 | v2_csn_default | 9941 | 4959 | 4982 | 0.4990 | 0.5139 | 0.0074 | 0.0146 | 35.0000 | 4945.0000 |
| hf_chinese_spam_10000 | v3_csn_aug_default | 9941 | 4959 | 4982 | 0.5341 | 0.8314 | 0.0881 | 0.1593 | 89.0000 | 4543.0000 |
| hf_chinese_spam_10000 | v4_bad_case_valid_tuned | 9941 | 4959 | 4982 | 0.5135 | 0.8318 | 0.0367 | 0.0704 | 37.0000 | 4799.0000 |
| hf_chinese_spam_10000 | v5_max_score_fusion | 9941 | 4959 | 4982 | 0.5157 | 0.8529 | 0.0407 | 0.0778 | 35.0000 | 4779.0000 |
| hf_chinese_conversation_spam | v0_char_logreg_default | 7731 | 4654 | 3077 | 0.7386 | 0.9818 | 0.3497 | 0.5157 | 20.0000 | 2001.0000 |
| hf_chinese_conversation_spam | v1_strong_baseline_default | 7731 | 4654 | 3077 | 0.7335 | 0.9838 | 0.3360 | 0.5010 | 17.0000 | 2043.0000 |
| hf_chinese_conversation_spam | v2_csn_default | 7731 | 4654 | 3077 | 0.7317 | 0.9836 | 0.3315 | 0.4959 | 17.0000 | 2057.0000 |
| hf_chinese_conversation_spam | v3_csn_aug_default | 7731 | 4654 | 3077 | 0.7471 | 0.9762 | 0.3737 | 0.5405 | 28.0000 | 1927.0000 |
| hf_chinese_conversation_spam | v4_bad_case_valid_tuned | 7731 | 4654 | 3077 | 0.6850 | 0.9893 | 0.2109 | 0.3477 | 7.0000 | 2428.0000 |
| hf_chinese_conversation_spam | v5_max_score_fusion | 7731 | 4654 | 3077 | 0.7149 | 0.9932 | 0.2857 | 0.4437 | 6.0000 | 2198.0000 |
| adversarial | v0_char_logreg_default | 131 | 0 | 131 | 0.9924 | 1.0000 | 0.9924 | 0.9962 | 0.0000 | 1.0000 |
| adversarial | v1_strong_baseline_default | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v2_csn_default | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v3_csn_aug_default | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v4_bad_case_valid_tuned | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v5_max_score_fusion | 131 | 0 | 131 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v0_char_logreg_default | 270 | 0 | 270 | 0.1444 | 1.0000 | 0.1444 | 0.2524 | 0.0000 | 231.0000 |
| keyword_challenge | v1_strong_baseline_default | 270 | 0 | 270 | 0.0667 | 1.0000 | 0.0667 | 0.1250 | 0.0000 | 252.0000 |
| keyword_challenge | v2_csn_default | 270 | 0 | 270 | 0.1037 | 1.0000 | 0.1037 | 0.1879 | 0.0000 | 242.0000 |
| keyword_challenge | v3_csn_aug_default | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v4_bad_case_valid_tuned | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v5_max_score_fusion | 270 | 0 | 270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
