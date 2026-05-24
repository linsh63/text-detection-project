# v5 分数融合实验

本实验在独立分支上测试新的融合方案，不影响 `main` 当前版本。

- 验证集选择参数：risk_bonus=0.40, threshold=0.40
- 融合公式：`max(v1_baseline_score, v3_csn_score) + risk_bonus * bad_case_risk_score`

## 固定测试集结果

| name | description | risk_bonus | threshold | clean_accuracy | clean_precision_spam | clean_recall_spam | clean_f1_spam | clean_false_positive | clean_false_negative | adv_recall_spam |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_strong_baseline_default | 字符级 TF-IDF + Linear SVM，默认阈值 | 0.0000 | 0.0000 | 0.9870 | 0.9021 | 0.9768 | 0.9380 | 64.0000 | 14.0000 | 0.0667 |
| v3_csn_aug_default | CSN 归一化 + 关键词增强，默认阈值 | 0.0000 | 0.0000 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84.0000 | 15.0000 | 1.0000 |
| v4_bad_case_valid_tuned | CSN 关键词增强 + bad-case 风险分数 + 验证集阈值调优 | 0.1000 | 0.3500 | 0.9902 | 0.9534 | 0.9487 | 0.9510 | 28.0000 | 31.0000 | 1.0000 |
| v5_max_score_fusion | max(v1 分数, v3 分数) + bad-case 风险分数 | 0.4000 | 0.4000 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 | 1.0000 |

## 多随机划分均值

| name | clean_accuracy | clean_f1_spam | clean_false_positive | clean_false_negative |
| --- | --- | --- | --- | --- |
| v4_bad_case_valid_tuned | 0.9906 | 0.9530 | 23.4286 | 32.8571 |
| v5_max_score_fusion | 0.9911 | 0.9558 | 23.4286 | 29.7143 |

## 初步结论

v5 在固定测试集上优于 v4：Spam F1 提升，漏检减少，并保持 keyword challenge 对抗召回为 1.0000。
多随机划分下 v5 平均 Spam F1 也略高于 v4，但提升幅度较小，说明这是一个可继续保留的候选优化，而不是压倒性的稳定提升。
