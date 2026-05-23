# Bad-case 驱动阈值优化对比

- 验证集选择参数：risk_bonus=0.10, threshold=0.35
- `v4_eval_oracle` 是在测试集上扫描得到的上界，只用于分析，不作为严格泛化结果。

| name | description | risk_bonus | threshold | clean_accuracy | clean_precision_spam | clean_recall_spam | clean_f1_spam | clean_false_positive | clean_false_negative | adv_recall_spam | adv_false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_strong_baseline_default | 字符级 TF-IDF + Linear SVM，默认阈值 | 0.0000 | 0.0000 | 0.9870 | 0.9021 | 0.9768 | 0.9380 | 64.0000 | 14.0000 | 0.0667 | 252.0000 |
| v3_csn_aug_default | CSN 归一化 + 关键词增强，默认阈值 | 0.0000 | 0.0000 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84.0000 | 15.0000 | 1.0000 | 0.0000 |
| v4_threshold_only | CSN 关键词增强 + 验证集阈值调优 | 0.0000 | 0.3500 | 0.9898 | 0.9533 | 0.9454 | 0.9493 | 28.0000 | 33.0000 | 1.0000 | 0.0000 |
| v4_bad_case_valid_tuned | CSN 关键词增强 + bad-case 风险分数 + 验证集阈值调优 | 0.1000 | 0.3500 | 0.9902 | 0.9534 | 0.9487 | 0.9510 | 28.0000 | 31.0000 | 1.0000 | 0.0000 |
| v4_eval_oracle | CSN 关键词增强 + bad-case 风险分数 + 测试集扫描上界 | 0.4000 | 0.4500 | 0.9923 | 0.9666 | 0.9570 | 0.9617 | 20.0000 | 26.0000 | 1.0000 | 0.0000 |
