# 多数据集 v4/v5 验证

本实验用主数据集训练模型，再在主数据集保留测试集和外部/挑战评测集上统一比较 v4 与 v5。

- v4 固定参数：risk_bonus=0.10, threshold=0.35
- v5 验证集选择参数：risk_bonus=0.40, threshold=0.40
- v5 公式：`max(v1_baseline_score, v3_csn_score) + risk_bonus * bad_case_risk_score`
- 结论：建议选用 v5：多数据集平均 Spam F1 不低于 v4，且平均漏检没有增加。

## 指标结果

| dataset | name | n_samples | n_normal | n_spam | risk_bonus | threshold | accuracy | precision_spam | recall_spam | f1_spam | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_holdout | v4_bad_case_valid_tuned | 6000 | 5396 | 604 | 0.1000 | 0.3500 | 0.9902 | 0.9534 | 0.9487 | 0.9510 | 28.0000 | 31.0000 |
| main_holdout | v5_max_score_fusion | 6000 | 5396 | 604 | 0.4000 | 0.4000 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27.0000 | 23.0000 |
| fbs_mixed | v4_bad_case_valid_tuned | 10000 | 5000 | 5000 | 0.1000 | 0.3500 | 0.8402 | 0.9948 | 0.6840 | 0.8106 | 18.0000 | 1580.0000 |
| fbs_mixed | v5_max_score_fusion | 10000 | 5000 | 5000 | 0.4000 | 0.4000 | 0.8409 | 0.9939 | 0.6860 | 0.8117 | 21.0000 | 1570.0000 |
| adversarial | v4_bad_case_valid_tuned | 131 | 0 | 131 | 0.1000 | 0.3500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| adversarial | v5_max_score_fusion | 131 | 0 | 131 | 0.4000 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v4_bad_case_valid_tuned | 270 | 0 | 270 | 0.1000 | 0.3500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| keyword_challenge | v5_max_score_fusion | 270 | 0 | 270 | 0.4000 | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

## v5 相对 v4 变化

| dataset | f1_delta | recall_delta | fp_delta | fn_delta |
| --- | --- | --- | --- | --- |
| main_holdout | 0.0077 | 0.0132 | -1.0000 | -8.0000 |
| fbs_mixed | 0.0011 | 0.0020 | 3.0000 | -10.0000 |
| adversarial | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| keyword_challenge | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
