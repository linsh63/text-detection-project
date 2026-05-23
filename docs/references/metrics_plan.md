# 指标设计

本项目指标分为两类：常规分类指标用于保证实验规范，展示型指标用于突出字符相似性网络在对抗垃圾文本上的效果。

## 常规指标

| 指标 | 用途 |
|---|---|
| Accuracy | 整体分类正确率，直观但受类别不平衡影响 |
| Precision(spam) | 预测为垃圾文本的样本中有多少是真的垃圾文本 |
| Recall(spam) | 真实垃圾文本中有多少被识别出来，是内容风控场景重点 |
| F1(spam) | 垃圾文本 precision 和 recall 的折中 |
| Macro F1 | 两类平均后的 F1，适合类别不平衡场景 |
| Weighted F1 | 按类别样本数加权后的 F1 |
| Balanced Accuracy | 对每个类别召回率求平均，缓解类别不平衡影响 |
| Confusion Matrix | 展示 TP、FP、TN、FN，用于错误分析 |
| ROC-AUC | 衡量分类器整体排序能力 |
| PR-AUC | 在垃圾文本识别这类偏不平衡任务中更敏感 |

## 展示型指标

| 指标 | 展示意义 |
|---|---|
| Spam Recall@Precision>=90% | 在误杀率可控时，能拦截多少垃圾文本 |
| Spam Recall@Precision>=95% | 更严格业务约束下的拦截能力 |
| False Positive Rate | 正常文本被误判为垃圾文本的比例，便于讨论用户体验 |
| Adversarial Recall | 对“薇信、胃星、维信、卫星”等变体垃圾文本的召回率 |
| Clean-vs-Adversarial Gap | 普通测试集与对抗测试集上的性能差距，展示鲁棒性 |
| Variant Hit Cases | 模型成功识别的典型变体案例，适合报告截图和表格 |
| Threshold Curve | 展示字符相似度阈值变化对 Precision/Recall/F1 的影响 |
| Bad-case FP/FN Count | 直接展示误杀和漏检数量，方便解释优化代价 |
| Risk-score Ablation | 对比只调阈值和加入 bad-case 风险分数后的差异 |

## 当前采用口径

baseline 先输出以下指标：

- Accuracy
- Precision(spam)
- Recall(spam)
- F1(spam)
- Macro F1
- Weighted F1
- Balanced Accuracy
- ROC-AUC
- PR-AUC
- Spam Recall@Precision>=90%
- Spam Recall@Precision>=95%
- False Positive Rate
- Confusion Matrix

后续字符相似性网络优化后，追加：

- Adversarial Recall
- Clean-vs-Adversarial Gap
- Threshold Curve
- Variant Hit Cases
- Clean Spam F1 drop
- Keyword Challenge false negatives
- Bad-case risk bonus
- Tuned decision threshold

## 当前可运行命令

普通测试集评估：

```bash
./.venv/bin/python -m src.cli train --data data/processed/spam_message_20k.tsv
./.venv/bin/python -m src.cli compare-baselines --data data/processed/spam_message_20k.tsv
```

展示型对抗评估：

```bash
./.venv/bin/python -m src.cli generate-adversarial --data data/processed/spam_message_20k.tsv --out data/processed/adversarial_eval.tsv
./.venv/bin/python -m src.cli evaluate --data data/processed/adversarial_eval.tsv
./.venv/bin/python -m src.cli generate-keyword-challenge --out data/processed/keyword_challenge.tsv
./.venv/bin/python -m src.cli compare-csn --data data/processed/spam_message_20k.tsv --adversarial data/processed/keyword_challenge.tsv
./.venv/bin/python -m src.cli compare-badcases --data data/processed/spam_message_20k.tsv --adversarial data/processed/keyword_challenge.tsv
./.venv/bin/python -m src.cli plot-comparison
```
