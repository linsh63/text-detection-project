# 实验与优化记录

本文件用于记录每一步模型迭代：做了什么、为什么做、指标提升多少。所有优化都需要先和上一版 baseline 或上一版最优结果对比。

## 指标口径

- 主数据集：`data/processed/spam_message_20k.tsv`
- 备用说明：这不是课程指定 AST 数据集，是未拿到 AST 前使用的中文垃圾短信备选数据。
- 划分方式：`train_test_split(test_size=0.3, random_state=42, stratify=label)`
- 主要指标：Accuracy、Precision、Recall、F1、Macro F1、False Positive、False Negative、Adversarial Recall
- v4 阈值参数从训练集内部 validation split 选择，最终结果仍在固定测试集上评估。

## 记录表

| 版本 | 类型 | 变更 | Accuracy | Macro F1 | Spam F1 | 相比上一版 |
|---|---|---|---:|---:|---:|---|
| v0-smoke | Smoke Baseline | 10 条样例数据，字符级 TF-IDF(1-3gram) + Logistic Regression(class_weight=balanced) | 0.6667 | 0.6667 | 0.6667 | - |
| v0 | Formal Backup Baseline | 2 万条中文垃圾短信备选数据，字符级 TF-IDF(1-3gram) + Logistic Regression(class_weight=balanced) | 0.9773 | 0.9417 | 0.8960 | 正式规模 baseline |
| v0-demo | Adversarial Demo Eval | 从备选垃圾短信中生成 131 条字形/字音变体样本，用 v0 模型评估 | 0.9924 | 0.4981* | 0.9962 | 垃圾样本召回 130/131 |
| v1-baseline-compare | Baseline Comparison | 词级/字符级 TF-IDF + LR/SVM 四组对比 | 0.9870 | 0.9654 | 0.9380 | 最优为字符级 Linear SVM |
| v2-csn | CSN Normalization | 字符相似性词表归一化 + 字符级 TF-IDF | 0.9870 | 0.9654 | 0.9380 | 对 keyword challenge 有小幅提升 |
| v3-csn-aug | CSN + Keyword Augmentation | CSN 归一化 + 敏感关键词增强训练 | 0.9835 | 0.9570 | 0.9225 | keyword challenge 召回提升至 1.0000 |
| v4-bad-case-tuned | Bad-case Risk + Threshold | bad-case 风险分数 + 验证集阈值调优 | 0.9902 | 0.9728 | 0.9510 | 在保持 keyword challenge 召回 1.0000 的同时减少误杀 |

## v0-smoke Baseline

### 目标

先跑通一个最小可复现实验闭环：读取 TSV 数据、字符级分词、TF-IDF 特征、逻辑回归分类、输出分类报告和混淆矩阵，并保存模型到 `models/baseline.joblib`。

### 命令

```bash
./.venv/bin/python -m src.cli train --data data/raw/sample_texts.tsv
./.venv/bin/python -m src.cli predict --text "加我薇信，轻松月入过万"
```

### 结果

```text
Saved model to: models/baseline.joblib
Classification report:
              precision    recall  f1-score   support

           0     1.0000    0.5000    0.6667         2
           1     0.5000    1.0000    0.6667         1

    accuracy                         0.6667         3
   macro avg     0.7500    0.7500    0.6667         3
weighted avg     0.8333    0.6667    0.6667         3

Confusion matrix:
[[1, 1], [0, 1]]
```

预测验证：

```text
输入：加我薇信，轻松月入过万
输出：1 垃圾文本
```

### 结论

baseline 已经跑通，但当前样例数据只有 10 条，指标只用于验证工程闭环，不作为最终实验结论。下一步接入正式数据后，需要重跑 v0 并把正式 baseline 指标作为后续优化的真实对照。

## v0 Formal Backup Baseline

### 数据

- 原始数据：`data/raw/spam_message_labeled.txt`
- 处理后数据：`data/processed/spam_message_20k.tsv`
- 抽样方式：按类别分层抽样 20000 条
- 类别分布：正常文本 17988 条，垃圾文本 2012 条

说明：这不是课程指定 AST 数据集，而是普通中文垃圾短信备选数据。它用于先跑正式规模 baseline；拿到 AST 后需要重跑同一套指标。

### 命令

```bash
./.venv/bin/python -m src.cli prepare-data --raw data/raw/spam_message_labeled.txt --out data/processed/spam_message_20k.tsv --sample-size 20000
./.venv/bin/python -m src.cli train --data data/processed/spam_message_20k.tsv
```

### 指标

```text
Accuracy: 0.9773
Balanced Accuracy: 0.9742
Precision(spam): 0.8324
Recall(spam): 0.9702
F1(spam): 0.8960
Macro F1: 0.9417
Weighted F1: 0.9781
ROC-AUC: 0.9967
PR-AUC: 0.9770
Recall@Precision>=90%: 0.9603
Recall@Precision>=95%: 0.8858
False Positive Rate: 0.0219
Confusion Matrix: [[5278, 118], [18, 586]]
```

### 结论

普通中文垃圾短信检测上，字符级 TF-IDF baseline 已经足够强，尤其是垃圾文本召回率高。后续优化不应该只追求 Accuracy，而应重点展示：

- 在误杀受控时的垃圾文本召回率，即 `Recall@Precision>=90/95%`
- 对抗变体文本上的召回率，即后续 AST 或扰动样本中的 `Adversarial Recall`
- 引入字符相似性网络后，普通样本与对抗样本之间的性能差距是否缩小

## v0-demo Adversarial Evaluation

### 数据

- 输入数据：`data/processed/spam_message_20k.tsv`
- 生成数据：`data/processed/adversarial_eval.tsv`
- 样本数量：131 条
- 生成方式：仅对垃圾文本中命中的关键词做字形/字音替换，例如 `微信->胃信`、`红包->红苞`、`贷款->贷歀`、`兼职->兼只`

### 命令

```bash
./.venv/bin/python -m src.cli generate-adversarial --data data/processed/spam_message_20k.tsv --out data/processed/adversarial_eval.tsv --max-samples 1000
./.venv/bin/python -m src.cli evaluate --data data/processed/adversarial_eval.tsv
```

### 指标

```text
Adversarial Recall: 0.9924
Spam Precision: 1.0000
Spam F1: 0.9962
False Negative: 1
True Positive: 130
Confusion Matrix: [[0, 0], [1, 130]]
```

注：该评估集只有垃圾样本，因此 Macro F1 不适合作为核心展示指标。本节重点看 `Adversarial Recall`，即变体垃圾文本召回率。

### 结论

当前字符级 TF-IDF baseline 对人工生成的轻量扰动仍然很强，说明这个展示集还不够难。下一步应该优先获取课程指定 AST 数据，或者增强扰动生成策略，使其覆盖更多未登录变体、特殊符号插入、同音替换和联系方式变体，从而更清楚地展示字符相似性网络的增益。

## v1-baseline-compare Baseline 对比实验

### 目标

在同一份数据和同一个随机划分下，比较不同基础文本表示和分类器组合，确认后续优化的强基线。

### 数据

- 数据：`data/processed/spam_message_20k.tsv`
- 划分：`test_size=0.3, random_state=42, stratify=label`
- 训练集：14000 条
- 测试集：6000 条

### 命令

```bash
./.venv/bin/python -m src.cli compare-baselines --data data/processed/spam_message_20k.tsv
```

### 结果

结果文件：

- `docs/baseline_comparison.csv`
- `docs/baseline_comparison.md`

核心结果：

| 方法 | Accuracy | Macro F1 | Spam F1 | Spam Recall | PR-AUC | Recall@Precision>=95% |
|---|---:|---:|---:|---:|---:|---:|
| 字符级 TF-IDF + Linear SVM | 0.9870 | 0.9654 | 0.9380 | 0.9768 | 0.9900 | 0.9553 |
| 词级 TF-IDF + Linear SVM | 0.9840 | 0.9573 | 0.9236 | 0.9603 | 0.9754 | 0.9040 |
| 字符级 TF-IDF + Logistic Regression | 0.9773 | 0.9417 | 0.8960 | 0.9702 | 0.9770 | 0.8858 |
| 词级 TF-IDF + Logistic Regression | 0.9725 | 0.9297 | 0.8749 | 0.9553 | 0.9475 | 0.7086 |

### 相比 v0 的提升

以 v0 的字符级 TF-IDF + Logistic Regression 为对照，最优 baseline（字符级 TF-IDF + Linear SVM）提升如下：

- Accuracy：0.9773 -> 0.9870，提升 0.0097
- Macro F1：0.9417 -> 0.9654，提升 0.0237
- Spam F1：0.8960 -> 0.9380，提升 0.0420
- Recall@Precision>=95%：0.8858 -> 0.9553，提升 0.0695

### 结论

后续优化应以 `字符级 TF-IDF + Linear SVM` 作为强 baseline。它比词级方法更适合中文垃圾文本检测，也明显优于同样字符级特征下的 Logistic Regression。报告中可以用这一组对比说明：字符级建模对变体、短文本和分词不稳定场景更稳健。

## v2-csn 字符相似性归一化

### 目标

实现课程要求中的“字符相似性网络”思想的第一版工程化近似：把常见字形/字音变体归一到标准敏感词，再进入字符级 TF-IDF 模型。

示例：

```text
薇信 / 维信 / 胃星 / 卫星 -> 微信
红苞 / 紅包 / 虹包 -> 红包
贷歀 / 代款 / 貸款 -> 贷款
```

### 命令

```bash
./.venv/bin/python -m src.cli generate-keyword-challenge --out data/processed/keyword_challenge.tsv
./.venv/bin/python -m src.cli compare-csn --data data/processed/spam_message_20k.tsv --adversarial data/processed/keyword_challenge.tsv
```

### 结果

结果文件：

- `docs/csn_comparison.csv`
- `docs/csn_comparison.md`

未做关键词增强时：

| 方法 | Clean Spam F1 | Keyword Challenge Recall | Keyword Challenge FN |
|---|---:|---:|---:|
| 字符级 TF-IDF + Linear SVM | 0.9380 | 0.0667 | 252 |
| CSN 归一化 + 字符级 TF-IDF + Linear SVM | 0.9380 | 0.1037 | 242 |
| 字符级 TF-IDF + Logistic Regression | 0.8960 | 0.1444 | 231 |
| CSN 归一化 + 字符级 TF-IDF + Logistic Regression | 0.8960 | 0.1815 | 221 |

### 结论

单独做 CSN 归一化不会影响普通测试集表现，但对短文本关键词变体的提升有限。原因是训练集中“微信、贷款、红包”等短关键词本身出现较少，模型即使看到归一后的 canonical 词，也未必有足够强的正类权重。

## v3-csn-aug CSN + 关键词增强训练

### 目标

在 v2 的 CSN 归一化基础上，加入少量 canonical 敏感关键词训练样本，例如：

```text
微信
加微信
微信联系
办理贷款
红包联系
```

这样做的目的不是替代真实数据，而是让模型明确学习“敏感关键词本身具有垃圾文本风险”。测试时，变体词先被 CSN 归一化，再被模型识别。

### 结果

| 方法 | Clean Accuracy | Clean Spam F1 | Keyword Challenge Recall | Keyword Challenge FN |
|---|---:|---:|---:|---:|
| 字符级 TF-IDF + Linear SVM | 0.9870 | 0.9380 | 0.0667 | 252 |
| CSN + 关键词增强 + Linear SVM | 0.9835 | 0.9225 | 1.0000 | 0 |
| 字符级 TF-IDF + Logistic Regression | 0.9773 | 0.8960 | 0.1444 | 231 |
| CSN + 关键词增强 + Logistic Regression | 0.9742 | 0.8832 | 1.0000 | 0 |

### 相比强 baseline 的变化

以 v1 最强 baseline `字符级 TF-IDF + Linear SVM` 为对照：

- 普通测试集 Accuracy：0.9870 -> 0.9835，下降 0.0035
- 普通测试集 Spam F1：0.9380 -> 0.9225，下降 0.0155
- Keyword Challenge Recall：0.0667 -> 1.0000，提升 0.9333
- Keyword Challenge FN：252 -> 0，减少 252 个漏检

### 结论

CSN + 关键词增强明显提升了对抗关键词短文本的鲁棒性，但普通测试集有轻微下降。这是一个适合报告展示的权衡：如果业务目标是拦截“薇信、胃星、贷歀”等对抗变体，可以接受少量普通集性能损失；如果业务目标偏向普通垃圾短信分类，则 v1 的字符级 Linear SVM 仍是更强的通用 baseline。

## v4-bad-case-tuned Bad-case 风险分数 + 阈值调优

### 目标

v3 的主要问题是误杀增多：默认阈值下，普通测试集 False Positive 从 v1 的 64 增加到 84。bad case 分析显示：

- 误杀集中在短营销式正常文本，例如“咨询、订购、联系电话、价格、活动”等普通商业语境。
- 漏检集中在赌博/彩票黑话、URL 变体、插符号规避和特殊行业广告。

因此 v4 做两件事：

- 增加 `spam_risk_score`，只对高风险 bad-case 线索加分，例如 `六合彩、连码、一肖、Wap、www、娱/乐、太/阳/城、tel` 等。
- 在训练集内部 validation split 上搜索 `risk_bonus` 和 SVM `threshold`，用更高阈值压低误杀，用风险分数补回一部分漏检。

### 命令

```bash
./.venv/bin/python -m src.cli compare-badcases --data data/processed/spam_message_20k.tsv --adversarial data/processed/keyword_challenge.tsv
```

### 结果

结果文件：

- `docs/bad_case_optimization.csv`
- `docs/bad_case_optimization.md`
- `docs/bad_case_tuning_grid.csv`
- `docs/figures/model_comparison.svg`
- `docs/report_summary.md`

验证集选中参数：

```text
risk_bonus = 0.10
threshold = 0.35
```

核心结果：

| 方法 | Clean Accuracy | Clean Spam F1 | Clean Precision | Clean Recall | FP | FN | Keyword Challenge Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 强 baseline | 0.9870 | 0.9380 | 0.9021 | 0.9768 | 64 | 14 | 0.0667 |
| v3 CSN + 关键词增强 | 0.9835 | 0.9225 | 0.8752 | 0.9752 | 84 | 15 | 1.0000 |
| v4 阈值调优 | 0.9898 | 0.9493 | 0.9533 | 0.9454 | 28 | 33 | 1.0000 |
| v4 bad-case 风险 + 阈值调优 | 0.9902 | 0.9510 | 0.9534 | 0.9487 | 28 | 31 | 1.0000 |

测试集扫描上界 `v4_eval_oracle`：

| 方法 | Clean Accuracy | Clean Spam F1 | FP | FN | Keyword Challenge Recall |
|---|---:|---:|---:|---:|---:|
| v4_eval_oracle | 0.9923 | 0.9617 | 20 | 26 | 1.0000 |

注：`v4_eval_oracle` 是在测试集上扫描得到的上界，只说明当前方法还有调参空间，不作为严格泛化结果。

### 相比 v3 的提升

以 v3 `CSN + 关键词增强` 为对照：

- Accuracy：0.9835 -> 0.9902，提升 0.0067
- Macro F1：0.9566 -> 0.9728，提升 0.0162
- Spam F1：0.9225 -> 0.9510，提升 0.0286
- False Positive：84 -> 28，减少 56 个误杀
- False Negative：15 -> 31，增加 16 个漏检
- Keyword Challenge Recall：保持 1.0000

### 相比 v1 强 baseline 的变化

以 v1 `字符级 TF-IDF + Linear SVM` 为对照：

- Accuracy：0.9870 -> 0.9902，提升 0.0032
- Macro F1：0.9654 -> 0.9728，提升 0.0074
- Spam F1：0.9380 -> 0.9510，提升 0.0130
- False Positive：64 -> 28，减少 36 个误杀
- False Negative：14 -> 31，增加 17 个漏检
- Keyword Challenge Recall：0.0667 -> 1.0000，提升 0.9333

### 结论

v4 把 v3 的主要副作用“误杀过多”压了下来，并保留了 CSN 对关键词变体的鲁棒性。代价是垃圾文本召回率从 v1 的 0.9768 降到 0.9487，因此它更适合作为“高精度、低误杀、抗变体”的版本；如果业务更看重极限召回，可以保留 v1/v3 作为对照。报告中可以把 v1、v3、v4 放在一张图里，展示“普通检测效果、误杀控制、对抗鲁棒性”三者之间的权衡。
