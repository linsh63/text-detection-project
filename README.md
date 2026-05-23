# 基于字符相似性网络的垃圾文本检测

本项目对应第四章“文本检测实践”，目标是完成一个可复现实验项目：

1. 文本清洗、分词、停用词处理
2. TF-IDF / Word2Vec 等文本特征基线
3. 字形、字音相似性建模
4. 字符相似性网络增强的对抗垃圾文本检测
5. 指标评估、超参分析、样例分析与演示接口

## 项目结构

```text
text-detection-project/
├── data/
│   ├── raw/                 # 原始数据，默认不提交大数据文件
│   └── processed/           # 预处理后的数据
├── docs/
│   ├── experiments/         # 实验结果、优化记录和对比表
│   ├── figures/             # 报告图表
│   ├── references/          # 课程要求、数据来源和指标设计
│   └── reports/             # 可直接写进报告的摘要材料
├── models/                  # 训练后的模型
└── src/
    ├── data/                # 数据读取和对抗样本生成
    ├── experiments/         # 实验 runner 和结果写出
    ├── features/            # 预处理、字符相似性和风险特征
    ├── models/              # 模型训练、评估和预测
    ├── reporting/           # 图表和报告摘要生成
    └── tests/               # 轻量测试
```

## 快速开始

```bash
cd text-detection-project
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
./.venv/bin/python -m pip install setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e . --no-build-isolation
./.venv/bin/python -m src.cli prepare-data --raw data/raw/spam_message_labeled.txt --out data/processed/spam_message_20k.tsv --sample-size 20000
./.venv/bin/python -m src.cli train --data data/processed/spam_message_20k.tsv
./.venv/bin/python -m src.cli generate-adversarial --data data/processed/spam_message_20k.tsv --out data/processed/adversarial_eval.tsv
./.venv/bin/python -m src.cli evaluate --data data/processed/adversarial_eval.tsv
./.venv/bin/python -m src.cli compare-baselines --data data/processed/spam_message_20k.tsv
./.venv/bin/python -m src.cli generate-keyword-challenge --out data/processed/keyword_challenge.tsv
./.venv/bin/python -m src.cli compare-csn --data data/processed/spam_message_20k.tsv --adversarial data/processed/keyword_challenge.tsv
./.venv/bin/python -m src.cli compare-badcases --data data/processed/spam_message_20k.tsv --adversarial data/processed/keyword_challenge.tsv
./.venv/bin/python -m src.cli plot-comparison
./.venv/bin/python -m src.cli predict --text "加我薇信，轻松月入过万"
```

如果拿到课程指定 AST 数据集，放到 `data/raw/ast_dataset.tsv` 后运行：

```bash
./.venv/bin/python -m src.cli prepare-ast --raw data/raw/ast_dataset.tsv --out data/processed/ast_dataset.tsv
./.venv/bin/python -m src.cli train --data data/processed/ast_dataset.tsv
```

## 数据格式

训练数据使用 TSV 文件，包含两列：

```text
label	text
1	加我薇信，轻松月入过万
0	今天下午三点开组会
```

其中 `label=1` 表示垃圾文本，`label=0` 表示正常文本。

## 后续实验计划

- 基线模型：TF-IDF + Logistic Regression / Linear SVM
- 对抗样本：将关键词替换为字形或字音相似变体
- 字符相似性：融合拼音相似性、常见变体词表和字符混淆规则
- Bad-case 优化：针对赌博黑话、URL 变体、插符号规避等漏检类型做风险分数融合和阈值调优
- 评价指标：Accuracy、Precision、Recall、F1、混淆矩阵
- 分析内容：阈值敏感性、模型对比、典型样例分析

## 报告材料

- 实验摘要：`docs/reports/report_summary.md`
- 对比图：`docs/figures/model_comparison.svg`
- 逐步优化记录：`docs/experiments/optimization_log.md`
