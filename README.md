# 基于传统机器学习的垃圾文本检测

本项目对应大数据课程第四章“文本检测实践”，只保留传统机器学习路线：文本清洗、字符/词级 TF-IDF、Logistic Regression、Linear SVM、阈值调优、分数融合和多数据集评测。语义编码器 v8 相关代码与结果已移除。

## 当前结论

| 场景 | 推荐版本 | 说明 |
|---|---|---|
| 课程 AST 数据集 | v1 / v5 | 去除硬词表后，v1 的 Spam F1 略高，v5 的召回更高、漏检更少 |
| 传统主线展示 | v1-v5 | 展示从 baseline、CSN 接口、阈值调优到分数融合的迭代过程 |
| 跨来源泛化展示 | v6 / v7 | 仍属于传统机器学习的多来源训练与融合实验 |

关键结果入口：

- `docs/experiments/multidataset/ast_v1_v5_no_hard_features_only.md`：课程数据集上 v1-v5 的无硬词表评测
- `docs/experiments/overview/latest_results_summary.md`：最新传统机器学习结果总览
- `docs/experiments/overview/optimization_log.md`：优化记录
- `docs/experiments/multidataset/evaluation_protocol_results.md`：统一评测协议结果
- `docs/references/evaluation/evaluation_protocol.md`：评测协议定义
- `docs/reports/summary/report_summary.md`：课程报告摘要素材

## 项目结构

```text
text-detection-project/
├── data/
│   ├── raw/                 # 原始数据，默认不提交大数据文件
│   └── processed/           # 预处理后的 TSV 数据，默认不提交
├── docs/
│   ├── experiments/         # 实验结果与优化记录
│   ├── figures/             # 传统模型图表
│   ├── references/          # 课程、数据、评测协议说明
│   └── reports/             # 课程报告摘要材料
├── models/                  # 训练后的模型，默认不提交
├── requirements.txt
├── pyproject.toml
└── src/
    ├── cli.py               # 命令行入口
    ├── adversarial.py       # 挑战集/增强接口；硬词表已禁用
    ├── datasets.py          # 数据读取、转换、汇总
    ├── features.py          # 清洗、分词、字符相似性接口、风险分数接口
    ├── modeling.py          # TF-IDF + LR/SVM 训练、评估、预测
    ├── runners.py           # v0-v7 传统实验 runner
    ├── visualization.py     # SVG 图表和报告摘要生成
    └── tests/               # 轻量测试
```

## 环境准备

推荐 Python `3.10+`，当前主要使用 Python `3.12` 验证。

macOS Homebrew：

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

## 数据准备

### 课程 AST 数据集

当前课程指定数据已放在：

```text
data/raw/dataset.txt
```

转换为项目统一 TSV：

```bash
python -m src.cli prepare-ast \
  --raw data/raw/dataset.txt \
  --out data/processed/ast_dataset.tsv
```

### 备用主数据集

若需要复现早期实验，可使用 `data/raw/spam_message_labeled.txt`：

```bash
python -m src.cli prepare-data \
  --raw data/raw/spam_message_labeled.txt \
  --out data/processed/spam_message_20k.tsv \
  --sample-size 20000
```

### 外部数据集

FBS：

```bash
git clone --depth 1 https://github.com/Cypher-Z/FBS_SMS_Dataset data/raw/fbs_sms_dataset
python -m src.cli prepare-fbs-mixed \
  --fbs-dir data/raw/fbs_sms_dataset \
  --normal-raw data/raw/spam_message_labeled.txt \
  --exclude data/processed/spam_message_20k.tsv \
  --out data/processed/fbs_mixed_eval.tsv \
  --sample-size 10000
```

Hugging Face 数据集：

```bash
hf auth login
python -m src.cli prepare-hf \
  --dataset reatiny/chinese-spam-10000 \
  --out data/processed/hf_chinese_spam_10000.tsv
python -m src.cli prepare-hf \
  --dataset paulkm/chinese_conversation_and_spam \
  --out data/processed/hf_chinese_conversation_and_spam.tsv
```

## 复现实验

### v1-v5 课程数据集评测

```bash
python -m src.cli validate-all-versions \
  --train-data data/processed/ast_dataset.tsv \
  --out-csv docs/experiments/multidataset/ast_v1_v5_no_hard_features.csv \
  --out-md docs/experiments/multidataset/ast_v1_v5_no_hard_features.md
```

### baseline 对比

```bash
python -m src.cli compare-baselines \
  --data data/processed/ast_dataset.tsv
```

### v0-v5 多数据集验证

```bash
python -m src.cli validate-all-versions \
  --train-data data/processed/spam_message_20k.tsv \
  --eval-data fbs_mixed=data/processed/fbs_mixed_eval.tsv \
  --eval-data hf_chinese_spam_10000=data/processed/hf_chinese_spam_10000.tsv \
  --eval-data hf_chinese_conversation_spam=data/processed/hf_chinese_conversation_and_spam.tsv
```

### 统一评测协议

```bash
python -m src.cli validate-protocols \
  --train-data data/processed/spam_message_20k.tsv \
  --external-data fbs_mixed=data/processed/fbs_mixed_eval.tsv \
  --external-data hf_chinese_spam_10000=data/processed/hf_chinese_spam_10000.tsv \
  --external-data hf_chinese_conversation_spam=data/processed/hf_chinese_conversation_and_spam.tsv \
  --adapt-train-size 0.3
```

## v1-v5 区别

| 版本 | 技术栈 | 当前无硬词表代码下的含义 |
|---|---|---|
| v1 | 字符级 TF-IDF(1-3gram) + Linear SVM | 强 baseline，当前最干净、最适合作为课程主模型的传统方法 |
| v2 | CSN 归一化接口 + 字符级 TF-IDF + Linear SVM | 保留字符相似性接口，但不使用人工混淆字/变体词表；因此当前基本等价于 v1 |
| v3 | v2 + 关键词增强接口 | 人工关键词增强已禁用，当前基本等价于 v1/v2 |
| v4 | v3 分数 + 风险分数接口 + 固定阈值 | 风险硬匹配已禁用，主要差异来自较高阈值；Precision 高但 Recall 容易下降 |
| v5 | `max(v1, v3)` 分数融合 + 验证集阈值 | 在不引入硬词表的情况下做分数融合和阈值选择；通常 Recall 更高、FN 更少 |

## 测试

```bash
python -m compileall src
python -m pytest src/tests
```

## 团队协作建议

- 代码、文档、实验结果 CSV/Markdown 可以提交。
- 原始大数据、处理后的大数据和模型文件默认不提交。
- 新实验请同步更新 `docs/experiments/overview/optimization_log.md`。
- 提交前建议运行测试并检查 `git status`。
