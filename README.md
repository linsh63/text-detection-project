# 中文垃圾文本检测

本项目对应大数据课程“文本检测实践”。当前仓库只保留传统机器学习路线，不使用人工敏感词表、人工变体词表或硬匹配风险词。

## 当前保留的两条路线

| 目录 | 方法 | 说明 |
|---|---|---|
| `src/tfidf_svm_baseline/` | TF-IDF+SVM | 传统机器学习强 baseline，字符级 TF-IDF(1-3gram) + Linear SVM |
| `src/rule_free_csn/` | Rule-free CSN+LR | 课件版非人工词表 CSN：字符相似性网络、字符嵌入、句子嵌入、逻辑回归 |

## 项目结构

```text
text-detection-project/
├── data/
│   ├── raw/dataset.txt              # 课程指定原始数据
│   └── processed/ast_dataset.tsv    # 已整理好的 label/text TSV
├── docs/
│   ├── experiment_report.md         # 实验报告
│   ├── bad_cases.md                 # 坏例分析
│   ├── metrics_comparison.svg       # 指标对比图
│   └── references/course/           # 课程要求备份
├── requirements.txt
├── pyproject.toml
└── src/
    ├── tfidf_svm_baseline/          # TF-IDF+SVM
    └── rule_free_csn/               # 课件版非人工词表 CSN
```

## 环境准备

推荐 Python `3.10+`。macOS 可以用 Homebrew：

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 运行实验

两个脚本都可以读取 `data/processed/ast_dataset.tsv`，也兼容课程原始三列格式 `data/raw/dataset.txt`。

运行 TF-IDF+SVM：

```bash
python -m src.tfidf_svm_baseline.experiment \
  --data data/raw/dataset.txt
```

运行 Rule-free CSN+LR：

```bash
python -m src.rule_free_csn.experiment \
  --data data/raw/dataset.txt
```

## 启动 API

API 使用 `TF-IDF+SVM` 作为默认检测模型。启动后首次预测会读取 `data/raw/dataset.txt` 训练模型并缓存。

```bash
python -m uvicorn src.tfidf_svm_baseline.api:app --host 0.0.0.0 --port 8000
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"加我领取优惠，回T退订"}'
```

响应体是布尔值：

```json
true
```

如需指定其他训练数据：

```bash
TEXT_DETECTION_DATA=data/processed/ast_dataset.tsv \
python -m uvicorn src.tfidf_svm_baseline.api:app --host 0.0.0.0 --port 8000
```

如需保存或复用模型文件：

```bash
TEXT_DETECTION_MODEL=models/tfidf_svm.joblib \
python -m uvicorn src.tfidf_svm_baseline.api:app --host 0.0.0.0 --port 8000
```

## 当前结果

在课程数据 `70%/30%` 分层划分上：

| 方法 | Accuracy | Spam F1 |
|---|---:|---:|
| TF-IDF+SVM | 0.9940 | 0.9956 |
| Rule-free CSN+LR | 0.9715 | 0.9791 |

主模型建议使用 **TF-IDF+SVM**，因为它结构最简洁、误报更少、解释成本低。Rule-free CSN+LR 可作为课程要求中“字符相似性网络、字符嵌入、句子嵌入”的实现版本。

## 文档

- `docs/experiment_report.md`：模型原理与评测结果
- `docs/bad_cases.md`：误报、漏检样本分析
- `docs/metrics_comparison.svg`：指标对比图
- `src/tfidf_svm_baseline/generate_figures.py`：生成报告中的两张 SVG 图

## 提交前检查

```bash
python -m compileall src
python -m src.tfidf_svm_baseline.experiment --data data/raw/dataset.txt
python -m src.rule_free_csn.experiment --data data/raw/dataset.txt
python -m uvicorn src.tfidf_svm_baseline.api:app --host 127.0.0.1 --port 8000
git status
```
