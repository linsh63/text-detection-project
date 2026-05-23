# 基于字符相似性网络的垃圾文本检测

本项目对应大数据课程第四章“文本检测实践”，目标是构建一个可复现的中文垃圾文本检测工程。项目从 baseline 开始，逐步加入字符相似性网络、bad-case 风险特征、分数融合和跨来源适配，最终形成可用于课程报告的实验链路。

## 当前结论

| 场景 | 推荐版本 | 说明 |
|---|---|---|
| 主数据集最终模型 | v5 | 主测试集 Spam F1 最高，`0.9587` |
| 跨来源泛化展示 | v6 CSN / v6 Fusion | 外部 holdout 上提升最明显 |
| 折中单模型 | v7 | 主数据集低于 v5，但外部集明显优于 main-only 模型 |

关键结果入口：

- `docs/experiments/latest_results_summary.md`：最新结果总览
- `docs/experiments/optimization_log.md`：完整优化记录
- `docs/experiments/domain_adaptation_validation.md`：v6/v7 跨来源适配实验
- `docs/reports/report_summary.md`：课程报告摘要素材

## 项目结构

```text
text-detection-project/
├── data/
│   ├── raw/                 # 原始数据，默认不提交大数据文件
│   └── processed/           # 预处理后的 TSV 数据，默认不提交
├── docs/
│   ├── experiments/         # 实验结果、优化记录和对比表
│   ├── figures/             # 报告图表
│   ├── references/          # 课程要求、数据来源和指标设计
│   └── reports/             # 可直接写进报告的摘要材料
├── models/                  # 训练后的模型，默认不提交
├── requirements.txt
├── pyproject.toml
└── src/
    ├── data/                # 数据读取、数据集转换和对抗样本生成
    ├── experiments/         # 实验 runner 和结果写出
    ├── features/            # 预处理、字符相似性和风险特征
    ├── models/              # 模型训练、评估和预测
    ├── reporting/           # 图表和报告摘要生成
    └── tests/               # 轻量测试
```

## 环境准备

推荐 Python `3.10+`，当前主要使用 Python `3.12` 验证。

### macOS

Homebrew 用户：

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
```

如果 `python3.12` 不在 PATH，可以使用：

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

如果系统默认 Python 版本太低，请先安装 Python `3.10+`。

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果执行策略拦截激活脚本，可以临时运行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 安装依赖

激活虚拟环境后，在项目根目录执行：

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

下面的命令默认已经激活虚拟环境。如果不想激活环境，可以把 `python` 替换成：

- macOS/Linux：`.venv/bin/python`
- Windows：`.venv\Scripts\python`

## 数据准备

数据文件较大，默认不提交到 Git。团队成员需要自行放置或生成以下文件。

### 主数据集

当前主数据集来自 `hrwhisper/SpamMessage` 的带标签短信数据，统一放到：

```text
data/raw/spam_message_labeled.txt
```

格式为：

```text
label<TAB>text
```

其中 `label=1` 表示垃圾文本，`label=0` 表示正常文本。

生成 2 万条主实验数据：

```bash
python -m src.cli prepare-data \
  --raw data/raw/spam_message_labeled.txt \
  --out data/processed/spam_message_20k.tsv \
  --sample-size 20000
```

### 对抗和关键词挑战集

```bash
python -m src.cli generate-adversarial \
  --data data/processed/spam_message_20k.tsv \
  --out data/processed/adversarial_eval.tsv

python -m src.cli generate-keyword-challenge \
  --out data/processed/keyword_challenge.tsv
```

### FBS 跨来源数据集

FBS 是公开的中国伪基站垃圾短信数据集，只包含垃圾短信。项目会将它和主数据源中的正常短信混合成二分类评测集。

```bash
git clone --depth 1 https://github.com/Cypher-Z/FBS_SMS_Dataset data/raw/fbs_sms_dataset

python -m src.cli prepare-fbs-mixed \
  --fbs-dir data/raw/fbs_sms_dataset \
  --normal-raw data/raw/spam_message_labeled.txt \
  --exclude data/processed/spam_message_20k.tsv \
  --out data/processed/fbs_mixed_eval.tsv \
  --sample-size 10000
```

### Hugging Face 数据集

本项目使用两个 gated dataset：

- `reatiny/chinese-spam-10000`
- `paulkm/chinese_conversation_and_spam`

先在网页端登录 Hugging Face，并进入对应数据集页面同意访问条款。然后登录本地 CLI：

```bash
hf auth login
hf auth whoami
```

转换为项目统一 TSV：

```bash
python -m src.cli prepare-hf \
  --dataset reatiny/chinese-spam-10000 \
  --out data/processed/hf_chinese_spam_10000.tsv

python -m src.cli prepare-hf \
  --dataset paulkm/chinese_conversation_and_spam \
  --out data/processed/hf_chinese_conversation_and_spam.tsv
```

如果数据集不是 gated，或你不想传 Hugging Face token，可以加 `--no-token`。

### AST 数据集

课程 PDF 指定 AST adversarial spam text 数据集，但当前没有找到可直接下载的公开原始文件。若后续拿到 AST，放到：

```text
data/raw/ast_dataset.tsv
```

然后执行：

```bash
python -m src.cli prepare-ast \
  --raw data/raw/ast_dataset.tsv \
  --out data/processed/ast_dataset.tsv
```

## 复现实验

### 1. baseline 对比

```bash
python -m src.cli compare-baselines \
  --data data/processed/spam_message_20k.tsv
```

输出：

- `docs/experiments/baseline_comparison.csv`
- `docs/experiments/baseline_comparison.md`

### 2. CSN 优化

```bash
python -m src.cli compare-csn \
  --data data/processed/spam_message_20k.tsv \
  --adversarial data/processed/keyword_challenge.tsv
```

### 3. bad-case 风险和阈值优化

```bash
python -m src.cli compare-badcases \
  --data data/processed/spam_message_20k.tsv \
  --adversarial data/processed/keyword_challenge.tsv
```

### 4. v5 分数融合

```bash
python -m src.cli compare-fusions \
  --data data/processed/spam_message_20k.tsv \
  --adversarial data/processed/keyword_challenge.tsv
```

### 5. v0-v5 多数据集验证

```bash
python -m src.cli validate-all-versions \
  --train-data data/processed/spam_message_20k.tsv \
  --eval-data fbs_mixed=data/processed/fbs_mixed_eval.tsv \
  --eval-data hf_chinese_spam_10000=data/processed/hf_chinese_spam_10000.tsv \
  --eval-data hf_chinese_conversation_spam=data/processed/hf_chinese_conversation_and_spam.tsv \
  --eval-data adversarial=data/processed/adversarial_eval.tsv \
  --eval-data keyword_challenge=data/processed/keyword_challenge.tsv
```

### 6. v6/v7 跨来源适配实验

默认版本使用每个外部二分类数据集的 30% 作为适配训练数据，70% 作为外部 holdout。

```bash
python -m src.cli validate-domain-adaptation \
  --train-data data/processed/spam_message_20k.tsv \
  --adapt-data fbs_mixed=data/processed/fbs_mixed_eval.tsv \
  --adapt-data hf_chinese_spam_10000=data/processed/hf_chinese_spam_10000.tsv \
  --adapt-data hf_chinese_conversation_spam=data/processed/hf_chinese_conversation_and_spam.tsv \
  --challenge-data adversarial=data/processed/adversarial_eval.tsv \
  --challenge-data keyword_challenge=data/processed/keyword_challenge.tsv \
  --adapt-train-size 0.3
```

也可以扫描不同外部适配比例：

```bash
python -m src.cli validate-domain-adaptation \
  --train-data data/processed/spam_message_20k.tsv \
  --adapt-data fbs_mixed=data/processed/fbs_mixed_eval.tsv \
  --adapt-data hf_chinese_spam_10000=data/processed/hf_chinese_spam_10000.tsv \
  --adapt-data hf_chinese_conversation_spam=data/processed/hf_chinese_conversation_and_spam.tsv \
  --challenge-data adversarial=data/processed/adversarial_eval.tsv \
  --challenge-data keyword_challenge=data/processed/keyword_challenge.tsv \
  --adapt-train-size 0.1 \
  --out-csv docs/experiments/domain_adaptation_validation_10pct.csv \
  --out-md docs/experiments/domain_adaptation_validation_10pct.md \
  --split-csv docs/experiments/domain_adaptation_splits_10pct.csv
```

将 `--adapt-train-size 0.1` 改为 `0.2` 或 `0.3` 即可复现实验记录中的比例扫描。

## 训练与预测

训练一个基础模型：

```bash
python -m src.cli train \
  --data data/processed/spam_message_20k.tsv \
  --model models/baseline.joblib \
  --analyzer char
```

预测单条文本：

```bash
python -m src.cli predict \
  --model models/baseline.joblib \
  --text "加我薇信，轻松月入过万"
```

## 测试

```bash
python -m pytest src/tests
python -m compileall src
```

## 常见问题

### 1. `hf auth login` 找不到

确认已经安装依赖，并且当前处于虚拟环境：

```bash
python -m pip install -r requirements.txt
python -m pip show huggingface_hub
```

如果仍不可用，可以直接使用：

```bash
python -m huggingface_hub.commands.huggingface_cli --help
```

### 2. Hugging Face 数据集提示无权限

需要先在网页端打开数据集页面，登录并同意访问条款，然后重新执行：

```bash
hf auth login
```

### 3. 找不到数据文件

确认 `data/raw/` 和 `data/processed/` 下的数据已经按 README 生成。大数据文件默认不进入 Git，因此 clone 仓库后通常需要重新准备数据。

### 4. Windows 路径或激活命令不同

Windows 下建议使用 PowerShell，并把 README 中的 `python` 命令保持不变；只要虚拟环境已激活，`python` 会指向 `.venv`。

## 团队协作建议

- 代码、文档、实验结果 CSV/Markdown 可以提交。
- 原始大数据、处理后的大数据和模型文件默认不提交。
- 新实验请同步更新 `docs/experiments/optimization_log.md`。
- 若新增外部数据源，请同步更新 `docs/references/data_sources.md`。
- 提交前建议运行：

```bash
python -m compileall src
python -m pytest src/tests
git status
```
