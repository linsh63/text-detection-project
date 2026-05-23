# 数据来源

## 课程指定数据：AST

第四章第三部分指定使用 AST adversarial spam text 数据集。

- 规模：16008 条
- 正常文本：5000 条
- 垃圾文本：11008 条
- 重点：包含字形和字音相似变体垃圾文本

如果拿到 AST 原文件，建议放在：

```text
data/raw/ast_dataset.tsv
```

并统一转换为：

```text
label<TAB>text
```

项目已提供 AST 适配命令：

```bash
./.venv/bin/python -m src.cli prepare-ast --raw data/raw/ast_dataset.tsv --out data/processed/ast_dataset.tsv
```

该命令会尝试兼容常见的 `label/text`、`tag/content`、无表头 TSV 等格式，并在输出时检查样本规模是否符合课程描述。

## 当前备选数据：中文垃圾短信带标签数据

为了先跑通正式规模 baseline，当前使用 `hrwhisper/SpamMessage` 公开仓库中的带标签短信数据作为备选数据。

- 仓库：https://github.com/hrwhisper/SpamMessage
- 原文件：`data/带标签短信.txt`
- 本地路径：`data/raw/spam_message_labeled.txt`
- 本地规模：800000 行
- 格式：`label<TAB>text`

说明：该数据不是课程第三部分指定的 AST，但适合作为普通中文垃圾文本分类 baseline。最终报告中应明确区分“普通垃圾文本检测”和“对抗垃圾文本检测”。

## 对抗展示数据

当前项目可从备选数据中生成一个轻量的对抗展示集：

```bash
./.venv/bin/python -m src.cli generate-adversarial --data data/processed/spam_message_20k.tsv --out data/processed/adversarial_eval.tsv
```

生成逻辑：

- 只选取垃圾文本。
- 对命中的关键词进行字形或字音替换，例如 `微信->薇信/胃信/卫星`、`红包->红苞`、`贷款->贷歀`。
- 该数据只用于展示和调试，不替代课程指定 AST。

后续如果 AST 无法取得，可以扩展该生成器，构造更难的扰动样本，用于展示模型鲁棒性。
