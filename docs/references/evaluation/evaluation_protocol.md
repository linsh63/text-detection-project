# 统一评测协议

本项目后续所有模型，包括旧版 v0-v7 和计划中的 v8 语义编码模型，都应接入同一套评测协议。这样可以避免只在单一数据集上比较模型，导致“主数据集效果好但跨来源泛化差”的结论被掩盖。

## 数据集分层

| 类型 | 数据集 | 用途 |
|---|---|---|
| 主数据集 | `spam_message_20k.tsv` | 衡量课程主任务表现 |
| 跨来源二分类数据 | `fbs_mixed_eval.tsv`、两个 HF 数据集 | 衡量跨来源泛化 |
| 对抗/挑战数据 | `adversarial_eval.tsv`、`keyword_challenge.tsv` | 衡量变体鲁棒性 |
| 课程指定数据 | AST，如果后续取得 | 作为课程要求核心评测 |

## 协议定义

| 协议 | 名称 | 训练数据 | 测试数据 | 目的 |
|---|---|---|---|---|
| Protocol A | In-domain | 主数据 train | 主数据 holdout | 衡量课程主任务表现 |
| Protocol B | Zero-shot cross-domain | 只用主数据 train | 外部 holdout | 衡量裸泛化能力 |
| Protocol C | Few-shot domain adaptation | 主数据 + 外部 adapt train | 外部 holdout | 衡量少量外部标注后的泛化 |
| Protocol D | Adversarial robustness | 对应模型训练数据 | adversarial / keyword challenge | 衡量对抗变体鲁棒性 |

## 外部数据切分

外部二分类数据统一切分为：

| split | 默认比例 | 用途 |
|---|---:|---|
| external adapt train | 30% | 只给 Protocol C 使用 |
| external adapt fit | adapt train 的 80% | 训练多来源模型 |
| external adapt validation | adapt train 的 20% | 选择阈值和融合参数 |
| external holdout | 70% | Protocol B/C 最终评测，不能进入训练和阈值选择 |

可以额外扫描 `10%`、`20%`、`30%` 外部适配比例，用于展示少量标注数据对泛化能力的影响。

## 指标口径

二分类数据集重点看：

| 指标 | 用途 |
|---|---|
| Spam F1 | 主指标，平衡 Precision 和 Recall |
| Spam Recall | 衡量漏检 |
| Spam Precision | 衡量误杀/误报控制 |
| FP / FN | 最直观，适合报告展示 |
| Macro F1 | 类别不均衡时的辅助指标 |
| PR-AUC | 垃圾文本类别不均衡时比 ROC-AUC 更有解释性 |
| Recall@Precision>=95% | 展示误杀受控时的召回能力 |

单类挑战集只重点看：

| 指标 | 原因 |
|---|---|
| Recall | 单类垃圾样本中检测出多少 |
| FN | 漏掉多少个挑战样本 |

不要把单类挑战集的 Accuracy 或 Macro F1 作为主结论。

## 模型选择规则

推荐同时报告两个分数：

| 分数 | 定义 | 用途 |
|---|---|---|
| Main Score | 主数据 holdout 的 Spam F1 | 课程主线模型选择 |
| Generalization Score | 外部 holdout 平均 Spam F1 | 跨来源泛化模型选择 |

当前推荐结论：

| 目标 | 推荐版本 |
|---|---|
| 主数据集最终模型 | v5 |
| 跨来源泛化模型 | v6 CSN / v6 Fusion |
| 折中单模型 | v7 |

## 运行命令

```bash
python -m src.cli validate-protocols \
  --train-data data/processed/spam_message_20k.tsv \
  --external-data fbs_mixed=data/processed/fbs_mixed_eval.tsv \
  --external-data hf_chinese_spam_10000=data/processed/hf_chinese_spam_10000.tsv \
  --external-data hf_chinese_conversation_spam=data/processed/hf_chinese_conversation_and_spam.tsv \
  --challenge-data adversarial=data/processed/adversarial_eval.tsv \
  --challenge-data keyword_challenge=data/processed/keyword_challenge.tsv \
  --adapt-train-size 0.3
```

输出文件：

- `docs/experiments/multidataset/evaluation_protocol_results.csv`
- `docs/experiments/multidataset/evaluation_protocol_results.md`
- `docs/experiments/multidataset/evaluation_protocol_splits.csv`

## v8 接入要求

后续 v8 语义编码模型必须接入 Protocol A/B/C/D。只有同时报告主数据集、zero-shot 跨来源、few-shot 适配和对抗挑战结果，才能判断语义模型是否真正提升泛化能力。
