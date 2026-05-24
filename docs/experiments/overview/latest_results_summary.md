# 最新实验结果总览

本页汇总当前项目最重要的实验结论，便于团队成员和课程报告快速引用。完整过程见 `optimization_log.md`，完整指标见各 CSV 文件。

当前旧版模型 v0-v7 的最终评测入口是 `../multidataset/evaluation_protocol_results.md`，v8.0 语义模型的第一轮结果见 `../semantic_v8/semantic_v8_protocol_results.md`。两者使用同一套四类协议：

| 协议 | 名称 | 用途 |
|---|---|---|
| A | In-domain | 主数据集表现 |
| B | Zero-shot cross-domain | 只用主数据训练时的外部泛化 |
| C | Few-shot domain adaptation | 使用少量外部标注后的泛化 |
| D | Adversarial robustness | 对抗/关键词挑战鲁棒性 |

## 版本说明

| 版本 | 方法 | 主要目的 |
|---|---|---|
| v1 | 字符级 TF-IDF + Linear SVM | 强 baseline |
| v3 | CSN 归一化 + 关键词增强 | 解决短关键词变体和对抗召回 |
| v4 | v3 + bad-case 风险分数 + 阈值调优 | 降低主数据集误杀 |
| v5 | `max(v1_score, v3_score) + risk_score` | 主数据集最终模型 |
| v6 | 主数据 + 外部适配数据训练 | 提升跨来源泛化 |
| v7 | `max(v5_main_score, v6_multisource_score)` | 主数据和外部泛化折中 |
| v8.0 | 冻结语义编码器 `BAAI/bge-small-zh-v1.5` + Logistic Regression | 去除人工变体词表，验证纯语义模型方案 |
| v8.1 | v8 分数诊断与阈值校准分析 | 判断问题来自阈值迁移还是语义表示 |
| v8.2 | 多语义 encoder 对比 | 选择 v8 默认 encoder |
| v8.3a | 自动 hard positive 增强 | 不写人工词表，提升短关键词鲁棒性 |
| v8.3b | 增强样本筛选 + hard negative | 缓解 v8.3a 的跨域退化 |

## 主数据集结果

数据集：`spam_message_20k.tsv` 的固定测试集，6000 条。

| 版本 | Accuracy | Precision | Recall | Spam F1 | FP | FN | 对抗召回 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 0.9870 | 0.9021 | 0.9768 | 0.9380 | 64 | 14 | 0.0667 |
| v3 | 0.9835 | 0.8752 | 0.9752 | 0.9225 | 84 | 15 | 1.0000 |
| v4 | 0.9902 | 0.9534 | 0.9487 | 0.9510 | 28 | 31 | 1.0000 |
| v5 | 0.9917 | 0.9556 | 0.9619 | 0.9587 | 27 | 23 | 1.0000 |
| v8.0 semantic main | **0.9923** | **0.9619** | 0.9619 | **0.9619** | 23 | 23 | 1.0000 |

结论：只看主数据集，v8.0 语义模型第一版已经略高于 v5；但它在 keyword challenge 上仍明显弱于 v3-v5，因此暂时不替换课程报告主线最终模型。v5 仍是“字符相似性网络主线”的稳定最终版本，v8.0 作为新架构继续优化。

## 跨来源泛化结果

下表使用最新的跨来源适配实验。v3/v5 是只用主数据训练的模型，v6/v7 引入外部适配训练数据。指标格式为 `Spam F1 / Recall`。

| 数据集 | v3 main-only | v5 main-only | v6 CSN 多来源 | v6 Fusion 多来源 | v7 桥接 |
|---|---:|---:|---:|---:|---:|
| 主测试集 | 0.9225 / 0.9752 | **0.9587 / 0.9619** | 0.9028 / 0.9685 | 0.9259 / 0.9719 | 0.9466 / 0.9685 |
| FBS holdout | 0.9063 / 0.8397 | 0.8091 / 0.6823 | **0.9833 / 0.9860** | 0.9831 / 0.9797 | 0.9786 / 0.9674 |
| HF chinese-spam holdout | 0.1577 / 0.0872 | 0.0782 / 0.0410 | 0.8863 / 0.8701 | **0.8876 / 0.8498** | 0.8761 / 0.8139 |
| HF conversation-spam holdout | 0.5319 / 0.3654 | 0.4441 / 0.2860 | **0.9489 / 0.9448** | 0.9470 / 0.9285 | 0.9316 / 0.8909 |
| Adversarial | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Keyword challenge | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |

结论：外部数据加入适配训练后，跨来源 holdout 提升非常明显，但会牺牲主数据集 F1。因此最终推荐采用“双结论”：

- 报告主线和主数据集最终模型：v5。
- 跨来源泛化展示方案：v6 CSN / v6 Fusion。
- 若只能展示一个折中方案：v7。

## v8.0 语义编码结果

v8.0 不再使用 `features/` 中的人工变体词表或特殊风险词匹配，而是使用通用文本归一化、冻结语义编码器和逻辑回归分类器完成自动推断。

| 协议/数据集 | v8 main-only Spam F1 / Recall | v8 multisource Spam F1 / Recall | 结论 |
|---|---:|---:|---|
| A main_holdout | **0.9619 / 0.9619** | 0.8716 / 0.9553 | 主数据集上 v8 main 略优于 v5 |
| B FBS holdout | 0.4524 / 0.2934 | - | zero-shot 漏检严重 |
| B HF chinese-spam holdout | 0.0063 / 0.0032 | - | zero-shot 几乎不可用 |
| B HF conversation-spam holdout | 0.5575 / 0.3895 | - | zero-shot 仍不足 |
| C FBS holdout | - | 0.9707 / 0.9643 | 适配后明显提升，但低于 v6 最优 |
| C HF chinese-spam holdout | - | 0.8465 / 0.8111 | 适配有效，但低于 v6_fusion |
| C HF conversation-spam holdout | - | 0.8998 / 0.9025 | 适配有效，但低于 v6_csn |
| D keyword_challenge | 0.0919 / 0.0481 | 0.7413 / 0.5889 | 仍弱于 v3-v7 的满召回 |

结论：v8.0 证明“纯语义模型”在主数据集上有潜力，但第一版还不能替代 v5/v6/v7。下一步优先优化 zero-shot 和 keyword challenge：更换/对比编码器、做监督微调，或设计不依赖人工变体词表的 hard negative / hard positive 训练策略。

## v8.1 评测诊断与校准

v8.1 不改变模型，只分析 v8.0 分数：对比当前阈值和每个评测集的 oracle 阈值，并输出 PR 曲线、阈值扫描和分数分布图。

| 数据集/模型 | 当前 F1 | Oracle F1 | PR-AUC | 诊断 |
|---|---:|---:|---:|---|
| main_holdout / v8 main | 0.9619 | 0.9619 | 0.9900 | 当前阈值健康 |
| FBS holdout / v8 main | 0.4524 | 0.9119 | 0.9627 | 排序能力好，主要是阈值过保守 |
| HF chinese-spam / v8 main | 0.0063 | 0.6678 | 0.4950 | oracle 阈值退化，语义表示分不开 |
| HF conversation-spam / v8 main | 0.5575 | 0.7788 | 0.8837 | 有排序信号，主要是阈值迁移问题 |
| keyword_challenge / v8 multisource | 0.7413 | 1.0000 | - | spam-only 阈值敏感性检查，仍需 hard-case 增强 |

结论：v8.2 应优先做编码器对比。FBS 和 conversation 可尝试无目标域标签的阈值校准；`hf_chinese_spam_10000` 需要换编码器、监督适配或微调，单纯调阈值不够。

## v8.2 Encoder 对比

v8.2 固定 v8 的训练方式和 A/B/C/D 协议，只替换冻结语义编码器。对比了：

- `bge_small_zh`: `BAAI/bge-small-zh-v1.5`
- `multilingual_minilm`: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `m3e_small`: `moka-ai/m3e-small`

| 场景 | 最优 encoder | Spam F1 | Recall | PR-AUC | 说明 |
|---|---|---:|---:|---:|---|
| A main_holdout / main-only | bge_small_zh | 0.9619 | 0.9619 | 0.9900 | 主数据集仍是 BGE 最稳 |
| B FBS / zero-shot | multilingual_minilm | 0.8518 | 0.7471 | 0.9816 | 当前阈值下显著优于 BGE |
| B HF chinese-spam / zero-shot | multilingual_minilm | 0.0417 | 0.0218 | 0.5080 | F1 仍很低；m3e PR-AUC 更高但阈值不适配 |
| B HF conversation / zero-shot | bge_small_zh | 0.5575 | 0.3895 | 0.8837 | BGE 当前 F1 最好，m3e 排序能力更强 |
| C HF chinese-spam / few-shot | m3e_small | 0.8513 | 0.8240 | 0.9291 | m3e 在该数据集适配后略优 |
| C HF conversation / few-shot | bge_small_zh | 0.8998 | 0.9025 | 0.9658 | BGE 适配后仍最好 |
| D keyword_challenge / multisource | bge_small_zh | 0.7413 | 0.5889 | - | 三者都未解决短关键词挑战 |

结论：没有单个 encoder 全面胜出。若只选一个默认 v8 encoder，仍建议保留 `bge_small_zh`；若追求 FBS zero-shot，可以考虑 `multilingual_minilm`；若重点优化 `hf_chinese_spam_10000`，`m3e_small` 值得继续做校准或微调。下一步 v8.3 不应只继续换 encoder，而应做自动 hard-case 增强或 encoder ensemble。

## v8.3a 轻量自动 Hard-case 增强

v8.3a 不写人工垃圾词表，而是从训练 split 自动挖掘 spam 高风险短 n-gram，并基于分隔符、重复字符和训练语料中的同音字符生成 hard positive 样本。最终采用轻量参数：`max_terms=80`、`max_augmented=200`、`min_spam_df=3`。

| 场景 | Baseline F1 | AutoAug F1 | 变化 | 说明 |
|---|---:|---:|---:|---|
| A main_holdout / main | 0.9619 | 0.9548 | -0.0071 | 主数据小幅下降 |
| B FBS / main zero-shot | 0.4524 | 0.4799 | +0.0275 | 有小幅提升 |
| B HF chinese-spam / main zero-shot | 0.0063 | 0.0145 | +0.0082 | 仍然很低 |
| B HF conversation / main zero-shot | 0.5575 | 0.4767 | -0.0808 | 有明显退化 |
| C HF chinese-spam / multisource | 0.8465 | 0.8406 | -0.0059 | 适配后小幅下降 |
| D keyword_challenge / main | 0.0919 | 0.4393 | +0.3474 | 短关键词鲁棒性明显提升 |
| D keyword_challenge / multisource | 0.7413 | 0.7586 | +0.0174 | 多来源下小幅提升 |

结论：v8.3a 证明自动 hard-case 增强能提升短关键词挑战，但会带来跨域副作用。它适合作为“自动增强有效但需要约束”的实验版本，不建议直接替换 v8.0/v8.2 默认模型。下一步应做 v8.3b：增强样本筛选或加入 hard negative，减少 HF conversation 退化。

## v8.3b 增强样本筛选 + Hard Negative

v8.3b 保留 v8.3a 的自动 hard positive 生成，但加入两层约束：先用原始语义模型分数过滤过易或过激进的增强正样本，再把高风险正常样本及其自动变体作为 hard negative 加回训练。当前默认参数采用 `positive_max_score=0.75`、`max_hard_negatives=200`。

本轮正式结果共保留 `644` 条 filtered positive、`400` 条 hard negative full normal、`118` 条 hard negative variant；自动变体生成顺序已固定，避免进程间 `set` 顺序导致结果漂移。

| 场景 | Baseline F1 | v8.3a F1 | v8.3b F1 | v8.3b vs Baseline | v8.3b vs v8.3a | 说明 |
|---|---:|---:|---:|---:|---:|---|
| A main_holdout / main | 0.9619 | 0.9548 | 0.9593 | -0.0026 | +0.0045 | 主数据退化小于 v8.3a |
| B FBS / main zero-shot | 0.4524 | 0.4799 | 0.5401 | +0.0877 | +0.0602 | zero-shot 召回提升更明显 |
| B HF chinese-spam / main zero-shot | 0.0063 | 0.0145 | 0.0162 | +0.0099 | +0.0017 | 仍很低，但略有改善 |
| B HF conversation / main zero-shot | 0.5575 | 0.4767 | 0.5144 | -0.0431 | +0.0377 | 明显缓解 v8.3a 副作用，但未超过 baseline |
| C HF conversation / multisource | 0.8998 | 0.8898 | 0.8949 | -0.0048 | +0.0052 | few-shot 下退化较小 |
| D keyword_challenge / main | 0.0919 | 0.4393 | 0.4211 | +0.3292 | -0.0183 | 保留大部分短关键词收益 |
| D keyword_challenge / multisource | 0.7413 | 0.7586 | 0.7727 | +0.0315 | +0.0141 | 多来源下优于 v8.3a |

结论：v8.3b 是比 v8.3a 更均衡的自动增强版本。它证明 hard negative 和增强样本筛选能减轻跨域副作用，但 main-only 的 HF conversation 仍低于 v8.0 baseline，说明下一步应继续做更细粒度的增强样本权重、域不变校准或小规模监督微调。

## 关键结果文件

| 文件 | 内容 |
|---|---|
| `../multidataset/evaluation_protocol_results.csv` | v0-v7 统一评测协议结果 |
| `../multidataset/evaluation_protocol_splits.csv` | 协议评测中的外部数据切分 |
| `../multidataset/all_versions_multidataset_validation.csv` | v0-v5 多数据集验证 |
| `../classic/baseline_comparison.csv` | v1 baseline 对比 |
| `../classic/csn_comparison.csv` | v2/v3 CSN 优化 |
| `../classic/bad_case_optimization.csv` | v4 bad-case 阈值优化 |
| `../classic/fusion_experiment.csv` | v5 分数融合 |
| `../domain_adaptation/domain_adaptation_validation.csv` | v6/v7 跨来源适配实验 |
| `../domain_adaptation/domain_adaptation_validation_10pct.csv` | 10% 外部适配比例扫描 |
| `../domain_adaptation/domain_adaptation_validation_20pct.csv` | 20% 外部适配比例扫描 |
| `../semantic_v8/semantic_v8_protocol_results.csv` | v8.0 语义编码统一评测结果 |
| `../semantic_v8/semantic_v8_protocol_splits.csv` | v8.0 协议评测中的外部数据切分 |
| `../semantic_v8/semantic_v8_calibration_diagnostics.md` | v8.1 诊断报告和图表入口 |
| `../semantic_v8/semantic_v8_threshold_grid.csv` | v8.1 阈值扫描明细 |
| `../semantic_v8/semantic_v8_encoder_comparison.md` | v8.2 多 encoder 对比报告 |
| `../semantic_v8/semantic_v8_autoaug_results.md` | v8.3a 自动 hard-case 增强报告 |
| `../semantic_v8/semantic_v8_autoaug_terms.csv` | v8.3a 自动挖掘片段 |
| `../semantic_v8/semantic_v8_autoaug_filtered_results.md` | v8.3b 筛选增强 + hard negative 报告 |
| `../semantic_v8/semantic_v8_autoaug_filtered_terms.csv` | v8.3b 正负增强片段 |
| `../semantic_v8/semantic_v8_autoaug_filtered_examples.csv` | v8.3b 生成样本明细 |
