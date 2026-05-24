# 文档目录说明

```text
docs/
├── experiments/   # 实验结果、优化日志、CSV/Markdown 对比表
├── figures/       # 报告图表
├── references/    # 课程要求、数据来源、指标设计
└── reports/       # 可直接写进课程报告的摘要材料
```

常用入口：

- `experiments/optimization_log.md`：完整优化过程记录
- `experiments/latest_results_summary.md`：当前最重要的结果总览和最终结论
- `experiments/all_versions_multidataset_validation.md`：v0-v5 在主数据、FBS、HF 和挑战集上的完整验证
- `experiments/domain_adaptation_validation.md`：v6/v7 跨来源适配实验
- `experiments/semantic_v8_protocol_results.md`：v8.0 语义编码统一评测结果
- `experiments/semantic_v8_calibration_diagnostics.md`：v8.1 阈值校准和分数分布诊断
- `experiments/semantic_v8_encoder_comparison.md`：v8.2 多 encoder 对比实验
- `experiments/semantic_v8_autoaug_results.md`：v8.3a 自动 hard-case 增强实验
- `experiments/semantic_v8_autoaug_filtered_results.md`：v8.3b 筛选增强 + hard negative 实验
- `experiments/multidataset_fusion_validation.md`：v4/v5 多数据集验证和是否采用 v5 的结论
- `reports/report_summary.md`：报告摘要和核心结论
- `figures/model_comparison.svg`：模型对比图
- `references/course_requirements.md`：课程 PDF 要求整理
- `references/evaluation_protocol.md`：统一评测协议
