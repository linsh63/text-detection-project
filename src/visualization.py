"""Minimal report helpers for the retained v1 model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report_summary(
    input_csv: str | Path = "docs/experiments/multidataset/v1_validation.csv",
    output_md: str | Path = "docs/reports/summary/report_summary.md",
) -> Path:
    """Write a concise v1-only report summary."""
    data = pd.read_csv(input_csv)
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    main = data[data["dataset"] == "main_holdout"].iloc[0]
    lines = [
        "# 实验结果报告摘要",
        "",
        "本项目当前只保留 v1：字符级 TF-IDF(1-3gram) + Linear SVM。",
        "",
        "## 核心指标",
        "",
        "| Dataset | Accuracy | Precision | Recall | Spam F1 | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| {main['dataset']} | {main['accuracy']:.4f} | "
            f"{main['precision_spam']:.4f} | {main['recall_spam']:.4f} | "
            f"{main['f1_spam']:.4f} | {main['false_positive']:.0f} | "
            f"{main['false_negative']:.0f} |"
        ),
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md
