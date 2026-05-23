"""Generate lightweight SVG figures for report-ready experiment summaries."""

from __future__ import annotations

from dataclasses import dataclass
from os.path import relpath
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MethodStyle:
    key: str
    label: str
    color: str


METHODS = (
    MethodStyle("v1_strong_baseline_default", "v1 强 baseline", "#2F6FED"),
    MethodStyle("v3_csn_aug_default", "v3 CSN增强", "#E05D44"),
    MethodStyle("v4_bad_case_valid_tuned", "v4 bad-case调优", "#2E9D68"),
)

PANELS = (
    ("clean_f1_spam", "Spam F1", "higher", 0.0, 1.0),
    ("clean_false_positive", "误杀 FP", "lower", 0.0, 100.0),
    ("clean_false_negative", "漏检 FN", "lower", 0.0, 40.0),
    ("adv_recall_spam", "对抗召回", "higher", 0.0, 1.0),
)


def _svg_text(
    x: float,
    y: float,
    value: str,
    size: int = 14,
    weight: str = "400",
    fill: str = "#253044",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, '
        f"'PingFang SC', 'Microsoft YaHei', sans-serif\" font-size=\"{size}\" "
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{value}</text>'
    )


def _format_value(metric: str, value: float) -> str:
    if metric.endswith("f1_spam") or metric.endswith("recall_spam"):
        return f"{value:.3f}"
    return f"{value:.0f}"


def _load_chart_data(input_csv: str | Path) -> pd.DataFrame:
    data = pd.read_csv(input_csv)
    wanted = [method.key for method in METHODS]
    data = data[data["name"].isin(wanted)].copy()
    if len(data) != len(METHODS):
        missing = sorted(set(wanted) - set(data["name"]))
        raise ValueError(f"Missing methods in comparison CSV: {missing}")
    order = {method.key: index for index, method in enumerate(METHODS)}
    data["order"] = data["name"].map(order)
    return data.sort_values("order").reset_index(drop=True)


def _panel_svg(
    data: pd.DataFrame,
    metric: str,
    title: str,
    direction: str,
    min_value: float,
    max_value: float,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    lines = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'rx="6" fill="#FFFFFF" stroke="#D9DEE8"/>',
        _svg_text(x + 20, y + 34, title, size=18, weight="700"),
    ]
    hint = "越高越好" if direction == "higher" else "越低越好"
    lines.append(_svg_text(x + width - 20, y + 34, hint, size=12, fill="#6B7280", anchor="end"))

    chart_x = x + 70
    chart_y = y + 62
    chart_width = width - 110
    chart_height = height - 120
    axis_y = chart_y + chart_height

    lines.extend(
        [
            f'<line x1="{chart_x:.1f}" y1="{axis_y:.1f}" x2="{chart_x + chart_width:.1f}" '
            f'y2="{axis_y:.1f}" stroke="#B8C0CC"/>',
            f'<line x1="{chart_x:.1f}" y1="{chart_y:.1f}" x2="{chart_x:.1f}" '
            f'y2="{axis_y:.1f}" stroke="#B8C0CC"/>',
        ]
    )

    for ratio in (0.0, 0.5, 1.0):
        tick_value = min_value + (max_value - min_value) * ratio
        tick_y = axis_y - chart_height * ratio
        label = f"{tick_value:.1f}" if max_value <= 1.0 else f"{tick_value:.0f}"
        lines.extend(
            [
                f'<line x1="{chart_x - 4:.1f}" y1="{tick_y:.1f}" x2="{chart_x:.1f}" '
                f'y2="{tick_y:.1f}" stroke="#B8C0CC"/>',
                _svg_text(chart_x - 10, tick_y + 4, label, size=11, fill="#6B7280", anchor="end"),
                f'<line x1="{chart_x:.1f}" y1="{tick_y:.1f}" x2="{chart_x + chart_width:.1f}" '
                f'y2="{tick_y:.1f}" stroke="#EEF1F5"/>',
            ]
        )

    bar_width = 58
    gap = (chart_width - bar_width * len(METHODS)) / (len(METHODS) + 1)
    for index, method in enumerate(METHODS):
        value = float(data.loc[data["name"] == method.key, metric].iloc[0])
        ratio = min(max((value - min_value) / (max_value - min_value), 0.0), 1.0)
        bar_height = chart_height * ratio
        bar_x = chart_x + gap + index * (bar_width + gap)
        bar_y = axis_y - bar_height
        lines.extend(
            [
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" rx="4" fill="{method.color}"/>',
                _svg_text(
                    bar_x + bar_width / 2,
                    max(bar_y - 8, chart_y + 12),
                    _format_value(metric, value),
                    size=13,
                    weight="700",
                    fill="#253044",
                    anchor="middle",
                ),
                _svg_text(
                    bar_x + bar_width / 2,
                    axis_y + 24,
                    method.label.split(" ")[0],
                    size=11,
                    fill="#3F4A5F",
                    anchor="middle",
                ),
            ]
        )
    return lines


def generate_model_comparison_svg(
    input_csv: str | Path = "docs/experiments/bad_case_optimization.csv",
    output_svg: str | Path = "docs/figures/model_comparison.svg",
) -> Path:
    """Create a four-panel SVG comparing v1, v3 and v4."""
    data = _load_chart_data(input_csv)
    output_svg = Path(output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    width = 1100
    height = 760
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="模型优化对比图">',
        '<rect width="1100" height="760" fill="#F6F8FB"/>',
        _svg_text(48, 58, "文本检测模型优化对比", size=28, weight="800", fill="#172033"),
        _svg_text(
            48,
            86,
            "v4 在保持对抗召回 1.000 的同时，显著降低误杀并提升 Spam F1",
            size=15,
            fill="#526071",
        ),
    ]

    legend_x = 760
    for index, method in enumerate(METHODS):
        lx = legend_x
        ly = 42 + index * 26
        lines.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="16" height="16" rx="3" fill="{method.color}"/>')
        lines.append(_svg_text(lx + 24, ly + 13, method.label, size=13, fill="#3F4A5F"))

    panel_positions = ((48, 118), (584, 118), (48, 430), (584, 430))
    for panel, position in zip(PANELS, panel_positions, strict=True):
        lines.extend(_panel_svg(data, *panel, x=position[0], y=position[1], width=468, height=264))

    lines.extend(
        [
            _svg_text(
                48,
                724,
                "数据：spam_message_20k 固定测试集；对抗集：keyword_challenge。v4 参数来自训练集内部 validation split。",
                size=13,
                fill="#6B7280",
            ),
            "</svg>",
        ]
    )
    output_svg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_svg


def write_report_summary(
    input_csv: str | Path = "docs/experiments/bad_case_optimization.csv",
    figure_path: str | Path = "docs/figures/model_comparison.svg",
    output_md: str | Path = "docs/reports/report_summary.md",
) -> Path:
    """Write a concise report-ready experiment summary."""
    data = _load_chart_data(input_csv).set_index("name")
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    figure_rel = Path(relpath(figure_path, start=output_md.parent))

    v1 = data.loc["v1_strong_baseline_default"]
    v3 = data.loc["v3_csn_aug_default"]
    v4 = data.loc["v4_bad_case_valid_tuned"]

    lines = [
        "# 实验结果报告摘要",
        "",
        "## 优化主线",
        "",
        "本项目先建立 `字符级 TF-IDF + Linear SVM` 强 baseline，再引入字符相似性网络（CSN）处理字形、字音变体，最后根据 bad case 分析加入风险分数和阈值调优。",
        "",
        f"![模型优化对比图]({figure_rel.as_posix()})",
        "",
        "## 核心结论",
        "",
        f"- v1 强 baseline 在普通测试集上 Spam F1 为 `{v1['clean_f1_spam']:.4f}`，但 keyword challenge 对抗召回只有 `{v1['adv_recall_spam']:.4f}`。",
        f"- v3 CSN + 关键词增强把对抗召回提升到 `{v3['adv_recall_spam']:.4f}`，但误杀从 `{v1['clean_false_positive']:.0f}` 增加到 `{v3['clean_false_positive']:.0f}`。",
        f"- v4 bad-case 调优保持对抗召回 `{v4['adv_recall_spam']:.4f}`，同时把误杀降到 `{v4['clean_false_positive']:.0f}`，Spam F1 提升到 `{v4['clean_f1_spam']:.4f}`。",
        "",
        "## 可展示指标",
        "",
        "| 方法 | Spam F1 | Precision | Recall | FP | FN | 对抗召回 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = data.loc[method.key]
        lines.append(
            f"| {method.label} | {row['clean_f1_spam']:.4f} | "
            f"{row['clean_precision_spam']:.4f} | {row['clean_recall_spam']:.4f} | "
            f"{row['clean_false_positive']:.0f} | {row['clean_false_negative']:.0f} | "
            f"{row['adv_recall_spam']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 报告表述建议",
            "",
            "v1 说明字符级建模对中文短文本有效；v3 说明字符相似性网络能解决对抗变体；v4 说明通过 bad-case 分析和阈值调优，可以在鲁棒性和误杀控制之间取得更好的平衡。",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def generate_report_assets(
    input_csv: str | Path = "docs/experiments/bad_case_optimization.csv",
    output_svg: str | Path = "docs/figures/model_comparison.svg",
    output_md: str | Path = "docs/reports/report_summary.md",
) -> tuple[Path, Path]:
    figure = generate_model_comparison_svg(input_csv=input_csv, output_svg=output_svg)
    summary = write_report_summary(input_csv=input_csv, figure_path=figure, output_md=output_md)
    return figure, summary
