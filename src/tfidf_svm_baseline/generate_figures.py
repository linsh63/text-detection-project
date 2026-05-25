"""Generate report figures with matplotlib bar charts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "text_detection_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"


AST_MODEL_METRICS = [
    {"name": "Majority", "accuracy": 0.6877, "spam_f1": 0.8150},
    {"name": "TF-IDF+LR", "accuracy": 0.9904, "spam_f1": 0.9930},
    {"name": "TF-IDF+SVM", "accuracy": 0.9940, "spam_f1": 0.9956},
    {"name": "CSN+LR", "accuracy": 0.9715, "spam_f1": 0.9791},
]


DATASET_ACCURACY = [
    {"name": "AST", "accuracy": 0.9940},
    {"name": "SMS 20k", "accuracy": 0.9870},
    {"name": "FBS mixed", "accuracy": 0.9943},
    {"name": "HF spam", "accuracy": 0.9085},
    {"name": "HF conv", "accuracy": 0.9793},
]


COLORS = {
    "blue": "#2F6FED",
    "green": "#2E9D68",
    "teal": "#18A0A7",
    "orange": "#D96C4A",
    "purple": "#9A6DDF",
    "gray": "#98A2B3",
    "grid": "#D9DEE8",
    "text": "#172033",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "PingFang SC",
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "figure.facecolor": "#F6F8FB",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": COLORS["text"],
        }
    )


def add_value_labels(ax: plt.Axes, bars, y_offset: float = 0.004) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def style_axis(ax: plt.Axes, y_min: float, y_max: float) -> None:
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])


def generate_ast_metric_comparison(path: Path) -> None:
    names = [item["name"] for item in AST_MODEL_METRICS]
    accuracy = [item["accuracy"] for item in AST_MODEL_METRICS]
    spam_f1 = [item["spam_f1"] for item in AST_MODEL_METRICS]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    fig.suptitle("AST 测试集：Accuracy 与 Spam F1", fontsize=18, fontweight="bold")

    bars = axes[0].bar(x, accuracy, color=COLORS["blue"], width=0.58)
    axes[0].set_title("Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x, names, rotation=18, ha="right")
    axes[0].set_ylabel("Score")
    style_axis(axes[0], 0.65, 1.02)
    add_value_labels(axes[0], bars)

    bars = axes[1].bar(x, spam_f1, color=COLORS["green"], width=0.58)
    axes[1].set_title("Spam F1", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x, names, rotation=18, ha="right")
    axes[1].set_ylabel("Score")
    style_axis(axes[1], 0.65, 1.02)
    add_value_labels(axes[1], bars)

    fig.savefig(path, format="svg")
    plt.close(fig)


def generate_dataset_accuracy(path: Path) -> None:
    names = [item["name"] for item in DATASET_ACCURACY]
    accuracy = [item["accuracy"] for item in DATASET_ACCURACY]
    colors = [COLORS["blue"], COLORS["green"], COLORS["teal"], COLORS["orange"], COLORS["purple"]]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(12, 5.8), constrained_layout=True)
    fig.suptitle("TF-IDF+SVM 在不同数据集上的 Accuracy", fontsize=18, fontweight="bold")

    bars = ax.bar(x, accuracy, color=colors, width=0.58)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    style_axis(ax, 0.85, 1.02)
    add_value_labels(ax, bars)

    fig.savefig(path, format="svg")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    generate_ast_metric_comparison(DOCS_DIR / "metrics_comparison.svg")
    generate_dataset_accuracy(DOCS_DIR / "tfidf_svm_dataset_accuracy.svg")


if __name__ == "__main__":
    main()
