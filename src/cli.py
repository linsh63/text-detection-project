"""Command-line interface for training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.adversarial import generate_adversarial_dataset, generate_keyword_challenge_dataset
from .data.datasets import (
    dataset_summary,
    prepare_ast_dataset,
    prepare_fbs_mixed_dataset,
    prepare_huggingface_dataset,
    prepare_labeled_dataset,
)
from .experiments.runners import (
    compare_all_versions_multidataset_validation,
    compare_bad_case_optimization,
    compare_baselines,
    compare_csn_optimization,
    compare_domain_adaptation_validation,
    compare_evaluation_protocols,
    compare_multidataset_fusion_validation,
    compare_score_fusion_optimization,
)
from .experiments.semantic_runners import (
    DEFAULT_ENCODERS,
    compare_semantic_v8_auto_augmentation,
    compare_semantic_v8_encoders,
    compare_semantic_v8_filtered_auto_augmentation,
    compare_semantic_v8_protocols,
    diagnose_semantic_v8_calibration,
)
from .models.modeling import evaluate_model, predict_text, train_baseline
from .reporting.visualization import generate_report_assets


def _parse_named_paths(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            parsed.append((name.strip(), path.strip()))
        else:
            parsed.append((Path(value).stem, value))
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text-detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data", help="Normalize a labeled TSV dataset")
    prepare_parser.add_argument("--raw", required=True)
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--sample-size", type=int, default=20000)
    prepare_parser.add_argument("--random-state", type=int, default=42)

    ast_parser = subparsers.add_parser("prepare-ast", help="Normalize the AST dataset")
    ast_parser.add_argument("--raw", required=True)
    ast_parser.add_argument("--out", default="data/processed/ast_dataset.tsv")
    ast_parser.add_argument("--sample-size", type=int)
    ast_parser.add_argument("--random-state", type=int, default=42)

    fbs_parser = subparsers.add_parser(
        "prepare-fbs-mixed",
        help="Build a binary cross-source dataset from FBS spam and local normal texts",
    )
    fbs_parser.add_argument("--fbs-dir", required=True)
    fbs_parser.add_argument("--normal-raw", required=True)
    fbs_parser.add_argument("--out", default="data/processed/fbs_mixed_eval.tsv")
    fbs_parser.add_argument("--sample-size", type=int, default=10000)
    fbs_parser.add_argument("--spam-ratio", type=float, default=0.5)
    fbs_parser.add_argument("--exclude")
    fbs_parser.add_argument("--random-state", type=int, default=42)

    hf_parser = subparsers.add_parser(
        "prepare-hf",
        help="Download a Hugging Face dataset and normalize it to label/text TSV",
    )
    hf_parser.add_argument("--dataset", required=True)
    hf_parser.add_argument("--out", required=True)
    hf_parser.add_argument(
        "--no-token",
        action="store_true",
        help="Do not pass the local Hugging Face auth token to load_dataset",
    )

    adversarial_parser = subparsers.add_parser(
        "generate-adversarial",
        help="Generate a spam-only adversarial variant evaluation set",
    )
    adversarial_parser.add_argument("--data", required=True)
    adversarial_parser.add_argument("--out", default="data/processed/adversarial_eval.tsv")
    adversarial_parser.add_argument("--max-samples", type=int, default=1000)
    adversarial_parser.add_argument("--random-state", type=int, default=42)

    challenge_parser = subparsers.add_parser(
        "generate-keyword-challenge",
        help="Generate a compact adversarial keyword challenge set",
    )
    challenge_parser.add_argument("--out", default="data/processed/keyword_challenge.tsv")

    train_parser = subparsers.add_parser("train", help="Train the baseline model")
    train_parser.add_argument("--data", default="data/raw/sample_texts.tsv")
    train_parser.add_argument("--model", default="models/baseline.joblib")
    train_parser.add_argument("--analyzer", choices=["char", "word"], default="char")

    predict_parser = subparsers.add_parser("predict", help="Predict one text")
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--model", default="models/baseline.joblib")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model")
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--model", default="models/baseline.joblib")

    compare_parser = subparsers.add_parser("compare-baselines", help="Run baseline comparison")
    compare_parser.add_argument("--data", required=True)
    compare_parser.add_argument("--out-csv", default="docs/experiments/classic/baseline_comparison.csv")
    compare_parser.add_argument("--out-md", default="docs/experiments/classic/baseline_comparison.md")
    compare_parser.add_argument("--test-size", type=float, default=0.3)
    compare_parser.add_argument("--random-state", type=int, default=42)

    csn_parser = subparsers.add_parser("compare-csn", help="Compare CSN optimization")
    csn_parser.add_argument("--data", required=True)
    csn_parser.add_argument("--adversarial", required=True)
    csn_parser.add_argument("--out-csv", default="docs/experiments/classic/csn_comparison.csv")
    csn_parser.add_argument("--out-md", default="docs/experiments/classic/csn_comparison.md")
    csn_parser.add_argument("--test-size", type=float, default=0.3)
    csn_parser.add_argument("--random-state", type=int, default=42)

    bad_case_parser = subparsers.add_parser(
        "compare-badcases",
        help="Run bad-case driven risk-score and threshold tuning",
    )
    bad_case_parser.add_argument("--data", required=True)
    bad_case_parser.add_argument("--adversarial", required=True)
    bad_case_parser.add_argument("--out-csv", default="docs/experiments/classic/bad_case_optimization.csv")
    bad_case_parser.add_argument("--out-md", default="docs/experiments/classic/bad_case_optimization.md")
    bad_case_parser.add_argument("--grid-csv", default="docs/experiments/classic/bad_case_tuning_grid.csv")
    bad_case_parser.add_argument("--test-size", type=float, default=0.3)
    bad_case_parser.add_argument("--validation-size", type=float, default=0.2)
    bad_case_parser.add_argument("--random-state", type=int, default=42)

    plot_parser = subparsers.add_parser(
        "plot-comparison",
        help="Generate report-ready model comparison figure and summary",
    )
    plot_parser.add_argument("--input", default="docs/experiments/classic/bad_case_optimization.csv")
    plot_parser.add_argument("--out-svg", default="docs/figures/classic/model_comparison.svg")
    plot_parser.add_argument("--out-md", default="docs/reports/summary/report_summary.md")

    fusion_parser = subparsers.add_parser(
        "compare-fusions",
        help="Compare v4 with a max-score fusion candidate",
    )
    fusion_parser.add_argument("--data", required=True)
    fusion_parser.add_argument("--adversarial", required=True)
    fusion_parser.add_argument("--out-csv", default="docs/experiments/classic/fusion_experiment.csv")
    fusion_parser.add_argument("--out-md", default="docs/experiments/classic/fusion_experiment.md")
    fusion_parser.add_argument("--stability-csv", default="docs/experiments/classic/fusion_stability.csv")
    fusion_parser.add_argument("--test-size", type=float, default=0.3)
    fusion_parser.add_argument("--validation-size", type=float, default=0.2)
    fusion_parser.add_argument("--random-state", type=int, default=42)

    multidata_parser = subparsers.add_parser(
        "validate-multidata",
        help="Validate v4/v5 on the main holdout plus named external datasets",
    )
    multidata_parser.add_argument("--train-data", required=True)
    multidata_parser.add_argument(
        "--eval-data",
        action="append",
        default=[],
        help="Named eval set in name=path form. Can be repeated.",
    )
    multidata_parser.add_argument(
        "--out-csv",
        default="docs/experiments/multidataset/multidataset_fusion_validation.csv",
    )
    multidata_parser.add_argument(
        "--out-md",
        default="docs/experiments/multidataset/multidataset_fusion_validation.md",
    )
    multidata_parser.add_argument("--test-size", type=float, default=0.3)
    multidata_parser.add_argument("--validation-size", type=float, default=0.2)
    multidata_parser.add_argument("--random-state", type=int, default=42)

    all_versions_parser = subparsers.add_parser(
        "validate-all-versions",
        help="Validate v0-v5 on the main holdout plus named external datasets",
    )
    all_versions_parser.add_argument("--train-data", required=True)
    all_versions_parser.add_argument(
        "--eval-data",
        action="append",
        default=[],
        help="Named eval set in name=path form. Can be repeated.",
    )
    all_versions_parser.add_argument(
        "--out-csv",
        default="docs/experiments/multidataset/all_versions_multidataset_validation.csv",
    )
    all_versions_parser.add_argument(
        "--out-md",
        default="docs/experiments/multidataset/all_versions_multidataset_validation.md",
    )
    all_versions_parser.add_argument("--test-size", type=float, default=0.3)
    all_versions_parser.add_argument("--validation-size", type=float, default=0.2)
    all_versions_parser.add_argument("--random-state", type=int, default=42)

    adaptation_parser = subparsers.add_parser(
        "validate-domain-adaptation",
        help="Compare main-only models with v6 trained on held-out external adaptation data",
    )
    adaptation_parser.add_argument("--train-data", required=True)
    adaptation_parser.add_argument(
        "--adapt-data",
        action="append",
        default=[],
        help="External binary dataset in name=path form. Can be repeated.",
    )
    adaptation_parser.add_argument(
        "--challenge-data",
        action="append",
        default=[],
        help="Challenge-only eval dataset in name=path form. Can be repeated.",
    )
    adaptation_parser.add_argument(
        "--out-csv",
        default="docs/experiments/domain_adaptation/domain_adaptation_validation.csv",
    )
    adaptation_parser.add_argument(
        "--out-md",
        default="docs/experiments/domain_adaptation/domain_adaptation_validation.md",
    )
    adaptation_parser.add_argument(
        "--split-csv",
        default="docs/experiments/domain_adaptation/domain_adaptation_splits.csv",
    )
    adaptation_parser.add_argument("--test-size", type=float, default=0.3)
    adaptation_parser.add_argument("--validation-size", type=float, default=0.2)
    adaptation_parser.add_argument("--adapt-train-size", type=float, default=0.3)
    adaptation_parser.add_argument("--random-state", type=int, default=42)

    protocol_parser = subparsers.add_parser(
        "validate-protocols",
        help="Run v0-v7 through fixed evaluation protocols A/B/C/D",
    )
    protocol_parser.add_argument("--train-data", required=True)
    protocol_parser.add_argument(
        "--external-data",
        action="append",
        default=[],
        help="External binary dataset in name=path form. Can be repeated.",
    )
    protocol_parser.add_argument(
        "--challenge-data",
        action="append",
        default=[],
        help="Challenge-only eval dataset in name=path form. Can be repeated.",
    )
    protocol_parser.add_argument(
        "--out-csv",
        default="docs/experiments/multidataset/evaluation_protocol_results.csv",
    )
    protocol_parser.add_argument(
        "--out-md",
        default="docs/experiments/multidataset/evaluation_protocol_results.md",
    )
    protocol_parser.add_argument(
        "--split-csv",
        default="docs/experiments/multidataset/evaluation_protocol_splits.csv",
    )
    protocol_parser.add_argument("--test-size", type=float, default=0.3)
    protocol_parser.add_argument("--validation-size", type=float, default=0.2)
    protocol_parser.add_argument("--adapt-train-size", type=float, default=0.3)
    protocol_parser.add_argument("--random-state", type=int, default=42)

    semantic_parser = subparsers.add_parser(
        "validate-semantic-v8",
        help="Run v8.0 frozen semantic encoder through protocols A/B/C/D",
    )
    semantic_parser.add_argument("--train-data", required=True)
    semantic_parser.add_argument(
        "--external-data",
        action="append",
        default=[],
        help="External binary dataset in name=path form. Can be repeated.",
    )
    semantic_parser.add_argument(
        "--challenge-data",
        action="append",
        default=[],
        help="Challenge-only eval dataset in name=path form. Can be repeated.",
    )
    semantic_parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    semantic_parser.add_argument("--batch-size", type=int, default=64)
    semantic_parser.add_argument("--device")
    semantic_parser.add_argument(
        "--out-csv",
        default="docs/experiments/semantic_v8/semantic_v8_protocol_results.csv",
    )
    semantic_parser.add_argument(
        "--out-md",
        default="docs/experiments/semantic_v8/semantic_v8_protocol_results.md",
    )
    semantic_parser.add_argument(
        "--split-csv",
        default="docs/experiments/semantic_v8/semantic_v8_protocol_splits.csv",
    )
    semantic_parser.add_argument("--test-size", type=float, default=0.3)
    semantic_parser.add_argument("--validation-size", type=float, default=0.2)
    semantic_parser.add_argument("--adapt-train-size", type=float, default=0.3)
    semantic_parser.add_argument("--random-state", type=int, default=42)

    semantic_diagnostics_parser = subparsers.add_parser(
        "diagnose-semantic-v8",
        help="Run v8.1 semantic threshold calibration diagnostics",
    )
    semantic_diagnostics_parser.add_argument("--train-data", required=True)
    semantic_diagnostics_parser.add_argument(
        "--external-data",
        action="append",
        default=[],
        help="External binary dataset in name=path form. Can be repeated.",
    )
    semantic_diagnostics_parser.add_argument(
        "--challenge-data",
        action="append",
        default=[],
        help="Challenge-only eval dataset in name=path form. Can be repeated.",
    )
    semantic_diagnostics_parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    semantic_diagnostics_parser.add_argument("--batch-size", type=int, default=64)
    semantic_diagnostics_parser.add_argument("--device")
    semantic_diagnostics_parser.add_argument(
        "--diagnostics-csv",
        default="docs/experiments/semantic_v8/semantic_v8_calibration_diagnostics.csv",
    )
    semantic_diagnostics_parser.add_argument(
        "--threshold-grid-csv",
        default="docs/experiments/semantic_v8/semantic_v8_threshold_grid.csv",
    )
    semantic_diagnostics_parser.add_argument(
        "--score-samples-csv",
        default="docs/experiments/semantic_v8/semantic_v8_score_samples.csv",
    )
    semantic_diagnostics_parser.add_argument(
        "--pr-curve-csv",
        default="docs/experiments/semantic_v8/semantic_v8_pr_curve.csv",
    )
    semantic_diagnostics_parser.add_argument(
        "--out-md",
        default="docs/experiments/semantic_v8/semantic_v8_calibration_diagnostics.md",
    )
    semantic_diagnostics_parser.add_argument(
        "--threshold-figure",
        default="docs/figures/semantic_v8/semantic_v8_threshold_gain.svg",
    )
    semantic_diagnostics_parser.add_argument(
        "--score-figure",
        default="docs/figures/semantic_v8/semantic_v8_score_distribution.svg",
    )
    semantic_diagnostics_parser.add_argument("--test-size", type=float, default=0.3)
    semantic_diagnostics_parser.add_argument("--validation-size", type=float, default=0.2)
    semantic_diagnostics_parser.add_argument("--adapt-train-size", type=float, default=0.3)
    semantic_diagnostics_parser.add_argument("--random-state", type=int, default=42)

    semantic_encoder_parser = subparsers.add_parser(
        "compare-semantic-encoders-v8",
        help="Run v8.2 encoder comparison under protocols A/B/C/D",
    )
    semantic_encoder_parser.add_argument("--train-data", required=True)
    semantic_encoder_parser.add_argument(
        "--external-data",
        action="append",
        default=[],
        help="External binary dataset in name=path form. Can be repeated.",
    )
    semantic_encoder_parser.add_argument(
        "--challenge-data",
        action="append",
        default=[],
        help="Challenge-only eval dataset in name=path form. Can be repeated.",
    )
    semantic_encoder_parser.add_argument(
        "--encoder",
        action="append",
        default=[],
        help=(
            "Encoder in name=model_id form. Can be repeated. "
            "Defaults to bge_small_zh, multilingual_minilm, and m3e_small."
        ),
    )
    semantic_encoder_parser.add_argument("--batch-size", type=int, default=64)
    semantic_encoder_parser.add_argument("--device")
    semantic_encoder_parser.add_argument(
        "--out-csv",
        default="docs/experiments/semantic_v8/semantic_v8_encoder_comparison.csv",
    )
    semantic_encoder_parser.add_argument(
        "--out-md",
        default="docs/experiments/semantic_v8/semantic_v8_encoder_comparison.md",
    )
    semantic_encoder_parser.add_argument(
        "--error-csv",
        default="docs/experiments/semantic_v8/semantic_v8_encoder_errors.csv",
    )
    semantic_encoder_parser.add_argument(
        "--figure",
        default="docs/figures/semantic_v8/semantic_v8_encoder_comparison.svg",
    )
    semantic_encoder_parser.add_argument("--test-size", type=float, default=0.3)
    semantic_encoder_parser.add_argument("--validation-size", type=float, default=0.2)
    semantic_encoder_parser.add_argument("--adapt-train-size", type=float, default=0.3)
    semantic_encoder_parser.add_argument("--random-state", type=int, default=42)
    semantic_encoder_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one encoder fails to load or run.",
    )

    semantic_autoaug_parser = subparsers.add_parser(
        "compare-semantic-autoaug-v8",
        help="Run v8.3a automatic hard-case augmentation comparison",
    )
    semantic_autoaug_parser.add_argument("--train-data", required=True)
    semantic_autoaug_parser.add_argument(
        "--external-data",
        action="append",
        default=[],
        help="External binary dataset in name=path form. Can be repeated.",
    )
    semantic_autoaug_parser.add_argument(
        "--challenge-data",
        action="append",
        default=[],
        help="Challenge-only eval dataset in name=path form. Can be repeated.",
    )
    semantic_autoaug_parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    semantic_autoaug_parser.add_argument("--batch-size", type=int, default=64)
    semantic_autoaug_parser.add_argument("--device")
    semantic_autoaug_parser.add_argument(
        "--out-csv",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_results.csv",
    )
    semantic_autoaug_parser.add_argument(
        "--out-md",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_results.md",
    )
    semantic_autoaug_parser.add_argument(
        "--terms-csv",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_terms.csv",
    )
    semantic_autoaug_parser.add_argument(
        "--examples-csv",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_examples.csv",
    )
    semantic_autoaug_parser.add_argument(
        "--figure",
        default="docs/figures/semantic_v8/semantic_v8_autoaug_delta.svg",
    )
    semantic_autoaug_parser.add_argument("--test-size", type=float, default=0.3)
    semantic_autoaug_parser.add_argument("--validation-size", type=float, default=0.2)
    semantic_autoaug_parser.add_argument("--adapt-train-size", type=float, default=0.3)
    semantic_autoaug_parser.add_argument("--random-state", type=int, default=42)
    semantic_autoaug_parser.add_argument("--max-terms", type=int, default=80)
    semantic_autoaug_parser.add_argument("--max-augmented", type=int, default=200)
    semantic_autoaug_parser.add_argument("--min-spam-df", type=int, default=3)

    semantic_filtered_autoaug_parser = subparsers.add_parser(
        "compare-semantic-filtered-autoaug-v8",
        help="Run v8.3b filtered automatic augmentation plus hard negatives",
    )
    semantic_filtered_autoaug_parser.add_argument("--train-data", required=True)
    semantic_filtered_autoaug_parser.add_argument(
        "--external-data",
        action="append",
        default=[],
        help="External binary dataset in name=path form. Can be repeated.",
    )
    semantic_filtered_autoaug_parser.add_argument(
        "--challenge-data",
        action="append",
        default=[],
        help="Challenge-only eval dataset in name=path form. Can be repeated.",
    )
    semantic_filtered_autoaug_parser.add_argument("--model-name", default="BAAI/bge-small-zh-v1.5")
    semantic_filtered_autoaug_parser.add_argument("--batch-size", type=int, default=64)
    semantic_filtered_autoaug_parser.add_argument("--device")
    semantic_filtered_autoaug_parser.add_argument(
        "--out-csv",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_results.csv",
    )
    semantic_filtered_autoaug_parser.add_argument(
        "--out-md",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_results.md",
    )
    semantic_filtered_autoaug_parser.add_argument(
        "--terms-csv",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_terms.csv",
    )
    semantic_filtered_autoaug_parser.add_argument(
        "--examples-csv",
        default="docs/experiments/semantic_v8/semantic_v8_autoaug_filtered_examples.csv",
    )
    semantic_filtered_autoaug_parser.add_argument(
        "--figure",
        default="docs/figures/semantic_v8/semantic_v8_autoaug_filtered_delta.svg",
    )
    semantic_filtered_autoaug_parser.add_argument("--test-size", type=float, default=0.3)
    semantic_filtered_autoaug_parser.add_argument("--validation-size", type=float, default=0.2)
    semantic_filtered_autoaug_parser.add_argument("--adapt-train-size", type=float, default=0.3)
    semantic_filtered_autoaug_parser.add_argument("--random-state", type=int, default=42)
    semantic_filtered_autoaug_parser.add_argument("--max-terms", type=int, default=80)
    semantic_filtered_autoaug_parser.add_argument("--max-augmented", type=int, default=200)
    semantic_filtered_autoaug_parser.add_argument("--min-spam-df", type=int, default=3)
    semantic_filtered_autoaug_parser.add_argument("--max-hard-negatives", type=int, default=200)
    semantic_filtered_autoaug_parser.add_argument("--positive-min-score", type=float, default=0.05)
    semantic_filtered_autoaug_parser.add_argument("--positive-max-score", type=float, default=0.75)
    semantic_filtered_autoaug_parser.add_argument("--hard-negative-min-score", type=float, default=0.25)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        result = train_baseline(args.data, args.model, args.analyzer)
        print(f"Saved model to: {result.model_path}")
        print("Metrics:")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True))
        print("Classification report:")
        print(result.report)
        print("Confusion matrix:")
        print(result.confusion)
        return

    if args.command == "prepare-data":
        data = prepare_labeled_dataset(
            args.raw,
            args.out,
            sample_size=args.sample_size,
            random_state=args.random_state,
        )
        print(f"Saved dataset to: {args.out}")
        print(f"Rows: {len(data)}")
        print("Label counts:")
        print(data["label"].value_counts().sort_index().to_string())
        return

    if args.command == "prepare-ast":
        data = prepare_ast_dataset(
            args.raw,
            args.out,
            sample_size=args.sample_size,
            random_state=args.random_state,
        )
        summary = dataset_summary(data)
        print(f"Saved AST dataset to: {args.out}")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        if summary != {"rows": 16008, "normal": 5000, "spam": 11008}:
            print("Warning: dataset counts differ from the course PDF AST description.")
        return

    if args.command == "prepare-fbs-mixed":
        data = prepare_fbs_mixed_dataset(
            args.fbs_dir,
            args.normal_raw,
            args.out,
            sample_size=args.sample_size,
            spam_ratio=args.spam_ratio,
            exclude_path=args.exclude,
            random_state=args.random_state,
        )
        print(f"Saved FBS mixed dataset to: {args.out}")
        print(json.dumps(dataset_summary(data), ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "prepare-hf":
        data = prepare_huggingface_dataset(
            args.dataset,
            args.out,
            token=None if args.no_token else True,
        )
        print(f"Saved Hugging Face dataset to: {args.out}")
        print(json.dumps(dataset_summary(data), ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "generate-adversarial":
        data = generate_adversarial_dataset(
            args.data,
            args.out,
            max_samples=args.max_samples,
            random_state=args.random_state,
        )
        print(f"Saved adversarial dataset to: {args.out}")
        print(f"Rows: {len(data)}")
        if len(data) > 0:
            print("Examples:")
            print(data[["variant", "text"]].head(5).to_string(index=False))
        return

    if args.command == "generate-keyword-challenge":
        data = generate_keyword_challenge_dataset(args.out)
        print(f"Saved keyword challenge dataset to: {args.out}")
        print(f"Rows: {len(data)}")
        print("Examples:")
        print(data[["variant", "text"]].head(8).to_string(index=False))
        return

    if args.command == "predict":
        label = predict_text(args.text, args.model)
        name = "垃圾文本" if label == 1 else "正常文本"
        print(f"{label}\t{name}")
        return

    if args.command == "evaluate":
        result = evaluate_model(args.data, args.model)
        print("Metrics:")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True))
        print("Classification report:")
        print(result.report)
        print("Confusion matrix:")
        print(result.confusion)
        return

    if args.command == "compare-baselines":
        results = compare_baselines(
            data_path=args.data,
            output_csv=args.out_csv,
            output_md=args.out_md,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(
            results[
                [
                    "name",
                    "accuracy",
                    "macro_f1",
                    "f1_spam",
                    "recall_spam",
                    "pr_auc",
                    "recall_at_precision_95",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "compare-csn":
        results = compare_csn_optimization(
            data_path=args.data,
            adversarial_path=args.adversarial,
            output_csv=args.out_csv,
            output_md=args.out_md,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(
            results[
                [
                    "name",
                    "clean_accuracy",
                    "clean_f1_spam",
                    "adv_recall_spam",
                    "adv_f1_spam",
                    "adv_false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "compare-badcases":
        results = compare_bad_case_optimization(
            data_path=args.data,
            adversarial_path=args.adversarial,
            output_csv=args.out_csv,
            output_md=args.out_md,
            grid_csv=args.grid_csv,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved tuning grid to: {args.grid_csv}")
        print(
            results[
                [
                    "name",
                    "risk_bonus",
                    "threshold",
                    "clean_accuracy",
                    "clean_f1_spam",
                    "clean_false_positive",
                    "clean_false_negative",
                    "adv_recall_spam",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "plot-comparison":
        figure, summary = generate_report_assets(
            input_csv=args.input,
            output_svg=args.out_svg,
            output_md=args.out_md,
        )
        print(f"Saved figure to: {figure}")
        print(f"Saved summary to: {summary}")
        return

    if args.command == "compare-fusions":
        results = compare_score_fusion_optimization(
            data_path=args.data,
            adversarial_path=args.adversarial,
            output_csv=args.out_csv,
            output_md=args.out_md,
            stability_csv=args.stability_csv,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved stability CSV to: {args.stability_csv}")
        print(
            results[
                [
                    "name",
                    "risk_bonus",
                    "threshold",
                    "clean_accuracy",
                    "clean_f1_spam",
                    "clean_false_positive",
                    "clean_false_negative",
                    "adv_recall_spam",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "validate-multidata":
        results = compare_multidataset_fusion_validation(
            train_data_path=args.train_data,
            eval_data_paths=_parse_named_paths(args.eval_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(
            results[
                [
                    "dataset",
                    "name",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "false_positive",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "validate-all-versions":
        results = compare_all_versions_multidataset_validation(
            train_data_path=args.train_data,
            eval_data_paths=_parse_named_paths(args.eval_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(
            results[
                [
                    "dataset",
                    "name",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "false_positive",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "validate-domain-adaptation":
        results = compare_domain_adaptation_validation(
            train_data_path=args.train_data,
            adapt_data_paths=_parse_named_paths(args.adapt_data),
            challenge_data_paths=_parse_named_paths(args.challenge_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            adaptation_summary_csv=args.split_csv,
            test_size=args.test_size,
            validation_size=args.validation_size,
            adapt_train_size=args.adapt_train_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved split summary to: {args.split_csv}")
        print(
            results[
                [
                    "dataset",
                    "name",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "false_positive",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "validate-protocols":
        results = compare_evaluation_protocols(
            train_data_path=args.train_data,
            external_data_paths=_parse_named_paths(args.external_data),
            challenge_data_paths=_parse_named_paths(args.challenge_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            split_csv=args.split_csv,
            test_size=args.test_size,
            validation_size=args.validation_size,
            adapt_train_size=args.adapt_train_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved split summary to: {args.split_csv}")
        print(
            results[
                [
                    "protocol_id",
                    "dataset",
                    "model_version",
                    "training_scope",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "false_positive",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "validate-semantic-v8":
        results = compare_semantic_v8_protocols(
            train_data_path=args.train_data,
            external_data_paths=_parse_named_paths(args.external_data),
            challenge_data_paths=_parse_named_paths(args.challenge_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            split_csv=args.split_csv,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            test_size=args.test_size,
            validation_size=args.validation_size,
            adapt_train_size=args.adapt_train_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved split summary to: {args.split_csv}")
        print(
            results[
                [
                    "protocol_id",
                    "dataset",
                    "model_version",
                    "training_scope",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "false_positive",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "diagnose-semantic-v8":
        results = diagnose_semantic_v8_calibration(
            train_data_path=args.train_data,
            external_data_paths=_parse_named_paths(args.external_data),
            challenge_data_paths=_parse_named_paths(args.challenge_data),
            diagnostics_csv=args.diagnostics_csv,
            threshold_grid_csv=args.threshold_grid_csv,
            score_samples_csv=args.score_samples_csv,
            pr_curve_csv=args.pr_curve_csv,
            output_md=args.out_md,
            threshold_figure=args.threshold_figure,
            score_figure=args.score_figure,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            test_size=args.test_size,
            validation_size=args.validation_size,
            adapt_train_size=args.adapt_train_size,
            random_state=args.random_state,
        )
        print(f"Saved diagnostics CSV to: {args.diagnostics_csv}")
        print(f"Saved threshold grid to: {args.threshold_grid_csv}")
        print(f"Saved score samples to: {args.score_samples_csv}")
        print(f"Saved PR curve to: {args.pr_curve_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved figures to: {args.threshold_figure}, {args.score_figure}")
        print(
            results[
                [
                    "protocol_id",
                    "dataset",
                    "model_version",
                    "threshold_current",
                    "threshold_oracle",
                    "current_f1_spam",
                    "oracle_f1_spam",
                    "oracle_f1_gain",
                    "pr_auc",
                    "diagnosis",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "compare-semantic-encoders-v8":
        encoders = _parse_named_paths(args.encoder) if args.encoder else list(DEFAULT_ENCODERS)
        results = compare_semantic_v8_encoders(
            train_data_path=args.train_data,
            external_data_paths=_parse_named_paths(args.external_data),
            challenge_data_paths=_parse_named_paths(args.challenge_data),
            encoders=encoders,
            output_csv=args.out_csv,
            output_md=args.out_md,
            error_csv=args.error_csv,
            figure_path=args.figure,
            batch_size=args.batch_size,
            device=args.device,
            test_size=args.test_size,
            validation_size=args.validation_size,
            adapt_train_size=args.adapt_train_size,
            random_state=args.random_state,
            continue_on_error=not args.fail_fast,
        )
        print(f"Saved encoder comparison CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved encoder errors to: {args.error_csv}")
        print(f"Saved figure to: {args.figure}")
        print(
            results[
                [
                    "protocol_id",
                    "dataset",
                    "model_version",
                    "encoder_name",
                    "training_scope",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "pr_auc",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "compare-semantic-autoaug-v8":
        results = compare_semantic_v8_auto_augmentation(
            train_data_path=args.train_data,
            external_data_paths=_parse_named_paths(args.external_data),
            challenge_data_paths=_parse_named_paths(args.challenge_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            terms_csv=args.terms_csv,
            examples_csv=args.examples_csv,
            figure_path=args.figure,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            test_size=args.test_size,
            validation_size=args.validation_size,
            adapt_train_size=args.adapt_train_size,
            random_state=args.random_state,
            max_terms=args.max_terms,
            max_augmented=args.max_augmented,
            min_spam_df=args.min_spam_df,
        )
        print(f"Saved auto-augmentation CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved mined terms to: {args.terms_csv}")
        print(f"Saved generated examples to: {args.examples_csv}")
        print(f"Saved figure to: {args.figure}")
        print(
            results[
                [
                    "protocol_id",
                    "dataset",
                    "model_version",
                    "training_scope",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "false_positive",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    if args.command == "compare-semantic-filtered-autoaug-v8":
        results = compare_semantic_v8_filtered_auto_augmentation(
            train_data_path=args.train_data,
            external_data_paths=_parse_named_paths(args.external_data),
            challenge_data_paths=_parse_named_paths(args.challenge_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            terms_csv=args.terms_csv,
            examples_csv=args.examples_csv,
            figure_path=args.figure,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            test_size=args.test_size,
            validation_size=args.validation_size,
            adapt_train_size=args.adapt_train_size,
            random_state=args.random_state,
            max_terms=args.max_terms,
            max_augmented=args.max_augmented,
            min_spam_df=args.min_spam_df,
            max_hard_negatives=args.max_hard_negatives,
            positive_min_score=args.positive_min_score,
            positive_max_score=args.positive_max_score,
            hard_negative_min_score=args.hard_negative_min_score,
        )
        print(f"Saved filtered auto-augmentation CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        print(f"Saved mined terms to: {args.terms_csv}")
        print(f"Saved generated examples to: {args.examples_csv}")
        print(f"Saved figure to: {args.figure}")
        print(
            results[
                [
                    "protocol_id",
                    "dataset",
                    "model_version",
                    "training_scope",
                    "accuracy",
                    "precision_spam",
                    "recall_spam",
                    "f1_spam",
                    "false_positive",
                    "false_negative",
                ]
            ].to_string(index=False)
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
