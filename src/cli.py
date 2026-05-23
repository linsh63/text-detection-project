"""Command-line interface for training and inference."""

from __future__ import annotations

import argparse
import json

from .adversarial import generate_adversarial_dataset, generate_keyword_challenge_dataset
from .data import dataset_summary, prepare_ast_dataset, prepare_labeled_dataset
from .experiments import (
    compare_bad_case_optimization,
    compare_baselines,
    compare_csn_optimization,
    compare_score_fusion_optimization,
)
from .modeling import evaluate_model, predict_text, train_baseline
from .visualization import generate_report_assets


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
    compare_parser.add_argument("--out-csv", default="docs/baseline_comparison.csv")
    compare_parser.add_argument("--out-md", default="docs/baseline_comparison.md")
    compare_parser.add_argument("--test-size", type=float, default=0.3)
    compare_parser.add_argument("--random-state", type=int, default=42)

    csn_parser = subparsers.add_parser("compare-csn", help="Compare CSN optimization")
    csn_parser.add_argument("--data", required=True)
    csn_parser.add_argument("--adversarial", required=True)
    csn_parser.add_argument("--out-csv", default="docs/csn_comparison.csv")
    csn_parser.add_argument("--out-md", default="docs/csn_comparison.md")
    csn_parser.add_argument("--test-size", type=float, default=0.3)
    csn_parser.add_argument("--random-state", type=int, default=42)

    bad_case_parser = subparsers.add_parser(
        "compare-badcases",
        help="Run bad-case driven risk-score and threshold tuning",
    )
    bad_case_parser.add_argument("--data", required=True)
    bad_case_parser.add_argument("--adversarial", required=True)
    bad_case_parser.add_argument("--out-csv", default="docs/bad_case_optimization.csv")
    bad_case_parser.add_argument("--out-md", default="docs/bad_case_optimization.md")
    bad_case_parser.add_argument("--grid-csv", default="docs/bad_case_tuning_grid.csv")
    bad_case_parser.add_argument("--test-size", type=float, default=0.3)
    bad_case_parser.add_argument("--validation-size", type=float, default=0.2)
    bad_case_parser.add_argument("--random-state", type=int, default=42)

    plot_parser = subparsers.add_parser(
        "plot-comparison",
        help="Generate report-ready model comparison figure and summary",
    )
    plot_parser.add_argument("--input", default="docs/bad_case_optimization.csv")
    plot_parser.add_argument("--out-svg", default="docs/figures/model_comparison.svg")
    plot_parser.add_argument("--out-md", default="docs/report_summary.md")

    fusion_parser = subparsers.add_parser(
        "compare-fusions",
        help="Compare v4 with a max-score fusion candidate",
    )
    fusion_parser.add_argument("--data", required=True)
    fusion_parser.add_argument("--adversarial", required=True)
    fusion_parser.add_argument("--out-csv", default="docs/fusion_experiment.csv")
    fusion_parser.add_argument("--out-md", default="docs/fusion_experiment.md")
    fusion_parser.add_argument("--stability-csv", default="docs/fusion_stability.csv")
    fusion_parser.add_argument("--test-size", type=float, default=0.3)
    fusion_parser.add_argument("--validation-size", type=float, default=0.2)
    fusion_parser.add_argument("--random-state", type=int, default=42)

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

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
