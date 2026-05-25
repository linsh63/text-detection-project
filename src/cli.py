"""Command-line interface for the v1 text detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import (
    dataset_summary,
    prepare_ast_dataset,
    prepare_fbs_mixed_dataset,
    prepare_huggingface_dataset,
    prepare_labeled_dataset,
)
from .modeling import evaluate_model, predict_text, train_baseline
from .runners import compare_all_versions_multidataset_validation, compare_baselines, validate_v1


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

    train_parser = subparsers.add_parser("train", help="Train v1")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--model", default="models/v1_char_svm.joblib")

    predict_parser = subparsers.add_parser("predict", help="Predict one text")
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--model", default="models/v1_char_svm.joblib")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a saved v1 model")
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--model", default="models/v1_char_svm.joblib")

    v1_parser = subparsers.add_parser("validate-v1", help="Train and evaluate v1")
    v1_parser.add_argument("--train-data", required=True)
    v1_parser.add_argument(
        "--eval-data",
        action="append",
        default=[],
        help="Named eval set in name=path form. Can be repeated.",
    )
    v1_parser.add_argument("--out-csv", default="docs/experiments/multidataset/v1_validation.csv")
    v1_parser.add_argument("--out-md", default="docs/experiments/multidataset/v1_validation.md")
    v1_parser.add_argument("--test-size", type=float, default=0.3)
    v1_parser.add_argument("--random-state", type=int, default=42)

    compare_parser = subparsers.add_parser(
        "compare-baselines",
        help="Compatibility alias for validate-v1 on one dataset",
    )
    compare_parser.add_argument("--data", required=True)
    compare_parser.add_argument("--out-csv", default="docs/experiments/classic/v1_baseline.csv")
    compare_parser.add_argument("--out-md", default="docs/experiments/classic/v1_baseline.md")
    compare_parser.add_argument("--test-size", type=float, default=0.3)
    compare_parser.add_argument("--random-state", type=int, default=42)

    all_versions_parser = subparsers.add_parser(
        "validate-all-versions",
        help="Compatibility alias; this repository now evaluates only v1.",
    )
    all_versions_parser.add_argument("--train-data", required=True)
    all_versions_parser.add_argument(
        "--eval-data",
        action="append",
        default=[],
        help="Named eval set in name=path form. Can be repeated.",
    )
    all_versions_parser.add_argument("--out-csv", default="docs/experiments/multidataset/v1_validation.csv")
    all_versions_parser.add_argument("--out-md", default="docs/experiments/multidataset/v1_validation.md")
    all_versions_parser.add_argument("--test-size", type=float, default=0.3)
    all_versions_parser.add_argument("--validation-size", type=float, default=0.2)
    all_versions_parser.add_argument("--random-state", type=int, default=42)

    return parser


def _print_result_table(results):
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


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "prepare-data":
        data = prepare_labeled_dataset(
            args.raw,
            args.out,
            sample_size=args.sample_size,
            random_state=args.random_state,
        )
        print(f"Saved dataset to: {args.out}")
        print(json.dumps(dataset_summary(data), ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "prepare-ast":
        data = prepare_ast_dataset(
            args.raw,
            args.out,
            sample_size=args.sample_size,
            random_state=args.random_state,
        )
        print(f"Saved AST dataset to: {args.out}")
        print(json.dumps(dataset_summary(data), ensure_ascii=False, indent=2, sort_keys=True))
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

    if args.command == "train":
        result = train_baseline(args.data, args.model)
        print(f"Saved model to: {result.model_path}")
        print("Metrics:")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True))
        print("Classification report:")
        print(result.report)
        print("Confusion matrix:")
        print(result.confusion)
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

    if args.command == "validate-v1":
        results = validate_v1(
            train_data_path=args.train_data,
            eval_data_paths=_parse_named_paths(args.eval_data),
            output_csv=args.out_csv,
            output_md=args.out_md,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Saved CSV to: {args.out_csv}")
        print(f"Saved Markdown to: {args.out_md}")
        _print_result_table(results)
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
        _print_result_table(results)
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
        _print_result_table(results)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
