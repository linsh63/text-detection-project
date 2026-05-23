#!/usr/bin/env python3
"""Train a baseline spam-text detector."""

from __future__ import annotations

import argparse

from text_detection.modeling import train_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/sample_texts.tsv")
    parser.add_argument("--model", default="models/baseline.joblib")
    parser.add_argument("--analyzer", choices=["char", "word"], default="char")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_baseline(args.data, args.model, args.analyzer)
    print(f"Saved model to: {result.model_path}")
    print("Classification report:")
    print(result.report)
    print("Confusion matrix:")
    print(result.confusion)


if __name__ == "__main__":
    main()

