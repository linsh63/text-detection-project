#!/usr/bin/env python3
"""Predict whether one text is spam."""

from __future__ import annotations

import argparse

from text_detection.modeling import predict_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", default="models/baseline.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = predict_text(args.text, args.model)
    name = "垃圾文本" if label == 1 else "正常文本"
    print(f"{label}\t{name}")


if __name__ == "__main__":
    main()

