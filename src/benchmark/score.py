from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark.models import _normalize_label


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = sorted({row["expected_label"] for row in rows})
    total = len(rows)
    correct = sum(1 for row in rows if row["predicted_label"] == row["expected_label"])
    accuracy = correct / total if total else 0.0

    per_label_f1: list[float] = []
    for label in labels:
        tp = sum(
            1
            for row in rows
            if row["predicted_label"] == label and row["expected_label"] == label
        )
        fp = sum(
            1
            for row in rows
            if row["predicted_label"] == label and row["expected_label"] != label
        )
        fn = sum(
            1
            for row in rows
            if row["predicted_label"] != label and row["expected_label"] == label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label_f1.append(f1)

    macro_f1 = sum(per_label_f1) / len(per_label_f1) if per_label_f1 else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "total": float(total),
        "correct": float(correct),
    }


def parse_label(text: str) -> str:
    return _normalize_label(text)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]
