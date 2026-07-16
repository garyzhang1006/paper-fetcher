"""Offline evaluation helpers for Tutorial 2 feature extraction."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Evidence, PaperFeatures


def set_metrics(gold: Iterable[str], predicted: Iterable[str]) -> dict[str, float | int]:
    """Calculate exact-match precision, recall, and F1 for set-valued fields."""

    gold_set = set(gold)
    predicted_set = set(predicted)
    true_positives = len(gold_set & predicted_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evidence_coverage(features: PaperFeatures) -> float:
    """Return fraction of semantic feature values backed by matching evidence."""

    semantic_fields = (
        "research_tasks",
        "methods",
        "datasets",
        "metrics",
        "domains",
        "contributions",
        "limitations",
    )
    values = {
        (field, value.casefold())
        for field in semantic_fields
        for value in getattr(features, field)
    }
    if not values:
        return 1.0
    supported = {
        (item.field, item.value.casefold())
        for item in features.evidence
        if isinstance(item, Evidence)
    }
    return len(values & supported) / len(values)
