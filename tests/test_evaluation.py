import pytest

from arxiv_kg.evaluation import evidence_coverage, set_metrics
from arxiv_kg.models import Evidence, PaperFeatures


def test_set_metrics_matches_hand_calculation():
    metrics = set_metrics(
        {"diffusion model", "graph neural network"},
        {"diffusion model", "transformer"},
    )

    assert metrics == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_set_metrics_treats_two_empty_sets_as_perfect():
    assert set_metrics(set(), set()) == {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }


def test_evidence_coverage_counts_exact_field_value_pairs():
    features = PaperFeatures(
        one_sentence_summary="A fictional result.",
        methods=["diffusion model"],
        datasets=["CIFAR-10"],
        evidence=[
            Evidence(
                field="methods",
                value="diffusion model",
                statement="Authors train a diffusion model.",
            )
        ],
        confidence=0.8,
    )

    assert evidence_coverage(features) == pytest.approx(0.5)


def test_evidence_coverage_is_one_when_no_semantic_values_exist():
    features = PaperFeatures(
        one_sentence_summary="No semantic features claimed.",
        confidence=0.2,
    )

    assert evidence_coverage(features) == 1.0
