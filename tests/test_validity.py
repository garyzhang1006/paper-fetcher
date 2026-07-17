import pytest

from arxiv_kg.models import PaperFeatures
from arxiv_kg.validity import extract_validity_envelopes


def test_extracts_evidence_backed_comparative_validity_envelope():
    abstract = (
        "Compared with ResNet-50, our results show 4.2 percentage points higher "
        "accuracy on CIFAR-10 under distribution shift (p < 0.01). "
        "A limitation is evaluation on one benchmark."
    )

    envelopes = extract_validity_envelopes(abstract)

    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope.claim == abstract.split(". A limitation", 1)[0] + "."
    assert envelope.claim_type == "comparative"
    assert envelope.direction == "positive"
    assert envelope.comparators == ["ResNet-50"]
    assert envelope.evaluation_contexts == ["CIFAR-10"]
    assert envelope.metrics == ["accuracy"]
    assert envelope.reported_values == ["4.2 percentage points"]
    assert envelope.effect_sizes == ["4.2 percentage points"]
    assert envelope.uncertainty == ["p < 0.01"]
    assert envelope.conditions == ["under distribution shift"]
    assert envelope.paper_level_boundaries == [
        "A limitation is evaluation on one benchmark."
    ]
    assert envelope.evidence.source == "abstract"
    assert envelope.evidence.sentence_index == 0
    assert envelope.evidence.statement == envelope.claim
    assert 0.0 <= envelope.confidence <= 1.0


def test_returns_empty_when_abstract_has_no_explicit_result_claim():
    assert extract_validity_envelopes(
        "This paper introduces a framework for analyzing complex systems."
    ) == []


def test_splits_result_sentence_when_arxiv_text_omits_space_after_period():
    envelopes = extract_validity_envelopes(
        "Background description without a claim.Our results show improved accuracy."
    )

    assert len(envelopes) == 1
    assert envelopes[0].claim == "Our results show improved accuracy."


def test_does_not_promote_related_work_to_paper_claim():
    assert extract_validity_envelopes(
        "Prior work shows improved accuracy on CIFAR-10. We describe our setup."
    ) == []


def test_separates_chained_comparators():
    envelopes = extract_validity_envelopes(
        "Experimental results show improved image quality compared to sparse-view "
        "reconstruction and can outperform clinical complete view (CCV) "
        "reconstruction under certain conditions."
    )

    assert envelopes[0].comparators == [
        "sparse-view reconstruction",
        "clinical complete view (CCV) reconstruction",
    ]


@pytest.mark.parametrize("connector", ["but also", "but even"])
def test_not_only_comparison_claim_is_not_truncated(connector):
    envelopes = extract_validity_envelopes(
        f"We find that our model outperforms not only baseline A {connector} baseline B."
    )

    assert envelopes[0].claim.endswith(f"{connector} baseline B.")
    assert envelopes[0].comparators == ["baseline A", "baseline B"]


def test_comparator_does_not_swallow_evaluation_context():
    envelopes = extract_validity_envelopes(
        "Our results show improved accuracy compared to ResNet-50 on CIFAR-10."
    )

    assert envelopes[0].comparators == ["ResNet-50"]
    assert envelopes[0].evaluation_contexts == ["CIFAR-10"]


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        (
            "Our results show improved accuracy compared to GTCRN while requiring only 40x less memory.",
            ["GTCRN"],
        ),
        (
            "We find that our model outperforms unprotected baselines while remaining competitive.",
            ["unprotected baselines"],
        ),
    ],
)
def test_comparator_does_not_swallow_following_result_clause(claim, expected):
    envelopes = extract_validity_envelopes(claim)

    assert envelopes[0].comparators == expected


def test_distinguishes_absolute_result_values_from_effect_sizes():
    envelopes = extract_validity_envelopes(
        "Results show a success rate of 71.3%, outperforming baselines by 11.9% "
        "to 27.6%."
    )

    assert envelopes[0].reported_values == ["71.3%", "11.9%", "27.6%"]
    assert envelopes[0].effect_sizes == ["by 11.9% to 27.6%"]


def test_records_hyphenated_fold_value_and_effect():
    envelopes = extract_validity_envelopes(
        "Our results show a 104-fold speedup over the baseline."
    )

    assert envelopes[0].reported_values == ["104-fold"]
    assert envelopes[0].effect_sizes == ["104-fold speedup"]


@pytest.mark.parametrize(
    ("claim", "reported_value"),
    [
        (r"Our results show a {5\%} gain.", r"{5\%}"),
        ("Our results show the method is 4-10x faster.", "4-10x"),
        ("Our results show the method is 40x faster.", "40x"),
        ("Our results show the method is 1.2x faster.", "1.2x"),
        ("Our results show a speedup by a factor of 3.", "factor of 3"),
        ("Our results show a signal gain of 23 dB.", "23 dB"),
        ("Our results show an error of 3.29mm.", "3.29mm"),
        ("Our results show ROC-AUC 0.99.", "0.99"),
    ],
)
def test_records_common_arxiv_numeric_notation(claim, reported_value):
    envelopes = extract_validity_envelopes(claim)

    assert reported_value in envelopes[0].reported_values


def test_extracts_decimal_confidence_interval_scientific_p_and_latex_plus_minus():
    envelopes = extract_validity_envelopes(
        r"Our results show a 95% confidence interval of 1.1 to 1.5, "
        r"p = 1e-5, and 0.798 \pm 0.045."
    )

    assert envelopes[0].uncertainty == [
        "p = 1e-5",
        "95% confidence interval of 1.1 to 1.5",
        r"0.798 \pm 0.045",
    ]


def test_extracts_latex_plus_minus_with_math_delimiters_and_percent():
    envelopes = extract_validity_envelopes(
        r"Our results show an error of -0.07% $\pm$ 0.01%."
    )

    assert envelopes[0].uncertainty == [r"-0.07% $\pm$ 0.01%"]


def test_keeps_comma_separated_condition_list():
    envelopes = extract_validity_envelopes(
        "Across convolutional, graph, Transformer, and hybrid architectures, "
        "our results show improved accuracy."
    )

    assert envelopes[0].conditions == [
        "Across convolutional, graph, Transformer, and hybrid architectures"
    ]


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        (
            "When the signal is low, while operating with lower compute, our results show improved accuracy.",
            "When the signal is low",
        ),
        (
            "Across four benchmarks and the results show that our framework achieves improved accuracy.",
            "Across four benchmarks",
        ),
    ],
)
def test_condition_does_not_swallow_following_result_clause(claim, expected):
    envelopes = extract_validity_envelopes(claim)

    assert envelopes[0].conditions == [expected]


def test_condition_splits_pronoun_clause_and_keeps_nested_condition():
    envelopes = extract_validity_envelopes(
        "Our results show improved performance when operations are limited to as "
        "few as one to four sites, and it also generalizes to longer chains even "
        "when trained on moderate system sizes."
    )

    assert envelopes[0].conditions == [
        "when operations are limited to as few as one to four sites",
        "when trained on moderate system sizes",
    ]


def test_does_not_attach_prior_method_limitation_as_current_paper_boundary():
    envelopes = extract_validity_envelopes(
        "To address these limitations of existing methods, we propose a new model. "
        "Our results show improved accuracy."
    )

    assert envelopes[0].paper_level_boundaries == []


def test_causal_wording_label_takes_precedence_over_comparison():
    envelopes = extract_validity_envelopes(
        "We show treatment causes improved survival compared with placebo."
    )

    assert envelopes[0].claim_type == "causal_language"


def test_detects_explicit_third_person_current_method_result():
    envelopes = extract_validity_envelopes(
        "Extensive experiments demonstrate that our method achieves 91% accuracy."
    )

    assert len(envelopes) == 1
    assert envelopes[0].reported_values == ["91%"]


def test_retains_explicit_negative_result_without_inventing_effect_size():
    envelopes = extract_validity_envelopes(
        "We find no improvement in accuracy on CIFAR-10."
    )

    assert len(envelopes) == 1
    assert envelopes[0].direction == "negative"
    assert envelopes[0].metrics == ["accuracy"]
    assert envelopes[0].evaluation_contexts == ["CIFAR-10"]
    assert envelopes[0].effect_sizes == []


def test_paper_features_json_without_validity_envelopes_remains_compatible():
    features = PaperFeatures.model_validate(
        {"one_sentence_summary": "Older stored feature record.", "confidence": 0.5}
    )

    assert features.validity_envelopes == []
