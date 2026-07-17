"""Evidence-first experimental validity-envelope extraction."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import ValidityEnvelope, ValidityEvidence

VALIDITY_EXTRACTOR_NAME = "abstract-validity-rules"
VALIDITY_EXTRACTOR_VERSION = "1.0"
MAX_ENVELOPES_PER_PAPER = 5

SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s+|(?=[A-Z]))")
CLAUSE_RE = re.compile(
    r"(?:[;\n]+|\s*,?\s+\b(?:but|however|whereas)\b\s*)",
    re.IGNORECASE,
)
CLAIM_RE = re.compile(
    r"\b(?:"
    r"we\s+(?:find|show|demonstrate|observe|report|establish|confirm|achieve)"
    r"|our\s+(?:results?|experiments?|analysis|evaluation)\s+"
    r"(?:show|shows|indicate|indicates|suggest|suggests|demonstrate|demonstrates|reveal|reveals)"
    r"|(?:the\s+)?results?\s+(?:show|shows|indicate|indicates|suggest|suggests|demonstrate|demonstrates|reveal|reveals)"
    r"|(?:experiments?|evaluation)\s+(?:show|shows|demonstrate|demonstrates|confirm|confirms)"
    r"|our\s+(?:method|model|approach|framework|system)\s+"
    r"(?:achieves?|attains?|outperforms?|improves?|reduces?|increases?|yields?)"
    r")\b",
    re.IGNORECASE,
)
RELATED_WORK_RE = re.compile(
    r"\b(?:prior work|previous work|related work|earlier work|other studies)\b",
    re.IGNORECASE,
)
BOUNDARY_RE = re.compile(
    r"\b(?:"
    r"limitations?|limited by|restricted to|fails? to|struggl(?:e|es|ed) to|"
    r"cannot|does not generalize|do not generalize|future work|"
    r"remains? (?:unclear|unknown|challenging)"
    r")\b",
    re.IGNORECASE,
)
NON_CURRENT_BOUNDARY_RE = re.compile(
    r"\b(?:to address (?:this|that|the|these) limitations?|"
    r"overcom(?:e|es|ing) (?:this|that|the|these) limitations?|"
    r"(?:existing|prior|previous) (?:methods?|approaches?|work))\b",
    re.IGNORECASE,
)
COMPARISON_SIGNAL_RE = re.compile(
    r"\b(?:compared (?:with|to)|relative to|versus|vs\.?|outperform(?:s|ed)?|"
    r"underperform(?:s|ed)?|improv(?:e|es|ed|ement)|higher|lower|better|worse)\b",
    re.IGNORECASE,
)
CAUSAL_RE = re.compile(
    r"\b(?:causes?|caused|leads? to|resulted? in|drives?|due to|because of)\b",
    re.IGNORECASE,
)
ASSOCIATION_RE = re.compile(
    r"\b(?:associated with|correlat(?:e|es|ed|ion)|linked to|relationship between)\b",
    re.IGNORECASE,
)
PREDICTIVE_RE = re.compile(
    r"\b(?:predict(?:s|ed|ion)?|forecast(?:s|ed|ing)?|prognostic)\b",
    re.IGNORECASE,
)
NEGATIVE_RE = re.compile(
    r"\b(?:no improvement|does not|do not|did not|fails? to|underperform(?:s|ed)?|worse)\b",
    re.IGNORECASE,
)
NO_IMPROVEMENT_RE = re.compile(r"\bno improvement\b", re.IGNORECASE)
POSITIVE_RE = re.compile(
    r"\b(?:outperform(?:s|ed)?|improv(?:e|es|ed|ement)|better|"
    r"higher (?:accuracy|score|performance)|lower (?:error|loss|latency)|"
    r"reduc(?:e|es|ed|tion)|increas(?:e|es|ed))\b",
    re.IGNORECASE,
)

COMPARATOR_PATTERNS = (
    re.compile(
        r"\b(?:compared (?:with|to)|relative to|versus|vs\.?)\s+"
        r"(?P<value>[^,;:.()]{2,100}?)(?="
        r"\s+(?:and|but)\s+(?:can\s+)?(?:outperform|underperform|improve)\b"
        r"|\s+(?:while|whereas|but)\b"
        r"|\s+and\s+(?:achieving|requiring|reducing|remaining|using)\b"
        r"|\s+(?:on|in|under|across|with|using)\b|[,;:.()]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\boutperform(?:s|ed)?\s+(?P<value>[^,;:.]{2,80}?)(?="
        r"\s+(?:on|in|under|by|with|while|whereas|using|across)\b"
        r"|\s+and\s+(?:achieving|requiring|reducing|remaining|using)\b|[,.]|$)",
        re.IGNORECASE,
    ),
)
NOT_ONLY_COMPARATOR_RE = re.compile(
    r"\boutperform(?:s|ed)?\s+not only\s+(?P<first>[^,;:.]+?)\s+"
    r"but (?:also|even)\s+(?P<second>[^,;:.]+?)(?=\s+(?:on|in|under|by|with)\b|[,.]|$)",
    re.IGNORECASE,
)
METRIC_RE = re.compile(
    r"\b(?:accuracy|F1(?:[- ]score)?|precision|recall|AUROC|ROC-AUC|AUC|BLEU|ROUGE|"
    r"perplexity|RMSE|MAE|MSE|error rate|success rate|mortality|hazard ratio|"
    r"odds ratio|correlation|throughput|latency)\b",
    re.IGNORECASE,
)
REPORTED_VALUE_PATTERNS = (
    re.compile(
        r"(?P<value>\{?[+-]?\d+(?:\.\d+)?\\?%\}?|"
        r"[+-]?\d+(?:\.\d+)?\s*(?:percent(?:age)?(?: points?)?|pp\b|-?fold\b|×))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<value>(?<![\w.])\d+(?:\.\d+)?"
        r"(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*[x×])\b",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<value>\bfactor of\s+\d+(?:\.\d+)?)\b", re.IGNORECASE),
    re.compile(
        r"(?P<value>\b\d+(?:\.\d+)?\s*(?:dB|mm|cm|km|ms|ns|GHz|MHz|kHz|MB|GB|TB))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:accuracy|F1(?:[- ]score)?|AUROC|ROC-AUC|AUC|BLEU|ROUGE|"
        r"perplexity|RMSE|MAE|MSE)\s*[:=]?\s*(?P<value>\d+(?:\.\d+)?)\b",
        re.IGNORECASE,
    ),
)
EFFECT_RE = re.compile(
    r"(?:"
    r"(?<!\w)[+-]?\d+(?:\.\d+)?\s*(?:percentage points?|pp\b)"
    r"|\b(?:by|(?:improvement|increase|decrease|reduction) of)\s+"
    r"[+-]?\d+(?:\.\d+)?\s*(?:%|percent|fold)"
    r"(?:\s+to\s+[+-]?\d+(?:\.\d+)?\s*(?:%|percent|fold))?"
    r"|\b\d+(?:\.\d+)?-fold\s+(?:speedup|improvement|increase|reduction)"
    r"|\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*[x×]\s+faster"
    r"|\bfactor of\s+\d+(?:\.\d+)?"
    r"|\{?\d+(?:\.\d+)?\\?%\}?\s+(?:gain|improvement|increase|reduction)"
    r"|\b\d+(?:\.\d+)?\s+percent(?:\s+to\s+\d+(?:\.\d+)?\s+percent)?\s+"
    r"(?:improvements?|increases?|decreases?|reductions?)"
    r")",
    re.IGNORECASE,
)
UNCERTAINTY_PATTERNS = (
    re.compile(
        r"\bp\s*[<=>]\s*(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|\.\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+(?:\.\d+)?%\s*(?:CI|confidence interval)\b"
        r"(?:\s*(?:of|[:=])?\s*\[?\(?[+-]?\d+(?:\.\d+)?\s*"
        r"(?:to|[-–,])\s*[+-]?\d+(?:\.\d+)?\)?\]?)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\w)[+-]?\d+(?:\.\d+)?\s*(?:%|percent)?\s*\$?\s*"
        r"(?:±|\\pm)\s*\$?\s*[+-]?\d+(?:\.\d+)?\s*(?:%|percent)?"
    ),
)
CONDITION_RE = re.compile(
    r"\b(?P<prefix>under|across|when|within)\s+"
    r"(?P<value>[^;.()]{3,180}?)(?="
    r"\s*\(|[;.]|,\s+(?=(?:and\s+(?:it|they|this|that|we|our)|"
    r"we|our|the|this|these|results?|experiments?|while|whereas|but)\b)"
    r"|\s+and\s+(?=(?:we|our|(?:the\s+)?results?|experiments?)\b)|$)",
    re.IGNORECASE,
)
CONTEXT_PATTERNS = (
    re.compile(
        r"\b(?:on|using)\s+(?:the\s+)?"
        r"(?P<value>[A-Z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*"
        r"(?:\s+[A-Z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*){0,3})"
        r"(?=\s+(?:dataset|benchmark|test set|cohort)\b|\s+(?:under|with|using)\b|[,.;()]|$)"
    ),
    re.compile(
        r"\b(?P<value>\d[\d,]*\s+(?:patients?|participants?|subjects?|samples?|cases?))\b",
        re.IGNORECASE,
    ),
)


def _deduplicate(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = " ".join(value.split()).strip(" ,;:")
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def _pattern_values(patterns: tuple[re.Pattern[str], ...], text: str) -> list[str]:
    return _deduplicate(
        [match.group("value") for pattern in patterns for match in pattern.finditer(text)]
    )


def _comparators(text: str) -> list[str]:
    not_only = NOT_ONLY_COMPARATOR_RE.search(text)
    if not_only:
        return _deduplicate([not_only.group("first"), not_only.group("second")])
    return _pattern_values(COMPARATOR_PATTERNS, text)


def _claim_clauses(sentence: str) -> list[str]:
    if re.search(r"\bnot only\b.+\bbut (?:also|even)\b", sentence, re.IGNORECASE):
        return [sentence]
    return CLAUSE_RE.split(sentence)


def _claim_type(text: str) -> str:
    if CAUSAL_RE.search(text):
        return "causal_language"
    if ASSOCIATION_RE.search(text):
        return "associational"
    if PREDICTIVE_RE.search(text):
        return "predictive"
    if COMPARISON_SIGNAL_RE.search(text):
        return "comparative"
    return "descriptive"


def _direction(text: str) -> str:
    if NO_IMPROVEMENT_RE.search(text):
        return "negative"
    negative = NEGATIVE_RE.search(text) is not None
    positive = POSITIVE_RE.search(text) is not None
    if negative and positive:
        return "mixed"
    if negative:
        return "negative"
    if positive:
        return "positive"
    return "unclear"


def _boundary_statements(abstract: str) -> list[str]:
    return _deduplicate(
        [
            sentence.strip()
            for sentence in SENTENCE_RE.split(" ".join(abstract.split()))
            if BOUNDARY_RE.search(sentence) and not NON_CURRENT_BOUNDARY_RE.search(sentence)
        ]
    )[:3]


def extract_validity_envelopes(abstract: str) -> list[ValidityEnvelope]:
    """Extract only claims with explicit support in an abstract."""

    clean = " ".join(abstract.split())
    if not clean:
        return []
    boundaries = _boundary_statements(clean)
    envelopes: list[ValidityEnvelope] = []
    seen_claims: set[str] = set()
    for sentence_index, sentence in enumerate(SENTENCE_RE.split(clean)):
        for clause in _claim_clauses(sentence):
            claim = clause.strip()
            if not claim or RELATED_WORK_RE.search(claim) or not CLAIM_RE.search(claim):
                continue
            key = claim.casefold()
            if key in seen_claims:
                continue
            seen_claims.add(key)

            comparators = _comparators(claim)
            contexts = _pattern_values(CONTEXT_PATTERNS, claim)
            metrics = _deduplicate(match.group(0) for match in METRIC_RE.finditer(claim))
            reported_values = _pattern_values(REPORTED_VALUE_PATTERNS, claim)
            effect_sizes = _deduplicate(match.group(0) for match in EFFECT_RE.finditer(claim))
            uncertainty = _deduplicate(
                match.group(0)
                for pattern in UNCERTAINTY_PATTERNS
                for match in pattern.finditer(claim)
            )
            conditions = _deduplicate(
                f"{match.group('prefix')} {match.group('value')}"
                for match in CONDITION_RE.finditer(claim)
            )
            supported_details = sum(
                bool(values)
                for values in (
                    comparators,
                    contexts,
                    metrics,
                    reported_values or effect_sizes,
                    uncertainty,
                    conditions,
                )
            )
            confidence = min(0.55 + supported_details * 0.05, 0.9)
            envelopes.append(
                ValidityEnvelope(
                    claim=claim,
                    claim_type=_claim_type(claim),
                    direction=_direction(claim),
                    comparators=comparators,
                    evaluation_contexts=contexts,
                    metrics=metrics,
                    reported_values=reported_values,
                    effect_sizes=effect_sizes,
                    uncertainty=uncertainty,
                    conditions=conditions,
                    paper_level_boundaries=boundaries,
                    evidence=ValidityEvidence(
                        source="abstract",
                        sentence_index=sentence_index,
                        statement=claim,
                    ),
                    confidence=confidence,
                )
            )
            if len(envelopes) >= MAX_ENVELOPES_PER_PAPER:
                return envelopes
    return envelopes
