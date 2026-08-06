"""Typed data models shared by the three pipeline components."""

from __future__ import annotations

import ipaddress
from datetime import datetime
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Reject unexpected fields so component contracts fail loudly."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PaperRecord(StrictModel):
    """Metadata for one logical arXiv paper.

    ``arxiv_id`` omits the version suffix, so revisions such as v1 and v2 map
    to the same database row. ``versioned_id`` preserves the exact version.
    """

    arxiv_id: str
    versioned_id: str
    version: int = Field(ge=1)
    title: str
    abstract: str
    authors: list[str]
    affiliations: dict[str, list[str]] = Field(default_factory=dict)
    categories: list[str]
    primary_category: str
    published_at: datetime
    updated_at: datetime
    abs_url: str
    pdf_url: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None


SourceKind = Literal["research_blog", "research_report", "social_media"]


class FeedSourceConfig(StrictModel):
    """Configuration for one bounded, public RSS or Atom source."""

    source_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str = Field(min_length=1)
    source_kind: SourceKind
    feed_url: str = Field(min_length=1)
    homepage_url: str | None = None
    enabled: bool = True
    max_items: int = Field(default=1000, ge=1, le=2000)
    default_topics: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("default_topics")
    @classmethod
    def normalize_default_topics(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values]
        if any(not value for value in cleaned):
            raise ValueError("default topics must not be blank")
        return list(dict.fromkeys(cleaned))

    @field_validator("feed_url", "homepage_url")
    @classmethod
    def require_public_https_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("source URLs must use HTTPS and include a host")
        if parsed.username or parsed.password:
            raise ValueError("source URLs must not contain credentials")
        hostname = parsed.hostname.casefold().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("source URLs must use a public host")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            if not address.is_global:
                raise ValueError("source URLs must use a public host")
        return value


class SourceItem(StrictModel):
    """Normalized item from a public research or social feed."""

    item_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_kind: SourceKind
    external_id: str = Field(min_length=1)
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_topics: list[str] = Field(default_factory=list)
    summary: str | None = None
    content_text: str = ""
    canonical_url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("authors", "tags", "source_topics")
    @classmethod
    def normalize_string_lists(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(value.split())
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned

    @field_validator("canonical_url")
    @classmethod
    def require_https_canonical_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("canonical_url must use HTTPS and include a host")
        if parsed.username or parsed.password:
            raise ValueError("canonical_url must not contain credentials")
        return value

    @field_validator("published_at", "updated_at", "retrieved_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("source timestamps must include a timezone")
        return value


FeatureField = Literal[
    "research_tasks",
    "methods",
    "datasets",
    "metrics",
    "domains",
    "contributions",
    "limitations",
]


class Evidence(StrictModel):
    """A short, inspectable reason for one extracted feature value."""

    field: FeatureField
    value: str = Field(description="The exact extracted item supported by this evidence")
    statement: str = Field(
        description="A short paraphrase or quotation, about 20 words at most",
        max_length=240,
    )
    page: int | None = Field(default=None, ge=1)


ValidityClaimType = Literal[
    "comparative",
    "causal_language",
    "associational",
    "predictive",
    "descriptive",
]
ValidityDirection = Literal["positive", "negative", "mixed", "unclear"]


class ValidityEvidence(StrictModel):
    """Verbatim support for one abstract- or full-text-level claim."""

    source: Literal["abstract", "full_text"]
    sentence_index: int = Field(ge=0)
    statement: str = Field(min_length=1, max_length=1000)


class ValidityEnvelope(StrictModel):
    """Conditions and numeric support that bound one scientific claim."""

    claim: str = Field(min_length=1, max_length=1000)
    claim_type: ValidityClaimType
    direction: ValidityDirection
    comparators: list[str] = Field(default_factory=list)
    evaluation_contexts: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    reported_values: list[str] = Field(default_factory=list)
    effect_sizes: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    paper_level_boundaries: list[str] = Field(default_factory=list)
    evidence: ValidityEvidence
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "comparators",
        "evaluation_contexts",
        "metrics",
        "reported_values",
        "effect_sizes",
        "uncertainty",
        "conditions",
        "paper_level_boundaries",
    )
    @classmethod
    def remove_duplicate_strings(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(value.split()).strip(" ,;:")
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned


class PaperFeatures(StrictModel):
    """A deliberately small ontology for a first knowledge graph."""

    one_sentence_summary: str = Field(min_length=1)
    research_tasks: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    code_urls: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    validity_envelopes: list[ValidityEnvelope] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator(
        "research_tasks",
        "methods",
        "datasets",
        "metrics",
        "domains",
        "contributions",
        "limitations",
        "code_urls",
        "keywords",
    )
    @classmethod
    def remove_empty_and_duplicate_items(cls, values: list[str]) -> list[str]:
        """Keep first-seen order while removing blank and exact duplicate items."""

        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = " ".join(value.split())
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned


class StoredFeatureRecord(StrictModel):
    arxiv_id: str
    source_versioned_id: str
    extractor: str
    extractor_version: str
    prompt_version: str | None = None
    extracted_at: datetime
    features: PaperFeatures


class PaperValidityRecord(StrictModel):
    """Compact batch artifact joined back to source metadata by arXiv ID."""

    arxiv_id: str = Field(min_length=1)
    source_versioned_id: str = Field(min_length=1)
    source_scope: Literal["abstract", "full_text"]
    extractor: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    status: Literal["extracted", "no_supported_claim"]
    validity_envelopes: list[ValidityEnvelope] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_matches_envelopes(self) -> Self:
        expected = "extracted" if self.validity_envelopes else "no_supported_claim"
        if self.status != expected:
            raise ValueError(f"status must be {expected} for supplied envelopes")
        return self
