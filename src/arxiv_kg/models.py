"""Typed data models shared by the three pipeline components."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
