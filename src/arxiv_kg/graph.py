"""Pure, deterministic construction of the Project 1 knowledge graph.

Semantic edges require extractor evidence. ArXiv categories and configured
feed topics are exceptions because they are explicit source metadata. Related
documents use IDF-weighted Jaccard overlap over supported concepts, suppress
concepts present in more than 10 percent of documents, require score >= 0.20,
and cap every document at 10 related neighbors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any, Iterable, Literal, Self

from pydantic import Field, field_validator, model_validator

from .ids import canonical_key, normalize_text_name, stable_node_id
from .models import PaperFeatures, StrictModel

DocumentNodeType = Literal["paper", "report", "blog_post", "social_post"]
ConceptNodeType = Literal[
    "topic",
    "method",
    "research_goal",
    "dataset",
    "metric",
    "category",
]
GraphNodeType = Literal[
    "paper",
    "report",
    "blog_post",
    "social_post",
    "topic",
    "method",
    "research_goal",
    "dataset",
    "metric",
    "category",
]
GraphRelation = Literal[
    "ABOUT_TOPIC",
    "PURSUES_GOAL",
    "USES_METHOD",
    "EVALUATES_ON",
    "REPORTS_METRIC",
    "IN_CATEGORY",
    "RELATED_TO",
]

DOCUMENT_NODE_TYPES = frozenset({"paper", "report", "blog_post", "social_post"})
CONCEPT_NODE_TYPES = frozenset(
    {"topic", "method", "research_goal", "dataset", "metric", "category"}
)
SEMANTIC_MAPPING: tuple[tuple[str, ConceptNodeType, GraphRelation], ...] = (
    ("domains", "topic", "ABOUT_TOPIC"),
    ("research_tasks", "research_goal", "PURSUES_GOAL"),
    ("methods", "method", "USES_METHOD"),
    ("datasets", "dataset", "EVALUATES_ON"),
    ("metrics", "metric", "REPORTS_METRIC"),
)
GRAPH_ALGORITHM_VERSION = "project-1-graph-v1"
DEFAULT_COMMON_CONCEPT_RATIO = 0.10
DEFAULT_RELATED_THRESHOLD = 0.20
DEFAULT_RELATED_LIMIT = 10


class GraphDocument(StrictModel):
    """One extracted document ready for graph placement."""

    document_id: str = Field(min_length=1)
    node_type: DocumentNodeType
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    title: str = ""
    url: str | None = None
    published_at: datetime
    categories: list[str] = Field(default_factory=list)
    source_topics: list[str] = Field(default_factory=list)
    features: PaperFeatures
    extractor: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    prompt_version: str | None = None

    @field_validator("categories", "source_topics")
    @classmethod
    def clean_categories(cls, values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = normalize_text_name(value)
            key = canonical_key(cleaned)
            if key and key not in seen:
                seen.add(key)
                output.append(cleaned)
        return output

    @model_validator(mode="after")
    def require_timezone(self) -> Self:
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("published_at must include a timezone")
        if not canonical_key(self.document_id):
            raise ValueError("document_id must contain letters or numbers")
        if self.node_type != "paper" and self.categories:
            raise ValueError("only paper documents may carry arXiv categories")
        return self


class GraphNode(StrictModel):
    node_id: str = Field(min_length=1)
    node_type: GraphNodeType
    name: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(StrictModel):
    source_id: str = Field(min_length=1)
    relation: GraphRelation
    target_id: str = Field(min_length=1)
    weight: float = Field(ge=0.0)
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_self_edge(self) -> Self:
        if self.source_id == self.target_id:
            raise ValueError("graph edges cannot be self edges")
        return self


class GraphSnapshot(StrictModel):
    algorithm_version: str = GRAPH_ALGORITHM_VERSION
    common_concept_ratio: float = DEFAULT_COMMON_CONCEPT_RATIO
    related_threshold: float = DEFAULT_RELATED_THRESHOLD
    related_limit: int = DEFAULT_RELATED_LIMIT
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    suppressed_concept_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references_and_uniqueness(self) -> Self:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph snapshot contains duplicate node IDs")
        known = set(node_ids)
        edge_keys = [
            (edge.source_id, edge.relation, edge.target_id) for edge in self.edges
        ]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("graph snapshot contains duplicate edges")
        missing = {
            endpoint
            for edge in self.edges
            for endpoint in (edge.source_id, edge.target_id)
            if endpoint not in known
        }
        if missing:
            raise ValueError(f"graph edges reference missing nodes: {sorted(missing)!r}")
        return self

    def document_node_id(
        self, document_id: str, node_type: DocumentNodeType | None = None
    ) -> str | None:
        """Resolve a source document ID without exposing ID hashing to callers."""

        for node in self.nodes:
            if node.node_type not in DOCUMENT_NODE_TYPES:
                continue
            if node_type is not None and node.node_type != node_type:
                continue
            if node.properties.get("document_id") == document_id:
                return node.node_id
        return None


def _node_id(node_type: GraphNodeType, source_name: str) -> tuple[str, str]:
    canonical_name = canonical_key(source_name)
    if not canonical_name:
        raise ValueError(f"{node_type} name must contain letters or numbers")
    return stable_node_id(node_type, canonical_name), canonical_name


def _display_name(current: str, candidate: str) -> str:
    return min((current, candidate), key=lambda value: (value.casefold(), value))


def _put_node(nodes: dict[str, GraphNode], candidate: GraphNode) -> None:
    existing = nodes.get(candidate.node_id)
    if existing is None:
        nodes[candidate.node_id] = candidate
        return
    if (
        existing.node_type != candidate.node_type
        or existing.canonical_name != candidate.canonical_name
    ):
        raise RuntimeError(
            "stable node ID collision: "
            f"{candidate.node_id!r} maps to both "
            f"{existing.node_type}/{existing.canonical_name!r} and "
            f"{candidate.node_type}/{candidate.canonical_name!r}"
        )
    if candidate.name != existing.name:
        nodes[candidate.node_id] = existing.model_copy(
            update={"name": _display_name(existing.name, candidate.name)}
        )


def _supported_evidence(
    document: GraphDocument, field: str, value: str
) -> list[dict[str, Any]]:
    key = canonical_key(value)
    matches = [
        item
        for item in document.features.evidence
        if item.field == field and canonical_key(item.value) == key
    ]
    return [
        {"statement": item.statement, "page": item.page}
        for item in sorted(matches, key=lambda item: (item.page or 0, item.statement))
    ]


def _semantic_edge_properties(
    document: GraphDocument, evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "basis": "extractor_evidence",
        "confidence": document.features.confidence,
        "evidence": evidence,
        "extractor": document.extractor,
        "extractor_version": document.extractor_version,
        "prompt_version": document.prompt_version,
        "source_version": document.source_version,
    }


def _related_edges(
    *,
    document_ids: list[str],
    concept_ids_by_document: dict[str, set[str]],
    concept_postings: dict[str, set[str]],
    nodes: dict[str, GraphNode],
    common_concept_ratio: float,
    related_threshold: float,
    related_limit: int,
) -> tuple[list[GraphEdge], list[str]]:
    document_count = len(document_ids)
    if document_count < 2:
        return [], []

    suppressed = {
        concept_id
        for concept_id, postings in concept_postings.items()
        if len(postings) / document_count > common_concept_ratio
    }
    usable_postings = {
        concept_id: postings
        for concept_id, postings in concept_postings.items()
        if concept_id not in suppressed
    }
    idf = {
        concept_id: math.log((document_count + 1) / (len(postings) + 1)) + 1.0
        for concept_id, postings in usable_postings.items()
    }

    candidates: set[tuple[str, str]] = set()
    for postings in usable_postings.values():
        candidates.update(combinations(sorted(postings), 2))

    ranked: list[tuple[float, str, str, set[str]]] = []
    usable_by_document = {
        document_id: concepts - suppressed
        for document_id, concepts in concept_ids_by_document.items()
    }
    for source_id, target_id in sorted(candidates):
        source_concepts = usable_by_document[source_id]
        target_concepts = usable_by_document[target_id]
        shared = source_concepts & target_concepts
        union = source_concepts | target_concepts
        if not shared or not union:
            continue
        numerator = sum(idf[concept_id] for concept_id in shared)
        denominator = sum(idf[concept_id] for concept_id in union)
        score = numerator / denominator
        if score >= related_threshold:
            ranked.append((score, source_id, target_id, shared))

    degrees: dict[str, int] = defaultdict(int)
    output: list[GraphEdge] = []
    for score, source_id, target_id, shared in sorted(
        ranked, key=lambda item: (-item[0], item[1], item[2])
    ):
        if degrees[source_id] >= related_limit or degrees[target_id] >= related_limit:
            continue
        shared_nodes = [nodes[node_id] for node_id in shared]
        shared_values = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "name": node.name,
                "idf": idf[node.node_id],
            }
            for node in sorted(shared_nodes, key=lambda node: node.node_id)
        ]
        output.append(
            GraphEdge(
                source_id=source_id,
                relation="RELATED_TO",
                target_id=target_id,
                weight=score,
                properties={
                    "score": score,
                    "shared_concepts": shared_values,
                },
            )
        )
        degrees[source_id] += 1
        degrees[target_id] += 1
    return output, sorted(suppressed)


def build_graph(
    documents: Iterable[GraphDocument],
    *,
    common_concept_ratio: float = DEFAULT_COMMON_CONCEPT_RATIO,
    related_threshold: float = DEFAULT_RELATED_THRESHOLD,
    related_limit: int = DEFAULT_RELATED_LIMIT,
) -> GraphSnapshot:
    """Build a typed, evidence-backed graph without persistence side effects."""

    if not 0.0 < common_concept_ratio <= 1.0:
        raise ValueError("common_concept_ratio must be in (0, 1]")
    if not 0.0 <= related_threshold <= 1.0:
        raise ValueError("related_threshold must be in [0, 1]")
    if related_limit < 1:
        raise ValueError("related_limit must be at least 1")

    ordered_documents = sorted(
        documents, key=lambda item: (item.node_type, canonical_key(item.document_id))
    )
    seen_documents: set[tuple[str, str]] = set()
    nodes: dict[str, GraphNode] = {}
    semantic_edges: list[GraphEdge] = []
    concept_postings: dict[str, set[str]] = defaultdict(set)
    concepts_by_document: dict[str, set[str]] = defaultdict(set)
    document_node_ids: list[str] = []

    for document in ordered_documents:
        document_key = (document.node_type, canonical_key(document.document_id))
        if document_key in seen_documents:
            raise ValueError(
                f"duplicate graph document: {document.node_type}/{document.document_id}"
            )
        seen_documents.add(document_key)

        node_id, canonical_name = _node_id(document.node_type, document.document_id)
        document_node_ids.append(node_id)
        _put_node(
            nodes,
            GraphNode(
                node_id=node_id,
                node_type=document.node_type,
                name=normalize_text_name(document.title)
                or document.features.one_sentence_summary,
                canonical_name=canonical_name,
                properties={
                    "document_id": document.document_id,
                    "source_id": document.source_id,
                    "source_version": document.source_version,
                    "url": document.url,
                    "published_at": document.published_at.isoformat(),
                    "summary": document.features.one_sentence_summary,
                    "summary_provenance": {
                        "extractor": document.extractor,
                        "extractor_version": document.extractor_version,
                        "prompt_version": document.prompt_version,
                        "source_version": document.source_version,
                    },
                },
            ),
        )

        for field, concept_type, relation in SEMANTIC_MAPPING:
            for value in getattr(document.features, field):
                evidence = _supported_evidence(document, field, value)
                if not evidence:
                    continue
                concept_id, concept_key = _node_id(concept_type, value)
                _put_node(
                    nodes,
                    GraphNode(
                        node_id=concept_id,
                        node_type=concept_type,
                        name=normalize_text_name(value),
                        canonical_name=concept_key,
                    ),
                )
                semantic_edges.append(
                    GraphEdge(
                        source_id=node_id,
                        relation=relation,
                        target_id=concept_id,
                        weight=1.0,
                        properties=_semantic_edge_properties(document, evidence),
                    )
                )
                concept_postings[concept_id].add(node_id)
                concepts_by_document[node_id].add(concept_id)

        for category in document.categories:
            category_id, category_key = _node_id("category", category)
            _put_node(
                nodes,
                GraphNode(
                    node_id=category_id,
                    node_type="category",
                    name=normalize_text_name(category),
                    canonical_name=category_key,
                    properties={"code": normalize_text_name(category)},
                ),
            )
            semantic_edges.append(
                GraphEdge(
                    source_id=node_id,
                    relation="IN_CATEGORY",
                    target_id=category_id,
                    weight=1.0,
                    properties={
                        "basis": "source_metadata",
                        "source_version": document.source_version,
                    },
                )
            )
            concept_postings[category_id].add(node_id)
            concepts_by_document[node_id].add(category_id)

        for topic in document.source_topics:
            topic_id, topic_key = _node_id("topic", topic)
            _put_node(
                nodes,
                GraphNode(
                    node_id=topic_id,
                    node_type="topic",
                    name=normalize_text_name(topic),
                    canonical_name=topic_key,
                ),
            )
            semantic_edges.append(
                GraphEdge(
                    source_id=node_id,
                    relation="ABOUT_TOPIC",
                    target_id=topic_id,
                    weight=1.0,
                    properties={
                        "basis": "source_metadata",
                        "metadata_field": "default_topics",
                        "source_version": document.source_version,
                    },
                )
            )
            concept_postings[topic_id].add(node_id)
            concepts_by_document[node_id].add(topic_id)

    for concept_id, postings in concept_postings.items():
        node = nodes[concept_id]
        nodes[concept_id] = node.model_copy(
            update={"properties": {**node.properties, "document_count": len(postings)}}
        )

    related_edges, suppressed = _related_edges(
        document_ids=document_node_ids,
        concept_ids_by_document=concepts_by_document,
        concept_postings=concept_postings,
        nodes=nodes,
        common_concept_ratio=common_concept_ratio,
        related_threshold=related_threshold,
        related_limit=related_limit,
    )
    edges = sorted(
        [*semantic_edges, *related_edges],
        key=lambda edge: (edge.source_id, edge.relation, edge.target_id),
    )
    return GraphSnapshot(
        common_concept_ratio=common_concept_ratio,
        related_threshold=related_threshold,
        related_limit=related_limit,
        nodes=sorted(nodes.values(), key=lambda node: node.node_id),
        edges=edges,
        suppressed_concept_ids=suppressed,
    )
