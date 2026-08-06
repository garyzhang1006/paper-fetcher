"""Deterministic hub, cluster, and trend analysis for graph snapshots.

Trend scoring is based on per-source unique-document shares, never raw feed
volume. For each equal window, each active source contributes
``topic_documents / all_documents`` and the window share is their mean.
Smoothed growth is ``log2((recent_share + 0.01) / (baseline_share + 0.01))``.
Emerging score is ``max(0, growth) * log1p(recent_count)`` and requires at
least three recent documents. Static snapshots use their latest publication
date as ``as_of``.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from datetime import UTC, date, datetime, timedelta
from typing import Literal, Self

from pydantic import Field, model_validator

from .graph import (
    CONCEPT_NODE_TYPES,
    DOCUMENT_NODE_TYPES,
    GraphNode,
    GraphSnapshot,
)
from .models import StrictModel

PAGERANK_DAMPING = 0.85
PAGERANK_TOLERANCE = 1e-8
PAGERANK_MAX_ITERATIONS = 100
CLUSTER_RELATED_THRESHOLD = 0.35
TREND_SMOOTHING = 0.01
TREND_MINIMUM_RECENT_DOCUMENTS = 3
TREND_MAXIMUM_WINDOW_DAYS = 7
TREND_MINIMUM_WINDOW_DAYS = 3
TREND_FORMULA = (
    "source_share=topic_unique_documents/source_unique_documents; "
    "window_share=mean(active_source_shares); "
    "growth=log2((recent_share+0.01)/(baseline_share+0.01)); "
    "emerging_score=max(0,growth)*log1p(recent_count); "
    "minimum_recent_count=3"
)


class HubMetric(StrictModel):
    node_id: str
    node_type: str
    name: str
    degree: int = Field(ge=0)
    weighted_degree: float = Field(ge=0.0)
    pagerank: float = Field(ge=0.0)


class ClusterResult(StrictModel):
    cluster_id: str
    label: str
    member_ids: list[str]
    concepts: list[dict[str, str | float]]


class TrendItem(StrictModel):
    topic_id: str
    topic: str
    recent_count: int = Field(ge=0)
    baseline_count: int = Field(ge=0)
    recent_share: float = Field(ge=0.0, le=1.0)
    baseline_share: float = Field(ge=0.0, le=1.0)
    growth: float
    emerging_score: float = Field(ge=0.0)


class TrendReport(StrictModel):
    status: Literal["ok", "insufficient_history", "no_documents"]
    as_of: date | None
    coverage_days: int = Field(ge=0)
    window_days: int = Field(ge=0)
    recent_start: date | None = None
    recent_end: date | None = None
    baseline_start: date | None = None
    baseline_end: date | None = None
    formula: str = TREND_FORMULA
    hot: list[TrendItem] = Field(default_factory=list)
    emerging: list[TrendItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_dates_for_ok_report(self) -> Self:
        if self.status == "ok" and None in {
            self.as_of,
            self.recent_start,
            self.recent_end,
            self.baseline_start,
            self.baseline_end,
        }:
            raise ValueError("successful trend report requires complete window dates")
        return self


def _undirected_adjacency(
    snapshot: GraphSnapshot,
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = {
        node.node_id: {} for node in snapshot.nodes
    }
    for edge in snapshot.edges:
        if edge.weight <= 0.0:
            continue
        adjacency[edge.source_id][edge.target_id] = (
            adjacency[edge.source_id].get(edge.target_id, 0.0) + edge.weight
        )
        adjacency[edge.target_id][edge.source_id] = (
            adjacency[edge.target_id].get(edge.source_id, 0.0) + edge.weight
        )
    return adjacency


def pagerank(
    snapshot: GraphSnapshot,
    *,
    damping: float = PAGERANK_DAMPING,
    tolerance: float = PAGERANK_TOLERANCE,
    max_iterations: int = PAGERANK_MAX_ITERATIONS,
) -> dict[str, float]:
    """Return deterministic weighted PageRank over an undirected snapshot."""

    if not 0.0 < damping < 1.0:
        raise ValueError("damping must be in (0, 1)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    node_ids = sorted(node.node_id for node in snapshot.nodes)
    node_count = len(node_ids)
    if node_count == 0:
        return {}
    adjacency = _undirected_adjacency(snapshot)
    ranks = {node_id: 1.0 / node_count for node_id in node_ids}

    for _ in range(max_iterations):
        dangling = sum(ranks[node_id] for node_id in node_ids if not adjacency[node_id])
        base = (1.0 - damping) / node_count + damping * dangling / node_count
        next_ranks = {node_id: base for node_id in node_ids}
        for source_id in node_ids:
            neighbors = adjacency[source_id]
            total_weight = sum(neighbors.values())
            if total_weight <= 0.0:
                continue
            for target_id, weight in sorted(neighbors.items()):
                next_ranks[target_id] += (
                    damping * ranks[source_id] * weight / total_weight
                )
        delta = sum(abs(next_ranks[node_id] - ranks[node_id]) for node_id in node_ids)
        ranks = next_ranks
        if delta <= tolerance:
            break

    total = sum(ranks.values())
    return {node_id: ranks[node_id] / total for node_id in node_ids}


def compute_hubs(snapshot: GraphSnapshot) -> list[HubMetric]:
    """Rank all nodes by PageRank, then weighted and unweighted degree."""

    nodes = {node.node_id: node for node in snapshot.nodes}
    adjacency = _undirected_adjacency(snapshot)
    ranks = pagerank(snapshot)
    metrics = [
        HubMetric(
            node_id=node_id,
            node_type=nodes[node_id].node_type,
            name=nodes[node_id].name,
            degree=len(adjacency[node_id]),
            weighted_degree=sum(adjacency[node_id].values()),
            pagerank=ranks[node_id],
        )
        for node_id in nodes
    ]
    return sorted(
        metrics,
        key=lambda item: (
            -item.pagerank,
            -item.weighted_degree,
            -item.degree,
            item.node_id,
        ),
    )


def _document_nodes(snapshot: GraphSnapshot) -> dict[str, GraphNode]:
    return {
        node.node_id: node
        for node in snapshot.nodes
        if node.node_type in DOCUMENT_NODE_TYPES
    }


def _semantic_concepts_by_document(
    snapshot: GraphSnapshot,
) -> dict[str, set[str]]:
    concept_ids = {
        node.node_id
        for node in snapshot.nodes
        if node.node_type in CONCEPT_NODE_TYPES
    }
    output: dict[str, set[str]] = defaultdict(set)
    for edge in snapshot.edges:
        if edge.relation == "RELATED_TO":
            continue
        if edge.target_id in concept_ids:
            output[edge.source_id].add(edge.target_id)
    return output


def _cluster_concepts(
    *,
    members: list[str],
    all_document_count: int,
    concepts_by_document: dict[str, set[str]],
    nodes: dict[str, GraphNode],
    concept_document_frequency: dict[str, int],
    limit: int,
) -> list[dict[str, str | float]]:
    frequency: dict[str, int] = defaultdict(int)
    for member_id in members:
        for concept_id in concepts_by_document.get(member_id, set()):
            frequency[concept_id] += 1
    ranked: list[tuple[float, str]] = []
    for concept_id, member_frequency in frequency.items():
        document_frequency = concept_document_frequency[concept_id]
        idf = math.log((all_document_count + 1) / (document_frequency + 1)) + 1.0
        ranked.append((member_frequency * idf, concept_id))
    return [
        {
            "node_id": concept_id,
            "node_type": nodes[concept_id].node_type,
            "name": nodes[concept_id].name,
            "score": score,
        }
        for score, concept_id in sorted(
            ranked, key=lambda item: (-item[0], item[1])
        )[:limit]
    ]


def find_clusters(
    snapshot: GraphSnapshot,
    *,
    related_threshold: float = CLUSTER_RELATED_THRESHOLD,
    label_concept_limit: int = 3,
) -> list[ClusterResult]:
    """Find deterministic document components over strong related edges."""

    if not 0.0 <= related_threshold <= 1.0:
        raise ValueError("related_threshold must be in [0, 1]")
    if label_concept_limit < 1:
        raise ValueError("label_concept_limit must be at least 1")

    documents = _document_nodes(snapshot)
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in documents}
    for edge in snapshot.edges:
        if edge.relation != "RELATED_TO" or edge.weight < related_threshold:
            continue
        if edge.source_id in documents and edge.target_id in documents:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

    components: list[list[str]] = []
    unvisited = set(documents)
    while unvisited:
        start = min(unvisited)
        queue = deque([start])
        unvisited.remove(start)
        component: list[str] = []
        while queue:
            node_id = queue.popleft()
            component.append(node_id)
            for neighbor in sorted(adjacency[node_id]):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
        if len(component) >= 2:
            components.append(sorted(component))

    nodes = {node.node_id: node for node in snapshot.nodes}
    concepts_by_document = _semantic_concepts_by_document(snapshot)
    concept_document_frequency: dict[str, int] = defaultdict(int)
    for concept_ids in concepts_by_document.values():
        for concept_id in concept_ids:
            concept_document_frequency[concept_id] += 1

    results: list[ClusterResult] = []
    for members in sorted(components, key=lambda values: values[0]):
        concepts = _cluster_concepts(
            members=members,
            all_document_count=len(documents),
            concepts_by_document=concepts_by_document,
            nodes=nodes,
            concept_document_frequency=concept_document_frequency,
            limit=label_concept_limit,
        )
        label = ", ".join(str(item["name"]) for item in concepts) or "Unlabeled"
        digest = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:16]
        results.append(
            ClusterResult(
                cluster_id=f"cluster:{digest}",
                label=label,
                member_ids=members,
                concepts=concepts,
            )
        )
    return results


def _published_documents(
    snapshot: GraphSnapshot,
) -> list[tuple[str, str, date]]:
    output: list[tuple[str, str, date]] = []
    for node in snapshot.nodes:
        if node.node_type not in DOCUMENT_NODE_TYPES:
            continue
        source_id = node.properties.get("source_id")
        raw_published = node.properties.get("published_at")
        if not isinstance(source_id, str) or not isinstance(raw_published, str):
            continue
        try:
            published = datetime.fromisoformat(raw_published)
        except ValueError:
            continue
        if published.tzinfo is None or published.utcoffset() is None:
            continue
        output.append((node.node_id, source_id, published.astimezone(UTC).date()))
    return output


def _mean_source_share(
    *,
    documents: list[tuple[str, str, date]],
    topic_documents: set[str],
    start: date,
    end: date,
) -> tuple[float, int]:
    totals: dict[str, set[str]] = defaultdict(set)
    topic_totals: dict[str, set[str]] = defaultdict(set)
    for document_id, source_id, published in documents:
        if not start <= published <= end:
            continue
        totals[source_id].add(document_id)
        if document_id in topic_documents:
            topic_totals[source_id].add(document_id)
    if not totals:
        return 0.0, 0
    source_shares = [
        len(topic_totals[source_id]) / len(source_documents)
        for source_id, source_documents in sorted(totals.items())
    ]
    unique_count = len(
        set().union(*(topic_totals[source_id] for source_id in topic_totals))
    )
    return sum(source_shares) / len(source_shares), unique_count


def compute_trends(
    snapshot: GraphSnapshot,
    *,
    as_of: date | None = None,
    window_days: int | None = None,
) -> TrendReport:
    """Rank hot and emerging topics using source-normalized equal windows."""

    documents = _published_documents(snapshot)
    if not documents:
        return TrendReport(
            status="no_documents", as_of=as_of, coverage_days=0, window_days=0
        )
    latest_date = max(item[2] for item in documents)
    selected_as_of = as_of or latest_date
    eligible_dates = [item[2] for item in documents if item[2] <= selected_as_of]
    if not eligible_dates:
        return TrendReport(
            status="no_documents", as_of=selected_as_of, coverage_days=0, window_days=0
        )
    first_date = min(eligible_dates)
    coverage_days = (selected_as_of - first_date).days + 1

    if window_days is None:
        selected_window = min(TREND_MAXIMUM_WINDOW_DAYS, coverage_days // 2)
    else:
        if window_days < TREND_MINIMUM_WINDOW_DAYS:
            raise ValueError(
                f"window_days must be at least {TREND_MINIMUM_WINDOW_DAYS}"
            )
        selected_window = window_days
    if selected_window < TREND_MINIMUM_WINDOW_DAYS or coverage_days < 2 * selected_window:
        return TrendReport(
            status="insufficient_history",
            as_of=selected_as_of,
            coverage_days=coverage_days,
            window_days=max(selected_window, 0),
        )

    recent_end = selected_as_of
    recent_start = recent_end - timedelta(days=selected_window - 1)
    baseline_end = recent_start - timedelta(days=1)
    baseline_start = baseline_end - timedelta(days=selected_window - 1)

    nodes = {node.node_id: node for node in snapshot.nodes}
    topic_documents: dict[str, set[str]] = defaultdict(set)
    for edge in snapshot.edges:
        if edge.relation != "ABOUT_TOPIC":
            continue
        target = nodes.get(edge.target_id)
        if target is not None and target.node_type == "topic":
            topic_documents[edge.target_id].add(edge.source_id)

    items: list[TrendItem] = []
    for topic_id, document_ids in sorted(topic_documents.items()):
        recent_share, recent_count = _mean_source_share(
            documents=documents,
            topic_documents=document_ids,
            start=recent_start,
            end=recent_end,
        )
        baseline_share, baseline_count = _mean_source_share(
            documents=documents,
            topic_documents=document_ids,
            start=baseline_start,
            end=baseline_end,
        )
        growth = math.log2(
            (recent_share + TREND_SMOOTHING)
            / (baseline_share + TREND_SMOOTHING)
        )
        emerging_score = (
            max(0.0, growth) * math.log1p(recent_count)
            if recent_count >= TREND_MINIMUM_RECENT_DOCUMENTS
            else 0.0
        )
        items.append(
            TrendItem(
                topic_id=topic_id,
                topic=nodes[topic_id].name,
                recent_count=recent_count,
                baseline_count=baseline_count,
                recent_share=recent_share,
                baseline_share=baseline_share,
                growth=growth,
                emerging_score=emerging_score,
            )
        )

    hot = sorted(
        (item for item in items if item.recent_count > 0),
        key=lambda item: (-item.recent_share, -item.recent_count, item.topic_id),
    )
    emerging = sorted(
        (item for item in items if item.emerging_score > 0.0),
        key=lambda item: (-item.emerging_score, -item.recent_share, item.topic_id),
    )
    return TrendReport(
        status="ok",
        as_of=selected_as_of,
        coverage_days=coverage_days,
        window_days=selected_window,
        recent_start=recent_start,
        recent_end=recent_end,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        hot=hot,
        emerging=emerging,
    )
