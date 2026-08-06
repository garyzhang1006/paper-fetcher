import math
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from arxiv_kg.graph import GraphEdge, GraphNode, GraphSnapshot
from arxiv_kg.graph_analytics import (
    TREND_FORMULA,
    compute_hubs,
    compute_trends,
    find_clusters,
    pagerank,
)


def node(
    node_id: str,
    node_type: str,
    *,
    name: str | None = None,
    properties: dict | None = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        name=name or node_id,
        canonical_name=node_id.replace(":", " "),
        properties=properties or {},
    )


def test_weighted_pagerank_and_hub_metrics_are_normalized_and_deterministic():
    snapshot = GraphSnapshot(
        nodes=[node("paper:a", "paper"), node("topic:x", "topic")],
        edges=[
            GraphEdge(
                source_id="paper:a",
                relation="ABOUT_TOPIC",
                target_id="topic:x",
                weight=1.0,
            )
        ],
    )

    ranks = pagerank(snapshot)
    hubs = compute_hubs(snapshot)

    assert ranks == {"paper:a": pytest.approx(0.5), "topic:x": pytest.approx(0.5)}
    assert sum(ranks.values()) == pytest.approx(1.0)
    assert [item.node_id for item in hubs] == ["paper:a", "topic:x"]
    assert all(item.degree == 1 for item in hubs)
    assert all(item.weighted_degree == pytest.approx(1.0) for item in hubs)


def test_pagerank_redistributes_isolates_and_validates_parameters():
    snapshot = GraphSnapshot(
        nodes=[node("paper:a", "paper"), node("paper:b", "paper")], edges=[]
    )
    assert pagerank(snapshot) == {
        "paper:a": pytest.approx(0.5),
        "paper:b": pytest.approx(0.5),
    }
    with pytest.raises(ValueError, match="damping"):
        pagerank(snapshot, damping=1.0)


def test_clusters_use_only_strong_related_edges_and_label_supported_concepts():
    snapshot = GraphSnapshot(
        nodes=[
            node("paper:a", "paper"),
            node("paper:b", "paper"),
            node("paper:c", "paper"),
            node("topic:vision", "topic", name="Computer Vision"),
            node("method:diffusion", "method", name="Diffusion Model"),
        ],
        edges=[
            GraphEdge(
                source_id="paper:a",
                relation="ABOUT_TOPIC",
                target_id="topic:vision",
                weight=1.0,
            ),
            GraphEdge(
                source_id="paper:b",
                relation="ABOUT_TOPIC",
                target_id="topic:vision",
                weight=1.0,
            ),
            GraphEdge(
                source_id="paper:a",
                relation="USES_METHOD",
                target_id="method:diffusion",
                weight=1.0,
            ),
            GraphEdge(
                source_id="paper:a",
                relation="RELATED_TO",
                target_id="paper:b",
                weight=0.40,
            ),
            GraphEdge(
                source_id="paper:b",
                relation="RELATED_TO",
                target_id="paper:c",
                weight=0.34,
            ),
        ],
    )

    clusters = find_clusters(snapshot)

    assert len(clusters) == 1
    assert clusters[0].member_ids == ["paper:a", "paper:b"]
    assert clusters[0].label.startswith("Computer Vision")
    assert clusters[0].concepts[0]["node_id"] == "topic:vision"
    assert "paper:c" not in clusters[0].member_ids


def trend_snapshot() -> GraphSnapshot:
    nodes: list[GraphNode] = [
        node("topic:alpha", "topic", name="Alpha"),
        node("topic:beta", "topic", name="Beta"),
    ]
    edges: list[GraphEdge] = []
    alpha_documents: set[str] = set()
    beta_documents: set[str] = set()
    document_index = 0

    for window, start in (("baseline", date(2026, 1, 1)), ("recent", date(2026, 1, 8))):
        for source_id in ("source-a", "source-b"):
            for offset in range(4):
                document_id = f"paper:{document_index:02d}"
                published = start + timedelta(days=min(offset * 2, 6))
                nodes.append(
                    node(
                        document_id,
                        "paper",
                        properties={
                            "document_id": document_id,
                            "source_id": source_id,
                            "published_at": datetime.combine(
                                published, datetime.min.time(), tzinfo=UTC
                            ).isoformat(),
                        },
                    )
                )
                if window == "baseline" and source_id == "source-a" and offset == 0:
                    alpha_documents.add(document_id)
                if window == "recent" and offset < 2:
                    alpha_documents.add(document_id)
                if window == "recent" and source_id == "source-a" and offset < 2:
                    beta_documents.add(document_id)
                document_index += 1

    for document_id in sorted(alpha_documents):
        edges.append(
            GraphEdge(
                source_id=document_id,
                relation="ABOUT_TOPIC",
                target_id="topic:alpha",
                weight=1.0,
            )
        )
    for document_id in sorted(beta_documents):
        edges.append(
            GraphEdge(
                source_id=document_id,
                relation="ABOUT_TOPIC",
                target_id="topic:beta",
                weight=1.0,
            )
        )
    return GraphSnapshot(nodes=nodes, edges=edges)


def test_trends_use_source_normalized_shares_and_exact_growth_formula():
    report = compute_trends(trend_snapshot())

    assert report.status == "ok"
    assert report.as_of == date(2026, 1, 14)
    assert report.window_days == 7
    assert report.formula == TREND_FORMULA
    alpha = next(item for item in report.hot if item.topic == "Alpha")
    assert alpha.recent_count == 4
    assert alpha.baseline_count == 1
    assert alpha.recent_share == pytest.approx(0.5)
    assert alpha.baseline_share == pytest.approx(0.125)
    expected_growth = math.log2((0.5 + 0.01) / (0.125 + 0.01))
    assert alpha.growth == pytest.approx(expected_growth)
    assert alpha.emerging_score == pytest.approx(
        expected_growth * math.log1p(4)
    )
    assert report.emerging[0].topic == "Alpha"

    beta = next(item for item in report.hot if item.topic == "Beta")
    assert beta.recent_count == 2
    assert beta.growth > 0.0
    assert beta.emerging_score == 0.0
    assert beta not in report.emerging


def test_trends_return_explicit_insufficient_history():
    snapshot = GraphSnapshot(
        nodes=[
            node(
                "paper:a",
                "paper",
                properties={
                    "source_id": "arxiv",
                    "published_at": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                },
            ),
            node(
                "paper:b",
                "paper",
                properties={
                    "source_id": "arxiv",
                    "published_at": datetime(2026, 1, 5, tzinfo=UTC).isoformat(),
                },
            ),
        ],
        edges=[],
    )

    report = compute_trends(snapshot)

    assert report.status == "insufficient_history"
    assert report.coverage_days == 5
    assert report.hot == []
    assert report.emerging == []


def test_hot_topics_exclude_topics_without_recent_documents():
    snapshot = trend_snapshot()
    snapshot.nodes.append(node("topic:old", "topic", name="Old topic"))
    baseline_document = next(
        item.node_id
        for item in snapshot.nodes
        if item.node_type == "paper"
        and item.properties["published_at"].startswith("2026-01-01")
    )
    snapshot.edges.append(
        GraphEdge(
            source_id=baseline_document,
            relation="ABOUT_TOPIC",
            target_id="topic:old",
            weight=1.0,
        )
    )

    report = compute_trends(snapshot)

    assert "Old topic" not in {item.topic for item in report.hot}


def test_trends_normalize_publication_timestamps_to_utc_dates():
    west = timezone(timedelta(hours=-5))
    snapshot = GraphSnapshot(
        nodes=[
            node(
                "paper:first",
                "paper",
                properties={
                    "source_id": "feed",
                    "published_at": datetime(2026, 1, 1, 23, tzinfo=west).isoformat(),
                },
            ),
            node(
                "paper:last",
                "paper",
                properties={
                    "source_id": "feed",
                    "published_at": datetime(2026, 1, 7, 23, tzinfo=west).isoformat(),
                },
            ),
        ],
        edges=[],
    )

    report = compute_trends(snapshot)

    assert report.as_of == date(2026, 1, 8)
    assert report.coverage_days == 7
