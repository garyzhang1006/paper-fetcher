from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from arxiv_kg.db import Database
from arxiv_kg.models import PaperRecord, SourceItem


def make_paper(arxiv_id: str) -> PaperRecord:
    now = datetime(2026, 7, 10, tzinfo=UTC)
    return PaperRecord(
        arxiv_id=arxiv_id,
        versioned_id=f"{arxiv_id}v1",
        version=1,
        title=f"Paper {arxiv_id}",
        abstract="A machine-learning paper.",
        authors=["Ada Researcher"],
        categories=["cs.LG"],
        primary_category="cs.LG",
        published_at=now,
        updated_at=now,
        abs_url=f"https://arxiv.org/abs/{arxiv_id}v1",
    )


def test_batch_paper_upsert_rolls_back_on_midstream_failure(tmp_path):
    db = Database(tmp_path / "batch.sqlite3")

    def broken_batch():
        yield make_paper("2607.00001")
        raise RuntimeError("dataset line 2 is malformed")

    with pytest.raises(RuntimeError, match="dataset line 2 is malformed"):
        db.upsert_papers(broken_batch())

    assert db.counts()["papers"] == 0


def test_graph_rebuild_rolls_back_to_previous_snapshot_on_insert_failure(tmp_path):
    db = Database(tmp_path / "graph.sqlite3")
    nodes = [
        {
            "node_id": "paper:one",
            "node_type": "paper",
            "name": "Paper one",
            "canonical_name": "paper one",
            "properties": {"document_id": "2607.00001"},
        },
        {
            "node_id": "topic:graphs",
            "node_type": "topic",
            "name": "Graphs",
            "canonical_name": "graphs",
            "properties": {},
        },
    ]
    edge = {
        "source_id": "paper:one",
        "relation": "ABOUT_TOPIC",
        "target_id": "topic:graphs",
        "properties": {"evidence": "Supported by abstract."},
    }
    db.replace_graph(
        build_id="a" * 64,
        document_count=1,
        nodes=nodes,
        edges=[edge],
        analysis={"hubs": [], "clusters": [], "trends": []},
    )

    with pytest.raises(sqlite3.IntegrityError):
        db.replace_graph(
            build_id="b" * 64,
            document_count=1,
            nodes=nodes,
            edges=[edge, edge],
            analysis={"hubs": [], "clusters": [], "trends": []},
        )

    assert db.get_graph_analysis()["build_id"] == "a" * 64
    assert len(db.list_graph_nodes()) == 2
    assert len(db.list_graph_edges()) == 1


def test_graph_rebuild_rejects_dangling_edge_before_mutation(tmp_path):
    db = Database(tmp_path / "dangling.sqlite3")
    with pytest.raises(ValueError, match="references a missing node"):
        db.replace_graph(
            build_id="b" * 64,
            document_count=0,
            nodes=[],
            edges=[
                {
                    "source_id": "paper:missing",
                    "relation": "ABOUT_TOPIC",
                    "target_id": "topic:missing",
                    "properties": {},
                }
            ],
            analysis={},
        )

    assert db.get_graph_analysis() is None


def test_source_batch_and_checkpoint_roll_back_together(tmp_path):
    db = Database(tmp_path / "sources.sqlite3")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    item = SourceItem(
        item_id="source_item:one",
        source_id="research-feed",
        source_name="Research Feed",
        source_kind="research_blog",
        external_id="entry-one",
        title="Graph research",
        content_text="A graph research post.",
        canonical_url="https://example.org/posts/one",
        published_at=now,
        retrieved_at=now,
        content_sha256="a" * 64,
        source_topics=["machine learning"],
    )

    def broken_items():
        yield item
        raise RuntimeError("feed normalization stopped")

    with pytest.raises(RuntimeError, match="feed normalization stopped"):
        db.upsert_source_items(
            broken_items(),
            checkpoint_key="feed:research-feed:last_success",
            checkpoint_value=now.isoformat(),
        )

    assert db.counts()["source_items"] == 0
    assert db.get_state("feed:research-feed:last_success") is None


def test_graph_rebuild_rejects_false_document_count_and_bad_build_id(tmp_path):
    db = Database(tmp_path / "metadata.sqlite3")
    nodes = [{
        "node_id": "paper:one",
        "node_type": "paper",
        "name": "One",
        "canonical_name": "one",
        "properties": {"document_id": "2607.00001"},
    }]

    with pytest.raises(ValueError, match="64-character"):
        db.replace_graph(
            build_id="not-a-hash",
            document_count=1,
            nodes=nodes,
            edges=[],
            analysis={},
        )
    with pytest.raises(ValueError, match="does not match"):
        db.replace_graph(
            build_id="c" * 64,
            document_count=999,
            nodes=nodes,
            edges=[],
            analysis={},
        )


def test_graph_search_normalizes_punctuation_and_original_id_is_indexed(tmp_path):
    db = Database(tmp_path / "lookup.sqlite3")
    nodes = [
        {
            "node_id": "paper:one",
            "node_type": "paper",
            "name": "One",
            "canonical_name": "one",
            "properties": {"document_id": "2607.00001"},
        },
        {
            "node_id": "category:cv",
            "node_type": "category",
            "name": "cs.CV",
            "canonical_name": "cs cv",
            "properties": {},
        },
    ]
    db.replace_graph(
        build_id="d" * 64,
        document_count=1,
        nodes=nodes,
        edges=[],
        analysis={},
    )

    assert db.list_graph_nodes(search="cs.CV")[0]["name"] == "cs.CV"
    assert db.get_graph_node("2607.00001")["node_id"] == "paper:one"
    with db.connect() as con:
        plan = con.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM kg_nodes WHERE document_id = ?",
            ("2607.00001",),
        ).fetchall()
    assert any("idx_kg_nodes_document_id" in row[3] for row in plan)
