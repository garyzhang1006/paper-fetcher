from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from arxiv_kg.agent import main, parse_args
from arxiv_kg.db import Database
from arxiv_kg.models import PaperRecord


def write_dataset(path) -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    papers = []
    for index in range(4):
        published = start + timedelta(days=index * 4)
        paper = PaperRecord(
            arxiv_id=f"2607.{index + 1:05d}",
            versioned_id=f"2607.{index + 1:05d}v1",
            version=1,
            title="Graph Neural Network for Image Classification",
            abstract=(
                "We use a graph neural network for image classification. "
                "We evaluate on CIFAR-10 and report accuracy."
            ),
            authors=["Ada Researcher"],
            categories=["cs.LG"],
            primary_category="cs.LG",
            published_at=published,
            updated_at=published,
            abs_url=f"https://arxiv.org/abs/2607.{index + 1:05d}v1",
        )
        papers.append(paper.model_dump(mode="json"))
    path.write_text(
        "".join(json.dumps(paper) + "\n" for paper in papers),
        encoding="utf-8",
    )


def test_agent_defaults_to_complete_daily_inputs():
    args = parse_args([])

    assert args.db == "data/arxiv_kg.sqlite3"
    assert args.dataset == "dataset/papers.jsonl"
    assert args.sources == "config/sources.json"
    assert args.categories == ["cs.LG"]
    assert args.offline is False
    assert args.no_arxiv is False


def test_offline_agent_builds_deterministic_evidence_graph_and_artifacts(
    tmp_path, capsys
):
    dataset = tmp_path / "papers.jsonl"
    database = tmp_path / "agent.sqlite3"
    output = tmp_path / "knowledge_graph"
    write_dataset(dataset)
    arguments = [
        "--offline",
        "--db",
        str(database),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output),
    ]

    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "ok"
    assert first["bootstrap"]["inserted"] == 4
    assert first["features_extracted"] == 4
    assert first["graph"]["documents"] == 4
    assert first["database_counts"]["papers"] == 4
    assert first["stored_sources"] == []

    db = Database(database)
    counts = db.counts()
    assert counts["papers"] == 4
    assert counts["features"] == 4
    assert counts["nodes"] > 4
    assert counts["edges"] > 4
    assert db.list_graph_nodes(node_type="method")[0]["name"] == "graph neural network"
    assert db.list_graph_nodes(node_type="research_goal")[0]["name"] == "classification"

    assert main(arguments) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["bootstrap"]["unchanged"] == 4
    assert second["features_extracted"] == 0
    assert second["graph"]["build_id"] == first["graph"]["build_id"]

    assert json.loads((output / "summary.json").read_text())["status"] == "ok"
    trend_artifact = json.loads((output / "trends.json").read_text())
    assert trend_artifact["formula"].startswith("source_share=")


def test_agent_persists_degraded_report_after_collection_failure(
    tmp_path, capsys, monkeypatch
):
    database = tmp_path / "degraded.sqlite3"
    output = tmp_path / "knowledge_graph"

    def fail_arxiv(*_args, **_kwargs):
        raise RuntimeError("simulated arXiv outage")

    monkeypatch.setattr("arxiv_kg.fetcher.fetch_recent_papers", fail_arxiv)
    exit_code = main(
        [
            "--no-bootstrap",
            "--no-feeds",
            "--db",
            str(database),
            "--output-dir",
            str(output),
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert report["status"] == "degraded"
    assert report["collection_errors"] == ["arxiv: simulated arXiv outage"]
    assert report["graph"]["documents"] == 0
    assert json.loads((output / "summary.json").read_text())["status"] == "degraded"
