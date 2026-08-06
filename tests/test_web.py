from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from arxiv_kg.models import PaperRecord
from arxiv_kg.web import create_server, validate_fetch_payload


@pytest.fixture
def running_server(tmp_path):
    server = create_server(port=0, db_path=tmp_path / "web.sqlite3")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_validate_fetch_payload_normalizes_categories():
    options = validate_fetch_payload(
        {
            "categories": ["cs.LG", "stat.ML", "cs.LG"],
            "max_results": 50,
            "lookback_hours": 72,
            "scan_revisions": False,
        }
    )
    assert options["categories"] == ["cs.LG", "stat.ML"]
    assert options["max_results"] == 50
    assert options["first_run_lookback_hours"] == 72
    assert options["scan_revisions"] is False


@pytest.mark.parametrize("value", ["hep-th", "quant-ph", "cs.LG", "astro-ph.CO"])
def test_validate_fetch_payload_accepts_valid_category_formats(value):
    assert validate_fetch_payload({"categories": [value]})["categories"] == [value]


def test_fetch_defaults_match_notebook_live_example():
    options = validate_fetch_payload({"categories": ["cs.LG"]})
    assert options["max_results"] == 200
    assert options["first_run_lookback_hours"] == 24
    assert options["revision_max_results"] == 200
    assert options["revision_first_run_lookback_hours"] == 24
    assert options["scan_revisions"] is True


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"categories": []}, "Choose at least one"),
        ({"categories": ["not a category"]}, "Invalid arXiv category"),
        ({"categories": ["cs.LG"], "max_results": 0}, "between 1 and 500"),
        ({"categories": ["cs.LG"], "scan_revisions": "yes"}, "true or false"),
    ],
)
def test_validate_fetch_payload_rejects_bad_input(payload, message):
    with pytest.raises(ValueError, match=message):
        validate_fetch_payload(payload)


def test_ui_and_stats_endpoints(running_server):
    _, base_url = running_server
    with urlopen(f"{base_url}/") as response:
        html = response.read().decode("utf-8")
        assert response.status == 200
        assert "Paper Fetcher" in html
        assert '<option value="200" selected>200 papers</option>' in html
        assert 'value="cs.LG" checked' in html
        assert 'value="stat.ML" checked' not in html
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]

    with urlopen(f"{base_url}/api/stats") as response:
        assert json.load(response) == {
            "papers": 0,
            "source_items": 0,
            "features": 0,
            "nodes": 0,
            "edges": 0,
        }

    with urlopen(f"{base_url}/favicon.svg") as response:
        assert response.status == 200
        assert response.headers.get_content_type() == "image/svg+xml"


def test_paper_search_endpoint(running_server):
    server, base_url = running_server
    now = datetime(2026, 7, 9, tzinfo=UTC)
    server.database.upsert_paper(
        PaperRecord(
            arxiv_id="2607.00001",
            versioned_id="2607.00001v1",
            version=1,
            title="Graph Learning for Robot Control",
            abstract="A graph model for reliable robot control.",
            authors=["Ada Student"],
            categories=["cs.RO", "cs.LG"],
            primary_category="cs.RO",
            published_at=now,
            updated_at=now,
            abs_url="https://arxiv.org/abs/2607.00001v1",
            pdf_url="https://arxiv.org/pdf/2607.00001v1",
        )
    )

    with urlopen(f"{base_url}/api/papers?search=robot&category=cs.RO") as response:
        payload = json.load(response)
    assert payload["total"] == 1
    assert payload["papers"][0]["title"] == "Graph Learning for Robot Control"
    assert payload["categories"] == ["cs.LG", "cs.RO"]

    with urlopen(f"{base_url}/api/papers?search=biology") as response:
        assert json.load(response)["papers"] == []

    with pytest.raises(HTTPError) as caught:
        urlopen(f"{base_url}/api/papers?limit=0")
    assert caught.value.code == 400


def test_paper_search_covers_rows_beyond_first_five_hundred(running_server):
    server, base_url = running_server
    now = datetime(2026, 7, 9, tzinfo=UTC)
    papers = []
    for index in range(501):
        identifier = f"2607.{index + 1:05d}"
        papers.append(
            PaperRecord(
                arxiv_id=identifier,
                versioned_id=f"{identifier}v1",
                version=1,
                title=(
                    "Unique oldest searchable paper"
                    if index == 0
                    else f"Routine paper {index}"
                ),
                abstract="Saved abstract.",
                authors=["Ada Student"],
                categories=["cs.LG"],
                primary_category="cs.LG",
                published_at=now.replace(day=1) if index == 0 else now,
                updated_at=now,
                abs_url=f"https://arxiv.org/abs/{identifier}v1",
            )
        )
    server.database.upsert_papers(papers)

    with urlopen(f"{base_url}/api/papers?search=unique%20oldest") as response:
        payload = json.load(response)

    assert payload["total"] == 1
    assert payload["papers"][0]["arxiv_id"] == "2607.00001"


def test_fetch_endpoint_returns_actionable_validation_error(running_server):
    _, base_url = running_server
    request = Request(
        f"{base_url}/api/fetch",
        data=json.dumps({"categories": []}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 400
    assert "Choose at least one arXiv category" in caught.value.read().decode("utf-8")


def test_graph_endpoints_report_empty_state_and_validate_limits(running_server):
    _, base_url = running_server

    with urlopen(f"{base_url}/api/graph") as response:
        payload = json.load(response)
    assert payload == {
        "status": "not_built",
        "build_id": None,
        "nodes": [],
        "edges": [],
    }

    with urlopen(f"{base_url}/api/trends") as response:
        assert json.load(response) == {
            "status": "not_built",
            "build_id": None,
            "trends": [],
        }

    with pytest.raises(HTTPError) as caught:
        urlopen(f"{base_url}/api/graph?limit=0")
    assert caught.value.code == 400
    assert "limit must be between 1 and 500" in caught.value.read().decode("utf-8")


def test_graph_and_placement_endpoints_return_active_snapshot(running_server):
    server, base_url = running_server
    nodes = [
        {
            "node_id": "paper:one",
            "node_type": "paper",
            "name": "Graph Learning",
            "canonical_name": "graph learning",
            "properties": {"document_id": "2607.00001", "summary": "A graph paper."},
        },
        {
            "node_id": "topic:graphs",
            "node_type": "topic",
            "name": "Graph learning",
            "canonical_name": "graph learning",
            "properties": {},
        },
    ]
    edges = [
        {
            "source_id": "paper:one",
            "relation": "ABOUT_TOPIC",
            "target_id": "topic:graphs",
            "properties": {"evidence": "Graph learning is the domain."},
        }
    ]
    server.database.replace_graph(
        build_id="f" * 64,
        document_count=1,
        nodes=nodes,
        edges=edges,
        analysis={
            "hubs": [{"node_id": "topic:graphs", "pagerank": 0.6}],
            "clusters": [{"cluster_id": "cluster-1", "size": 1}],
            "trends": [{"concept": "Graph learning", "status": "hot"}],
        },
    )

    with urlopen(f"{base_url}/api/graph?node_type=paper") as response:
        graph = json.load(response)
    assert graph["status"] == "ready"
    assert graph["build_id"] == "f" * 64
    assert [node["node_id"] for node in graph["nodes"]] == ["paper:one"]

    with urlopen(f"{base_url}/api/placement?id=2607.00001") as response:
        placement = json.load(response)
    assert placement["document"]["node_id"] == "paper:one"
    assert placement["relationships"][0]["relation"] == "ABOUT_TOPIC"
    assert placement["neighbors"][0]["node_id"] == "topic:graphs"

    with urlopen(f"{base_url}/api/hubs") as response:
        hubs = json.load(response)
    assert hubs["build_id"] == "f" * 64
    assert hubs["hubs"][0]["node_id"] == "topic:graphs"

    with pytest.raises(HTTPError) as caught:
        urlopen(f"{base_url}/api/placement?id=missing")
    assert caught.value.code == 404
