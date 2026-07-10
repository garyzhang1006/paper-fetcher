from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import arxiv
import pytest

from arxiv_kg.db import Database
from arxiv_kg.fetcher import (
    FETCH_STATE_KEY,
    UPDATE_STATE_KEY,
    build_category_query,
    fetch_recent_papers,
)


def fake_result(number: int, *, version: int = 1, updated: datetime) -> SimpleNamespace:
    short_id = f"2606.{number:05d}v{version}"
    return SimpleNamespace(
        get_short_id=lambda: short_id,
        title=f"Paper {number}",
        summary="A test abstract about machine learning.",
        authors=[SimpleNamespace(name="Test Student", affiliation=[])],
        categories=["cs.LG"],
        primary_category="cs.LG",
        published=datetime(2026, 6, 1, tzinfo=UTC),
        updated=updated,
        entry_id=f"https://arxiv.org/abs/{short_id}",
        pdf_url=f"https://arxiv.org/pdf/{short_id}",
        doi=None,
        journal_ref=None,
        comment=None,
    )


class FakeClient:
    new_results: list[SimpleNamespace] = []
    revision_results: list[SimpleNamespace] = []

    def __init__(self, **_: object):
        pass

    def results(self, search: arxiv.Search):
        if search.sort_by == arxiv.SortCriterion.SubmittedDate:
            yield from self.new_results
        else:
            yield from self.revision_results


def test_category_query_uses_utc_minute_window():
    query = build_category_query(
        ["cs.LG", "stat.ML"],
        datetime(2026, 6, 23, 1, 2, tzinfo=UTC),
        datetime(2026, 6, 24, 3, 4, tzinfo=UTC),
    )
    assert "cat:cs.LG OR cat:stat.ML" in query
    assert "submittedDate:[202606230102 TO 202606240304]" in query


def test_saturated_query_does_not_advance_checkpoint(tmp_path, monkeypatch):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    FakeClient.new_results = [
        fake_result(1, updated=now),
        fake_result(2, updated=now),
        fake_result(3, updated=now),
    ]
    FakeClient.revision_results = []
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")

    with pytest.raises(RuntimeError, match="max-results"):
        fetch_recent_papers(
            db,
            categories=["cs.LG"],
            max_results=2,
            scan_revisions=False,
            now=now,
        )

    assert db.counts()["papers"] == 2
    assert db.get_state(FETCH_STATE_KEY) is None


def test_three_day_gap_after_checkpoint_misses_nothing(tmp_path, monkeypatch):
    """A skipped daily job still covers its full gap on the next run."""
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")

    day0 = datetime(2026, 6, 20, 12, tzinfo=UTC)
    FakeClient.new_results = [fake_result(1, updated=day0)]
    FakeClient.revision_results = []
    fetch_recent_papers(
        db, categories=["cs.LG"], max_results=10, scan_revisions=False, now=day0
    )

    day3 = day0 + timedelta(days=3)
    FakeClient.new_results = [
        fake_result(2, updated=day0 + timedelta(days=1)),
        fake_result(3, updated=day0 + timedelta(days=2)),
    ]
    report = fetch_recent_papers(
        db, categories=["cs.LG"], max_results=10, scan_revisions=False, now=day3
    )

    assert report.start_utc == day0 - timedelta(hours=24)
    assert report.start_utc < day0
    assert db.get_paper("2606.00002") is not None
    assert db.get_paper("2606.00003") is not None
    assert db.counts()["papers"] == 3
    assert db.get_state(FETCH_STATE_KEY) == day3.isoformat()


def test_query_with_cs_ro_exact_expression():
    query = build_category_query(
        ["cs.LG", "stat.ML", "cs.RO"],
        datetime(2026, 6, 23, 0, 0, tzinfo=UTC),
        datetime(2026, 6, 24, 0, 0, tzinfo=UTC),
    )
    assert query == (
        "((cat:cs.LG OR cat:stat.ML OR cat:cs.RO) AND "
        "submittedDate:[202606230000 TO 202606240000])"
    )


def test_revised_paper_title_change_is_stored(tmp_path, monkeypatch):
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)

    FakeClient.new_results = [fake_result(1, version=1, updated=now - timedelta(days=1))]
    FakeClient.revision_results = []
    fetch_recent_papers(
        db,
        categories=["cs.LG"],
        max_results=10,
        scan_revisions=False,
        now=now - timedelta(hours=12),
    )
    assert db.get_paper("2606.00001").title == "Paper 1"

    v2 = fake_result(1, version=2, updated=now)
    v2.title = "Paper 1: Revised With a New Title"
    FakeClient.new_results = [v2]
    fetch_recent_papers(db, categories=["cs.LG"], max_results=10, scan_revisions=False, now=now)

    stored = db.get_paper("2606.00001")
    assert stored.version == 2
    assert stored.title == "Paper 1: Revised With a New Title"


def test_revision_scan_updates_existing_paper(tmp_path, monkeypatch):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    db = Database(tmp_path / "test.sqlite3")

    FakeClient.new_results = [fake_result(1, version=1, updated=now - timedelta(days=2))]
    FakeClient.revision_results = []
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    fetch_recent_papers(
        db,
        categories=["cs.LG"],
        max_results=10,
        scan_revisions=False,
        now=now - timedelta(days=1),
    )
    assert db.get_paper("2606.00001").version == 1

    FakeClient.new_results = []
    FakeClient.revision_results = [
        fake_result(1, version=2, updated=now - timedelta(hours=1)),
        fake_result(99, version=1, updated=now - timedelta(days=10)),
    ]
    report = fetch_recent_papers(
        db,
        categories=["cs.LG"],
        max_results=10,
        revision_max_results=10,
        revision_first_run_lookback_hours=48,
        now=now,
    )

    assert report.updated == 1
    assert db.get_paper("2606.00001").version == 2
    assert db.get_state(FETCH_STATE_KEY) == now.isoformat()
    assert db.get_state(UPDATE_STATE_KEY) == now.isoformat()
