from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import arxiv
import pytest

from arxiv_kg.db import Database
from arxiv_kg.fetcher import (
    FETCH_STATE_KEY,
    UPDATE_STATE_KEY,
    build_category_query,
    category_state_key,
    fetch_recent_papers,
    result_to_paper,
)
from arxiv_kg.ids import split_arxiv_version

FETCH_KEY = category_state_key(FETCH_STATE_KEY, ["cs.LG"])
UPDATE_KEY = category_state_key(UPDATE_STATE_KEY, ["cs.LG"])


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


def test_split_arxiv_version_handles_notebook_examples():
    assert split_arxiv_version("2107.05580v3") == ("2107.05580", 3)
    assert split_arxiv_version("quant-ph/0201082v1") == ("quant-ph/0201082", 1)
    assert split_arxiv_version("2107.05580") == ("2107.05580", 1)


def test_paper_upsert_is_idempotent_and_revision_clears_stale_files(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    v1 = result_to_paper(
        fake_result(1, version=1, updated=datetime(2026, 6, 20, tzinfo=UTC))
    )

    assert db.upsert_paper(v1) == "inserted"
    assert db.upsert_paper(v1) == "unchanged"
    db.set_paper_file(v1.arxiv_id, kind="pdf", path="papers/v1.pdf")
    db.set_paper_file(v1.arxiv_id, kind="text", path="papers/v1.txt")

    v2_result = fake_result(
        1, version=2, updated=datetime(2026, 6, 24, tzinfo=UTC)
    )
    v2_result.title = "Paper 1: Revised"
    assert db.upsert_paper(result_to_paper(v2_result)) == "updated"
    assert db.counts()["papers"] == 1
    assert db.get_paper(v1.arxiv_id).title == "Paper 1: Revised"
    assert db.get_paper_paths(v1.arxiv_id) == (None, None)


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

    # The first two rows are safe to retain, but the next run must retry the
    # same interval because the successful-run checkpoint was not advanced.
    assert db.counts()["papers"] == 2
    assert db.get_state(FETCH_KEY) is None


def test_new_query_final_page_failure_retains_rows_without_checkpoint(
    tmp_path, monkeypatch
):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    previous = now - timedelta(days=1)

    class FailingClient(FakeClient):
        def results(self, search: arxiv.Search):
            if search.sort_by == arxiv.SortCriterion.SubmittedDate:
                yield fake_result(1, updated=now)
                raise RuntimeError("final page failed")
            return

    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FailingClient)
    db = Database(tmp_path / "test.sqlite3")
    db.set_state(FETCH_KEY, previous.isoformat())

    with pytest.raises(RuntimeError, match="final page failed"):
        fetch_recent_papers(
            db,
            categories=["cs.LG"],
            max_results=10,
            scan_revisions=False,
            now=now,
        )

    assert db.counts()["papers"] == 1
    assert db.get_state(FETCH_KEY) == previous.isoformat()


def test_three_day_gap_after_checkpoint_misses_nothing(tmp_path, monkeypatch):
    """If the daily job is skipped for 3 days, the next run must still cover
    the whole gap because the query starts at (last checkpoint - overlap)."""
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")

    # Day 0: one paper, run succeeds, checkpoint is written.
    day0 = datetime(2026, 6, 20, 12, tzinfo=UTC)
    FakeClient.new_results = [fake_result(1, updated=day0)]
    FakeClient.revision_results = []
    fetch_recent_papers(db, categories=["cs.LG"], max_results=10,
                        scan_revisions=False, now=day0)

    # Days 1 and 2: job never runs (no fetch calls at all).

    # Day 3: two papers were submitted during the gap. The job runs again.
    day3 = day0 + timedelta(days=3)
    FakeClient.new_results = [
        fake_result(2, updated=day0 + timedelta(days=1)),
        fake_result(3, updated=day0 + timedelta(days=2)),
    ]
    FakeClient.revision_results = []
    report = fetch_recent_papers(db, categories=["cs.LG"], max_results=10,
                                 scan_revisions=False, now=day3)

    # 1. The day-3 query starts at (day0 checkpoint - 24h overlap), so the whole
    #    gap [day0, day3] sits inside the query window and nothing is skipped.
    assert report.start_utc == day0 - timedelta(hours=24)
    assert report.start_utc < day0

    # 2. Both papers submitted during the gap landed in the database.
    assert db.get_paper("2606.00002") is not None
    assert db.get_paper("2606.00003") is not None
    assert db.counts()["papers"] == 3

    # 3. The checkpoint advanced to day3 only after the complete, uncapped run.
    assert db.get_state(FETCH_KEY) == day3.isoformat()


def test_different_category_set_uses_its_own_first_run_window(tmp_path, monkeypatch):
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")
    first_run = datetime(2026, 6, 20, 12, tzinfo=UTC)
    second_run = first_run + timedelta(days=3)
    FakeClient.new_results = []
    FakeClient.revision_results = []

    fetch_recent_papers(
        db,
        categories=["cs.LG"],
        max_results=10,
        scan_revisions=False,
        first_run_lookback_hours=24,
        now=first_run,
    )
    report = fetch_recent_papers(
        db,
        categories=["cs.RO"],
        max_results=10,
        scan_revisions=False,
        first_run_lookback_hours=72,
        now=second_run,
    )

    assert report.start_utc == second_run - timedelta(hours=72)


def test_query_with_cs_ro_exact_expression():
    """Notebook 1, exercise 1: add cs.RO and test the exact query string."""
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
    """Notebook 1, exercise 3: a v2 with a new title replaces the stored title."""
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)

    FakeClient.new_results = [fake_result(1, version=1, updated=now - timedelta(days=1))]
    FakeClient.revision_results = []
    fetch_recent_papers(db, categories=["cs.LG"], max_results=10,
                        scan_revisions=False, now=now - timedelta(hours=12))
    assert db.get_paper("2606.00001").title == "Paper 1"

    v2 = fake_result(1, version=2, updated=now)
    v2.title = "Paper 1: Revised With a New Title"
    FakeClient.new_results = [v2]
    fetch_recent_papers(db, categories=["cs.LG"], max_results=10,
                        scan_revisions=False, now=now)

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

    # A later run has no new submission, but its last-updated scan finds v2.
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
    assert db.get_state(FETCH_KEY) == now.isoformat()
    assert db.get_state(UPDATE_KEY) == now.isoformat()


def test_revision_old_sentinel_reaches_cutoff_without_saturation(
    tmp_path, monkeypatch
):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    FakeClient.new_results = []
    FakeClient.revision_results = [
        fake_result(1, updated=now - timedelta(hours=1)),
        fake_result(99, updated=now - timedelta(days=2)),
    ]
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")

    report = fetch_recent_papers(
        db,
        categories=["cs.LG"],
        max_results=10,
        revision_max_results=1,
        revision_first_run_lookback_hours=24,
        now=now,
    )

    assert report.unique_results_processed == 1
    assert db.get_state(FETCH_KEY) == now.isoformat()
    assert db.get_state(UPDATE_KEY) == now.isoformat()


def test_saturated_revision_query_does_not_advance_either_checkpoint(
    tmp_path, monkeypatch
):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    previous = now - timedelta(days=1)
    FakeClient.new_results = []
    FakeClient.revision_results = [
        fake_result(1, updated=now - timedelta(hours=1)),
        fake_result(2, updated=now - timedelta(hours=2)),
    ]
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")
    db.set_state(FETCH_KEY, previous.isoformat())
    db.set_state(UPDATE_KEY, previous.isoformat())

    with pytest.raises(RuntimeError, match="revision-max-results"):
        fetch_recent_papers(
            db,
            categories=["cs.LG"],
            max_results=10,
            revision_max_results=1,
            now=now,
        )

    assert db.get_paper("2606.00001") is not None
    assert db.get_state(FETCH_KEY) == previous.isoformat()
    assert db.get_state(UPDATE_KEY) == previous.isoformat()


def test_revision_final_page_failure_retains_rows_without_advancing_checkpoints(
    tmp_path, monkeypatch
):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    previous = now - timedelta(days=1)

    class FailingRevisionClient(FakeClient):
        def results(self, search: arxiv.Search):
            if search.sort_by == arxiv.SortCriterion.SubmittedDate:
                return
            yield fake_result(1, version=2, updated=now - timedelta(hours=1))
            raise RuntimeError("revision final page failed")

    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FailingRevisionClient)
    db = Database(tmp_path / "test.sqlite3")
    db.set_state(FETCH_KEY, previous.isoformat())
    db.set_state(UPDATE_KEY, previous.isoformat())

    with pytest.raises(RuntimeError, match="revision final page failed"):
        fetch_recent_papers(
            db,
            categories=["cs.LG"],
            max_results=10,
            revision_max_results=10,
            now=now,
        )

    assert db.get_paper("2606.00001").version == 2
    assert db.get_state(FETCH_KEY) == previous.isoformat()
    assert db.get_state(UPDATE_KEY) == previous.isoformat()


def test_checkpoint_updates_are_atomic(tmp_path, monkeypatch):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    FakeClient.new_results = []
    FakeClient.revision_results = []
    monkeypatch.setattr("arxiv_kg.fetcher.arxiv.Client", FakeClient)
    db = Database(tmp_path / "test.sqlite3")
    with db.connect() as con:
        con.execute(
            f"""
            CREATE TRIGGER fail_revision_checkpoint
            BEFORE INSERT ON pipeline_state
            WHEN NEW.state_key = '{UPDATE_KEY}'
            BEGIN
                SELECT RAISE(ABORT, 'forced checkpoint failure');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced checkpoint failure"):
        fetch_recent_papers(
            db,
            categories=["cs.LG"],
            max_results=10,
            now=now,
        )

    assert db.get_state(FETCH_KEY) is None
    assert db.get_state(UPDATE_KEY) is None
