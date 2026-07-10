from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from arxiv_kg.daily import main, parse_args
from arxiv_kg.fetcher import FetchReport


def test_daily_defaults_match_bounded_notebook_live_run():
    args = parse_args([])

    assert args.db == "data/arxiv_kg.sqlite3"
    assert args.categories == ["cs.LG"]
    assert args.max_results == 200
    assert args.first_run_lookback_hours == 24
    assert args.overlap_hours == 24
    assert args.revision_max_results == 200
    assert args.revision_first_run_lookback_hours == 24


@pytest.mark.parametrize("value", ["hep-th", "quant-ph", "cs.LG", "astro-ph.CO"])
def test_daily_cli_accepts_valid_arxiv_category_formats(value):
    assert parse_args(["--category", value]).categories == [value]


def test_daily_cli_runs_fetch_and_prints_json_report(tmp_path, monkeypatch, capsys):
    now = datetime(2026, 6, 24, 12, tzinfo=UTC)
    received = {}

    def fake_fetch(db, **options):
        received["db_path"] = db.path
        received["options"] = options
        return FetchReport(
            start_utc=now,
            revision_start_utc=now,
            end_utc=now,
            new_query_received=1,
            revision_query_received=0,
            unique_results_processed=1,
            inserted=1,
            updated=0,
            unchanged=0,
        )

    monkeypatch.setattr("arxiv_kg.daily.fetch_recent_papers", fake_fetch)
    db_path = tmp_path / "daily.sqlite3"

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--category",
            "cs.LG",
            "--category",
            "cs.RO",
            "--max-results",
            "50",
        ]
    )

    assert exit_code == 0
    assert received["db_path"] == db_path
    assert received["options"]["categories"] == ["cs.LG", "cs.RO"]
    assert received["options"]["max_results"] == 50
    assert json.loads(capsys.readouterr().out)["end_utc"] == now.isoformat()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-results", "0"],
        ["--first-run-lookback-hours", "0"],
        ["--overlap-hours", "-1"],
        ["--revision-max-results", "0"],
        ["--revision-first-run-lookback-hours", "0"],
    ],
)
def test_daily_cli_rejects_nonpositive_bounds(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_daily_cli_propagates_fetch_failure(tmp_path, monkeypatch):
    def fail_fetch(*_args, **_kwargs):
        raise RuntimeError("configured cap reached")

    monkeypatch.setattr("arxiv_kg.daily.fetch_recent_papers", fail_fetch)

    with pytest.raises(RuntimeError, match="configured cap reached"):
        main(["--db", str(tmp_path / "daily.sqlite3")])
