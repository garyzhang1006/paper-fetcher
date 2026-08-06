from __future__ import annotations

import json

import pytest

from arxiv_kg.bootstrap import (
    BootstrapDataError,
    BootstrapSummary,
    iter_curated_papers,
    load_curated_papers,
)
from arxiv_kg.models import PaperRecord


def record(
    versioned_id: str = "2607.00001v1",
    *,
    arxiv_id: str = "2607.00001",
    title: str = "A Valid Curated Paper",
) -> dict[str, object]:
    return {
        "arxiv_id": arxiv_id,
        "versioned_id": versioned_id,
        "title": title,
        "abstract": "We evaluate a method on a documented task.",
        "authors": ["Ada Student"],
        "categories": ["cs.LG", "stat.ML"],
        "primary_category": "cs.LG",
        "published_at": "2026-07-03T00:00:00Z",
        "updated_at": "2026-07-03T01:00:00Z",
        "abs_url": f"https://arxiv.org/abs/{versioned_id}",
        "pdf_url": f"https://arxiv.org/pdf/{versioned_id}",
        "doi": None,
        "journal_ref": None,
        "comment": None,
        "submitted_date": "2026-07-03",
    }


def write_jsonl(path, rows: list[object], *, blank_after_first: bool = False) -> None:
    lines: list[str] = []
    for index, row in enumerate(rows):
        lines.append(json.dumps(row))
        if blank_after_first and index == 0:
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_curated_papers_returns_records_and_summary(tmp_path):
    path = tmp_path / "papers.jsonl"
    write_jsonl(
        path,
        [record(), record("2607.00002v1", arxiv_id="2607.00002")],
        blank_after_first=True,
    )

    result = load_curated_papers(path)

    assert all(isinstance(paper, PaperRecord) for paper in result)
    assert [paper.arxiv_id for paper in result] == ["2607.00001", "2607.00002"]
    assert result.summary == BootstrapSummary(
        lines_read=3,
        blank_lines=1,
        records_parsed=2,
        unique_papers=2,
        duplicate_versions=0,
        superseded_versions=0,
    )


def test_deduplicates_exact_versions_and_keeps_newest_canonical_version(tmp_path):
    path = tmp_path / "papers.jsonl"
    v2 = record(
        "https://arxiv.org/abs/2607.00001v2",
        arxiv_id="2607.00001v1",
        title="Revised Paper",
    )
    v2["updated_at"] = "2026-07-04T01:00:00+00:00"
    v1 = record()
    write_jsonl(path, [v2, v1, v2])

    result = load_curated_papers(path)

    assert len(result) == 1
    assert result.papers[0].arxiv_id == "2607.00001"
    assert result.papers[0].versioned_id == "2607.00001v2"
    assert result.papers[0].version == 2
    assert result.papers[0].title == "Revised Paper"
    assert result.summary.records_parsed == 3
    assert result.summary.duplicate_versions == 1
    assert result.summary.superseded_versions == 1


def test_missing_required_field_reports_source_line(tmp_path):
    path = tmp_path / "papers.jsonl"
    invalid = record()
    invalid.pop("abstract")
    write_jsonl(path, [record("2607.00002v1", arxiv_id="2607.00002"), invalid])

    with pytest.raises(
        BootstrapDataError,
        match=r"papers\.jsonl: line 2: missing required field\(s\): abstract",
    ):
        load_curated_papers(path)


def test_invalid_json_reports_line_and_column(tmp_path):
    path = tmp_path / "papers.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(
        BootstrapDataError,
        match=r"papers\.jsonl: line 1: invalid JSON at column 2",
    ):
        load_curated_papers(path)


def test_mismatched_canonical_ids_report_line(tmp_path):
    path = tmp_path / "papers.jsonl"
    write_jsonl(path, [record("2607.99999v3")])

    with pytest.raises(
        BootstrapDataError,
        match=r"line 1: arxiv_id and versioned_id identify different canonical papers",
    ):
        load_curated_papers(path)


def test_conflicting_copy_of_same_version_reports_both_lines(tmp_path):
    path = tmp_path / "papers.jsonl"
    write_jsonl(path, [record(), record(title="Conflicting Title")])

    with pytest.raises(
        BootstrapDataError,
        match=r"line 2: conflicting records for 2607\.00001v1; first seen on line 1",
    ):
        load_curated_papers(path)


def test_streaming_reader_is_lazy_and_reports_late_invalid_line(tmp_path):
    path = tmp_path / "papers.jsonl"
    write_jsonl(path, [record()])
    papers = iter_curated_papers(path)
    path.write_text(
        json.dumps(record()) + "\n{broken-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(BootstrapDataError, match=r"line 2: invalid JSON"):
        list(papers)


def test_streaming_reader_skips_exact_duplicate_versions(tmp_path):
    path = tmp_path / "papers.jsonl"
    write_jsonl(path, [record(), record()])

    papers = list(iter_curated_papers(path))

    assert [paper.versioned_id for paper in papers] == ["2607.00001v1"]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("authors", "Ada Student", "'authors' must be a non-empty list"),
        ("categories", [], "'categories' must be a non-empty list"),
        ("primary_category", "cs.CV", "primary_category must also appear"),
        ("published_at", "2026-07-03T00:00:00", "must include a timezone"),
        ("updated_at", "2026-07-02T00:00:00Z", "cannot precede published_at"),
    ],
)
def test_strict_field_validation_is_actionable(tmp_path, field, value, message):
    path = tmp_path / "papers.jsonl"
    invalid = record()
    invalid[field] = value
    write_jsonl(path, [invalid])

    with pytest.raises(BootstrapDataError, match=f"line 1: .*{message}"):
        load_curated_papers(path)
