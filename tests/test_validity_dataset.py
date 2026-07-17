import json
from pathlib import Path

import pytest

from arxiv_kg.models import PaperValidityRecord
from arxiv_kg.validity_dataset import _replace_directory, write_validity_dataset


def paper(identifier: str, day: str, abstract: str) -> dict[str, object]:
    return {
        "arxiv_id": identifier,
        "versioned_id": f"{identifier}v1",
        "submitted_date": day,
        "title": f"Paper {identifier}",
        "abstract": abstract,
    }


def write_source(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_batch_writes_one_valid_record_per_paper_and_date_shards(tmp_path):
    source = tmp_path / "papers.jsonl"
    output = tmp_path / "validity_envelopes"
    records = [
        paper("1", "2026-07-03", "We find improved accuracy on CIFAR-10."),
        paper("2", "2026-07-03", "This paper presents a descriptive framework."),
        paper("3", "2026-07-04", "Results show lower error under distribution shift."),
    ]
    write_source(source, records)

    manifest = write_validity_dataset(source, output, expected_count=3)

    assert manifest["processed_paper_count"] == 3
    assert manifest["paper_with_envelopes_count"] == 2
    assert manifest["paper_without_supported_claim_count"] == 1
    assert manifest["validity_envelope_count"] == 2
    assert [item["file"] for item in manifest["shards"]] == [
        "2026-07-03.jsonl",
        "2026-07-04.jsonl",
    ]
    output_records = []
    for shard in manifest["shards"]:
        lines = (output / shard["file"]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == shard["paper_count"]
        output_records.extend(PaperValidityRecord.model_validate_json(line) for line in lines)
    assert [record.arxiv_id for record in output_records] == ["1", "2", "3"]
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_count_mismatch_preserves_existing_output(tmp_path):
    source = tmp_path / "papers.jsonl"
    output = tmp_path / "validity_envelopes"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old output", encoding="utf-8")
    write_source(source, [paper("1", "2026-07-03", "We find a result.")])

    with pytest.raises(ValueError, match="expected 2 papers, got 1"):
        write_validity_dataset(source, output, expected_count=2)

    assert marker.read_text(encoding="utf-8") == "old output"
    assert list(tmp_path.glob(".validity_envelopes.staged-*")) == []


def test_duplicate_id_reports_input_line_and_preserves_existing_output(tmp_path):
    source = tmp_path / "papers.jsonl"
    output = tmp_path / "validity_envelopes"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old output", encoding="utf-8")
    write_source(
        source,
        [
            paper("1", "2026-07-03", "We find a result."),
            paper("1", "2026-07-04", "We show another result."),
        ],
    )

    with pytest.raises(ValueError, match="line 2: duplicate arxiv_id 1"):
        write_validity_dataset(source, output, expected_count=2)

    assert marker.read_text(encoding="utf-8") == "old output"
    assert list(tmp_path.glob(".validity_envelopes.staged-*")) == []


def test_record_status_must_match_presence_of_envelopes():
    with pytest.raises(ValueError, match="status must be no_supported_claim"):
        PaperValidityRecord(
            arxiv_id="1",
            source_versioned_id="1v1",
            source_scope="abstract",
            extractor="rules",
            extractor_version="1",
            status="extracted",
            validity_envelopes=[],
        )


def test_atomic_swap_recovers_prior_backup_when_new_swap_fails(tmp_path, monkeypatch):
    output = tmp_path / "validity_envelopes"
    backup = tmp_path / ".validity_envelopes.backup"
    staged = tmp_path / ".validity_envelopes.staged-test"
    backup.mkdir()
    (backup / "old.txt").write_text("old output", encoding="utf-8")
    staged.mkdir()
    (staged / "new.txt").write_text("new output", encoding="utf-8")
    real_replace = __import__("os").replace

    def fail_new_swap(source, destination):
        if Path(source) == staged and Path(destination) == output:
            raise OSError("simulated swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr("arxiv_kg.validity_dataset.os.replace", fail_new_swap)

    with pytest.raises(OSError, match="simulated swap failure"):
        _replace_directory(staged, output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "old output"
