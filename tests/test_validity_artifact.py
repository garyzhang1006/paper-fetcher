import hashlib
import json
from pathlib import Path

from arxiv_kg.models import PaperValidityRecord


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_committed_validity_artifact_matches_all_source_papers():
    root = Path(__file__).parents[1]
    source_path = root / "dataset/papers.jsonl"
    output_dir = root / "dataset/validity_envelopes"
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["processed_paper_count"] == 7751
    assert manifest["source_scope"] == "abstract"
    assert manifest["source_sha256"] == sha256(source_path)

    processed = 0
    papers_with_envelopes = 0
    envelope_count = 0
    with source_path.open(encoding="utf-8") as source_handle:
        source_records = (json.loads(line) for line in source_handle)
        for shard in manifest["shards"]:
            shard_path = output_dir / shard["file"]
            assert shard["sha256"] == sha256(shard_path)
            shard_count = 0
            with shard_path.open(encoding="utf-8") as output_handle:
                for output_line in output_handle:
                    source = next(source_records)
                    record = PaperValidityRecord.model_validate_json(output_line)
                    assert record.arxiv_id == source["arxiv_id"]
                    assert record.source_versioned_id == source["versioned_id"]
                    abstract = " ".join(source["abstract"].split()).casefold()
                    for envelope in record.validity_envelopes:
                        assert envelope.claim.casefold() in abstract
                        assert envelope.evidence.statement == envelope.claim
                    processed += 1
                    shard_count += 1
                    papers_with_envelopes += bool(record.validity_envelopes)
                    envelope_count += len(record.validity_envelopes)
            assert shard_count == shard["paper_count"]
        assert next(source_records, None) is None

    assert processed == manifest["processed_paper_count"]
    assert papers_with_envelopes == manifest["paper_with_envelopes_count"]
    assert envelope_count == manifest["validity_envelope_count"]
