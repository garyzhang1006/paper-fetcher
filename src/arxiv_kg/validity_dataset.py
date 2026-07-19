"""Batch abstract-level validity extraction for JSONL paper datasets."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import TextIO

from .models import PaperValidityRecord
from .validity import (
    VALIDITY_EXTRACTOR_NAME,
    VALIDITY_EXTRACTOR_VERSION,
    extract_validity_envelopes,
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OUTPUT_README = """# Abstract-level experimental validity envelopes

Each date-sharded JSONL file contains one record for every source paper from
that UTC submission date. Join records to `../papers.jsonl` with `arxiv_id` and
`source_versioned_id`.

An envelope preserves an explicit result claim plus any comparator, evaluation
context, metric, reported numeric value, effect size, uncertainty, or condition found in the claim.
Explicit limitations elsewhere in the abstract are labeled as paper-level
boundaries. Empty fields mean the abstract did not state that detail.
`no_supported_claim` means the conservative rules found no explicit result
claim; it does not mean the paper has no results.

Confidence measures how many supported detail types were extracted. It is not a
calibrated probability and does not estimate whether the scientific claim is true.

These records do not claim table, figure, page, seed, or compute-budget evidence
because the repository does not contain the 8,406 full-text PDFs. See
`manifest.json` for counts, hashes, extractor version, and source provenance.

Regenerate from the repository root:

```bash
paper-fetcher-validity --expected-count 8406
```
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(record: dict[str, object], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a nonempty string")
    return value.strip()


def _replace_directory(staged: Path, output: Path) -> None:
    backup = output.with_name(f".{output.name}.backup")
    if backup.exists():
        if output.exists():
            shutil.rmtree(backup)
        else:
            os.replace(backup, output)
    if output.exists():
        os.replace(output, backup)
    try:
        os.replace(staged, output)
    except Exception:
        if backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def write_validity_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    expected_count: int = 8406,
) -> dict[str, object]:
    """Write date-sharded records, replacing prior output only after validation."""

    source = Path(source_path)
    output = Path(output_dir)
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"paper dataset not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}.staged-", dir=output.parent))
    handles: dict[str, TextIO] = {}
    shard_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    processed = 0
    papers_with_envelopes = 0
    envelope_count = 0
    try:
        with source.open(encoding="utf-8") as source_handle:
            for line_number, line in enumerate(source_handle, start=1):
                try:
                    source_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(source_record, dict):
                    raise ValueError(f"line {line_number}: paper record must be an object")
                arxiv_id = _required_string(source_record, "arxiv_id", line_number)
                if arxiv_id in seen_ids:
                    raise ValueError(f"line {line_number}: duplicate arxiv_id {arxiv_id}")
                seen_ids.add(arxiv_id)
                versioned_id = _required_string(
                    source_record, "versioned_id", line_number
                )
                submitted_date = _required_string(
                    source_record, "submitted_date", line_number
                )
                try:
                    date.fromisoformat(submitted_date)
                except ValueError:
                    valid_date = False
                else:
                    valid_date = DATE_RE.fullmatch(submitted_date) is not None
                if not valid_date:
                    raise ValueError(
                        f"line {line_number}: submitted_date must use YYYY-MM-DD"
                    )
                _required_string(source_record, "title", line_number)
                abstract = _required_string(source_record, "abstract", line_number)
                try:
                    envelopes = extract_validity_envelopes(abstract)
                except Exception as exc:
                    raise RuntimeError(
                        f"line {line_number} ({arxiv_id}): validity extraction failed: {exc}"
                    ) from exc
                status = "extracted" if envelopes else "no_supported_claim"
                output_record = PaperValidityRecord(
                    arxiv_id=arxiv_id,
                    source_versioned_id=versioned_id,
                    source_scope="abstract",
                    extractor=VALIDITY_EXTRACTOR_NAME,
                    extractor_version=VALIDITY_EXTRACTOR_VERSION,
                    status=status,
                    validity_envelopes=envelopes,
                )
                shard_name = f"{submitted_date}.jsonl"
                if shard_name not in handles:
                    handles[shard_name] = (staged / shard_name).open(
                        "w", encoding="utf-8"
                    )
                handles[shard_name].write(output_record.model_dump_json() + "\n")
                shard_counts[shard_name] += 1
                processed += 1
                papers_with_envelopes += bool(envelopes)
                envelope_count += len(envelopes)
        if processed != expected_count:
            raise ValueError(f"expected {expected_count} papers, got {processed}")
    except Exception:
        for handle in handles.values():
            handle.close()
        if staged.exists():
            shutil.rmtree(staged)
        raise
    else:
        for handle in handles.values():
            handle.close()

    try:
        shards = [
            {
                "file": shard_name,
                "paper_count": shard_counts[shard_name],
                "sha256": _sha256(staged / shard_name),
            }
            for shard_name in sorted(shard_counts)
        ]
        manifest: dict[str, object] = {
            "schema_version": 1,
            "source_file": source.name,
            "source_sha256": _sha256(source),
            "source_scope": "abstract",
            "extractor": VALIDITY_EXTRACTOR_NAME,
            "extractor_version": VALIDITY_EXTRACTOR_VERSION,
            "processed_paper_count": processed,
            "paper_with_envelopes_count": papers_with_envelopes,
            "paper_without_supported_claim_count": processed
            - papers_with_envelopes,
            "validity_envelope_count": envelope_count,
            "shards": shards,
        }
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staged / "README.md").write_text(OUTPUT_README, encoding="utf-8")
        lock_key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
        lock_path = Path(tempfile.gettempdir()) / f"arxiv-kg-validity-{lock_key}.lock"
        with lock_path.open("w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            _replace_directory(staged, output)
        return manifest
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract abstract-level experimental validity envelopes"
    )
    parser.add_argument("--input", type=Path, default=Path("dataset/papers.jsonl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/validity_envelopes"),
    )
    parser.add_argument("--expected-count", type=int, default=8406)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = write_validity_dataset(
        args.input,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
