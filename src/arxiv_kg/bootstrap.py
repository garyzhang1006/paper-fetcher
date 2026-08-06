"""Validated, persistence-free bootstrap reader for curated paper JSONL."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .ids import split_arxiv_version
from .models import PaperRecord


REQUIRED_FIELDS = frozenset(
    {
        "arxiv_id",
        "versioned_id",
        "title",
        "abstract",
        "authors",
        "categories",
        "primary_category",
        "published_at",
        "updated_at",
        "abs_url",
    }
)


class BootstrapDataError(ValueError):
    """A curated-dataset error with source location and recovery context."""


@dataclass(frozen=True)
class BootstrapSummary:
    lines_read: int
    blank_lines: int
    records_parsed: int
    unique_papers: int
    duplicate_versions: int
    superseded_versions: int


@dataclass(frozen=True)
class BootstrapResult:
    papers: tuple[PaperRecord, ...]
    summary: BootstrapSummary

    def __iter__(self) -> Iterator[PaperRecord]:
        return iter(self.papers)

    def __len__(self) -> int:
        return len(self.papers)


def _error(path: Path, line_number: int, message: str) -> BootstrapDataError:
    return BootstrapDataError(f"{path}: line {line_number}: {message}")


def _required_string(
    payload: Mapping[str, Any], field: str, *, path: Path, line_number: int
) -> str:
    value = payload[field]
    if not isinstance(value, str) or not value.strip():
        raise _error(path, line_number, f"{field!r} must be a non-empty string")
    return value.strip()


def _optional_string(
    payload: Mapping[str, Any], field: str, *, path: Path, line_number: int
) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise _error(path, line_number, f"{field!r} must be a string or null")
    return value.strip() or None


def _string_list(
    payload: Mapping[str, Any], field: str, *, path: Path, line_number: int
) -> list[str]:
    value = payload[field]
    if not isinstance(value, list) or not value:
        raise _error(path, line_number, f"{field!r} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise _error(
            path,
            line_number,
            f"{field!r} must contain only non-empty strings",
        )
    return [item.strip() for item in value]


def _timestamp(
    payload: Mapping[str, Any], field: str, *, path: Path, line_number: int
) -> datetime:
    raw = _required_string(payload, field, path=path, line_number=line_number)
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error(
            path,
            line_number,
            f"{field!r} must be an ISO-8601 timestamp with a timezone",
        ) from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise _error(
            path,
            line_number,
            f"{field!r} must include a timezone offset",
        )
    return value


def parse_curated_paper(
    payload: object, *, path: str | Path, line_number: int
) -> PaperRecord:
    """Validate one decoded JSON object and return its canonical paper record."""

    source = Path(path)
    if not isinstance(payload, Mapping):
        raise _error(source, line_number, "expected one JSON object")

    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        raise _error(
            source,
            line_number,
            f"missing required field(s): {', '.join(missing)}",
        )

    raw_id = _required_string(payload, "arxiv_id", path=source, line_number=line_number)
    raw_versioned_id = _required_string(
        payload, "versioned_id", path=source, line_number=line_number
    )
    canonical_id, version = split_arxiv_version(raw_versioned_id)
    supplied_id, _ = split_arxiv_version(raw_id)
    if not canonical_id or supplied_id != canonical_id:
        raise _error(
            source,
            line_number,
            "arxiv_id and versioned_id identify different canonical papers "
            f"({raw_id!r} versus {raw_versioned_id!r})",
        )

    authors = _string_list(payload, "authors", path=source, line_number=line_number)
    categories = _string_list(
        payload, "categories", path=source, line_number=line_number
    )
    primary_category = _required_string(
        payload, "primary_category", path=source, line_number=line_number
    )
    if primary_category not in categories:
        raise _error(
            source,
            line_number,
            "primary_category must also appear in categories",
        )

    published_at = _timestamp(
        payload, "published_at", path=source, line_number=line_number
    )
    updated_at = _timestamp(payload, "updated_at", path=source, line_number=line_number)
    if updated_at < published_at:
        raise _error(source, line_number, "updated_at cannot precede published_at")

    affiliations = payload.get("affiliations", {})
    if not isinstance(affiliations, dict):
        raise _error(source, line_number, "'affiliations' must be an object")

    try:
        return PaperRecord(
            arxiv_id=canonical_id,
            versioned_id=f"{canonical_id}v{version}",
            version=version,
            title=_required_string(payload, "title", path=source, line_number=line_number),
            abstract=_required_string(
                payload, "abstract", path=source, line_number=line_number
            ),
            authors=authors,
            affiliations=affiliations,
            categories=categories,
            primary_category=primary_category,
            published_at=published_at,
            updated_at=updated_at,
            abs_url=_required_string(
                payload, "abs_url", path=source, line_number=line_number
            ),
            pdf_url=_optional_string(
                payload, "pdf_url", path=source, line_number=line_number
            ),
            doi=_optional_string(payload, "doi", path=source, line_number=line_number),
            journal_ref=_optional_string(
                payload, "journal_ref", path=source, line_number=line_number
            ),
            comment=_optional_string(
                payload, "comment", path=source, line_number=line_number
            ),
        )
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        raise _error(
            source,
            line_number,
            f"invalid {location or 'record'}: {first['msg']}",
        ) from exc


def load_curated_papers(path: str | Path) -> BootstrapResult:
    """Read JSONL, retain each paper's newest version, and report scan counts."""

    source = Path(path)
    newest: dict[str, tuple[PaperRecord, int]] = {}
    seen_versions: dict[tuple[str, int], tuple[PaperRecord, int]] = {}
    lines_read = 0
    blank_lines = 0
    records_parsed = 0
    duplicate_versions = 0
    superseded_versions = 0

    try:
        handle = source.open(encoding="utf-8")
    except OSError as exc:
        raise BootstrapDataError(f"{source}: unable to open curated dataset: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            lines_read = line_number
            if not raw_line.strip():
                blank_lines += 1
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise _error(
                    source,
                    line_number,
                    f"invalid JSON at column {exc.colno}: {exc.msg}",
                ) from exc

            paper = parse_curated_paper(
                payload, path=source, line_number=line_number
            )
            records_parsed += 1
            version_key = (paper.arxiv_id, paper.version)
            previous_version = seen_versions.get(version_key)
            if previous_version is not None:
                previous_paper, previous_line = previous_version
                if previous_paper != paper:
                    raise _error(
                        source,
                        line_number,
                        "conflicting records for "
                        f"{paper.versioned_id}; first seen on line {previous_line}",
                    )
                duplicate_versions += 1
                continue
            seen_versions[version_key] = (paper, line_number)

            current = newest.get(paper.arxiv_id)
            if current is None:
                newest[paper.arxiv_id] = (paper, line_number)
                continue
            superseded_versions += 1
            if paper.version > current[0].version:
                newest[paper.arxiv_id] = (paper, line_number)

    papers = tuple(item[0] for item in newest.values())
    return BootstrapResult(
        papers=papers,
        summary=BootstrapSummary(
            lines_read=lines_read,
            blank_lines=blank_lines,
            records_parsed=records_parsed,
            unique_papers=len(papers),
            duplicate_versions=duplicate_versions,
            superseded_versions=superseded_versions,
        ),
    )


def iter_curated_papers(path: str | Path) -> Iterator[PaperRecord]:
    """Stream validated records while retaining only conflict fingerprints."""

    source = Path(path)
    seen_versions: dict[tuple[str, int], tuple[str, int]] = {}
    try:
        handle = source.open(encoding="utf-8")
    except OSError as exc:
        raise BootstrapDataError(f"{source}: unable to open curated dataset: {exc}") from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise _error(
                    source,
                    line_number,
                    f"invalid JSON at column {exc.colno}: {exc.msg}",
                ) from exc
            paper = parse_curated_paper(payload, path=source, line_number=line_number)
            fingerprint = hashlib.sha256(
                paper.model_dump_json().encode("utf-8")
            ).hexdigest()
            key = (paper.arxiv_id, paper.version)
            previous = seen_versions.get(key)
            if previous is not None:
                previous_fingerprint, previous_line = previous
                if previous_fingerprint != fingerprint:
                    raise _error(
                        source,
                        line_number,
                        "conflicting records for "
                        f"{paper.versioned_id}; first seen on line {previous_line}",
                    )
                continue
            seen_versions[key] = (fingerprint, line_number)
            yield paper
