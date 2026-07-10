"""Tutorial 1: query arXiv and idempotently store paper metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import arxiv
import requests

from .db import Database
from .ids import split_arxiv_version
from .models import PaperRecord

LOGGER = logging.getLogger(__name__)
FETCH_STATE_KEY = "last_successful_arxiv_fetch_utc"
UPDATE_STATE_KEY = "last_successful_arxiv_revision_scan_utc"
ARXIV_CATEGORY_RE = re.compile(r"^[A-Za-z]+(?:[.-][A-Za-z]+)+$")


@dataclass(frozen=True)
class FetchReport:
    start_utc: datetime
    revision_start_utc: datetime | None
    end_utc: datetime
    new_query_received: int
    revision_query_received: int
    unique_results_processed: int
    inserted: int
    updated: int
    unchanged: int


def category_state_key(base_key: str, categories: list[str]) -> str:
    """Scope a checkpoint to an order-independent category set."""

    category_set = ",".join(sorted(set(categories)))
    return f"{base_key}:{category_set}"


def _category_part(categories: list[str]) -> str:
    if not categories:
        raise ValueError("At least one arXiv category is required")
    return " OR ".join(f"cat:{category}" for category in categories)


def build_category_query(
    categories: list[str], start_utc: datetime, end_utc: datetime
) -> str:
    """Build a new-submission query using arXiv's submittedDate filter."""

    start_utc = start_utc.astimezone(UTC)
    end_utc = end_utc.astimezone(UTC)
    date_part = (
        "submittedDate:["
        f"{start_utc:%Y%m%d%H%M} TO {end_utc:%Y%m%d%H%M}]"
    )
    return f"(({_category_part(categories)}) AND {date_part})"


def build_category_only_query(categories: list[str]) -> str:
    """Build a query used for a bounded last-updated-date revision scan."""

    return f"({_category_part(categories)})"


def result_to_paper(result: arxiv.Result) -> PaperRecord:
    versioned_id = result.get_short_id()
    arxiv_id, version = split_arxiv_version(versioned_id)
    affiliations: dict[str, list[str]] = {}
    for author in result.authors:
        values = list(getattr(author, "affiliation", []) or [])
        if values:
            affiliations[author.name] = values
    return PaperRecord(
        arxiv_id=arxiv_id,
        versioned_id=versioned_id,
        version=version,
        title=" ".join(result.title.split()),
        abstract=" ".join(result.summary.split()),
        authors=[author.name for author in result.authors],
        affiliations=affiliations,
        categories=list(result.categories),
        primary_category=result.primary_category,
        published_at=result.published,
        updated_at=result.updated,
        abs_url=result.entry_id,
        pdf_url=result.pdf_url,
        doi=result.doi,
        journal_ref=result.journal_ref,
        comment=result.comment,
    )


def fetch_recent_papers(
    db: Database,
    *,
    categories: list[str],
    max_results: int = 300,
    first_run_lookback_hours: int = 168,
    overlap_hours: int = 24,
    scan_revisions: bool = True,
    revision_max_results: int = 300,
    revision_first_run_lookback_hours: int = 168,
    now: datetime | None = None,
) -> FetchReport:
    """Fetch new submissions and optionally scan recently updated papers.

    The overlap intentionally fetches some records again. Database upserts make
    that safe and protect against late announcements or a previous partial run.
    A checkpoint is written only after all requested queries finish safely.

    The submitted-date query finds new papers. The second query is sorted by
    last-updated date and stops at a cutoff, allowing older papers with new
    versions to be refreshed as well.
    """

    if max_results < 1 or revision_max_results < 1:
        raise ValueError("Result limits must be positive")

    end_utc = (now or datetime.now(UTC)).astimezone(UTC)
    fetch_state_key = category_state_key(FETCH_STATE_KEY, categories)
    update_state_key = category_state_key(UPDATE_STATE_KEY, categories)
    previous = db.get_state(fetch_state_key)
    if previous:
        start_utc = datetime.fromisoformat(previous).astimezone(UTC) - timedelta(
            hours=overlap_hours
        )
    else:
        start_utc = end_utc - timedelta(hours=first_run_lookback_hours)

    revision_start_utc: datetime | None = None
    if scan_revisions:
        previous_revision = db.get_state(update_state_key)
        if previous_revision:
            revision_start_utc = datetime.fromisoformat(previous_revision).astimezone(
                UTC
            ) - timedelta(hours=overlap_hours)
        else:
            revision_start_utc = end_utc - timedelta(
                hours=revision_first_run_lookback_hours
            )

    client = arxiv.Client(
        page_size=min(max(max_results + 1, revision_max_results + 1), 100),
        delay_seconds=3.0,
        num_retries=3,
    )

    status_counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    seen_versioned_ids: set[str] = set()
    unique_processed = 0

    query = build_category_query(categories, start_utc, end_utc)
    LOGGER.info("arXiv new-submission query: %s", query)
    search = arxiv.Search(
        query=query,
        max_results=max_results + 1,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )

    new_received = 0
    new_query_saturated = False
    for result in client.results(search):
        new_received += 1
        if new_received > max_results:
            new_query_saturated = True
            break
        versioned_id = result.get_short_id()
        seen_versioned_ids.add(versioned_id)
        status = db.upsert_paper(result_to_paper(result))
        status_counts[status] += 1
        unique_processed += 1

    # Refuse to move the checkpoint when the configured cap may have truncated
    # the interval. The already inserted rows are harmless; the next run will
    # revisit them because the checkpoint was not advanced.
    if new_query_saturated:
        raise RuntimeError(
            "The new-paper query reached --max-results. Increase the limit or "
            "reduce the lookback window; the checkpoint was not advanced."
        )

    revision_received = 0
    if scan_revisions and revision_start_utc is not None:
        revision_query = build_category_only_query(categories)
        LOGGER.info("arXiv revision query: %s", revision_query)
        revision_search = arxiv.Search(
            query=revision_query,
            max_results=revision_max_results + 1,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        reached_cutoff = False
        revision_query_saturated = False
        for result in client.results(revision_search):
            revision_received += 1
            if result.updated.astimezone(UTC) < revision_start_utc:
                reached_cutoff = True
                break
            if revision_received > revision_max_results:
                revision_query_saturated = True
                break
            versioned_id = result.get_short_id()
            if versioned_id in seen_versioned_ids:
                continue
            seen_versioned_ids.add(versioned_id)
            status = db.upsert_paper(result_to_paper(result))
            status_counts[status] += 1
            unique_processed += 1

        if revision_query_saturated and not reached_cutoff:
            raise RuntimeError(
                "The revision scan reached --revision-max-results before its "
                "time cutoff. Increase the limit; checkpoints were not advanced."
            )

    checkpoints = {fetch_state_key: end_utc.isoformat()}
    if scan_revisions:
        checkpoints[update_state_key] = end_utc.isoformat()
    db.set_states(checkpoints)

    return FetchReport(
        start_utc=start_utc,
        revision_start_utc=revision_start_utc,
        end_utc=end_utc,
        new_query_received=new_received,
        revision_query_received=revision_received,
        unique_results_processed=unique_processed,
        inserted=status_counts["inserted"],
        updated=status_counts["updated"],
        unchanged=status_counts["unchanged"],
    )


def safe_pdf_filename(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_") + ".pdf"


def download_pdf(
    paper: PaperRecord,
    *,
    destination_dir: str | Path,
    user_agent: str,
    timeout_seconds: int = 90,
) -> Path:
    """Download one PDF atomically. This does not make the PDF public."""

    if not paper.pdf_url:
        raise ValueError(f"Paper {paper.arxiv_id} has no PDF URL")
    parsed = urlparse(paper.pdf_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc.endswith(
        "arxiv.org"
    ):
        raise ValueError("Refusing an unexpected PDF host")

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / safe_pdf_filename(paper.arxiv_id)
    temporary_path = final_path.with_suffix(".pdf.part")

    with requests.get(
        paper.pdf_url,
        headers={"User-Agent": user_agent},
        timeout=timeout_seconds,
        stream=True,
    ) as response:
        response.raise_for_status()
        with temporary_path.open("wb") as output:
            for block in response.iter_content(chunk_size=1024 * 128):
                if block:
                    output.write(block)

    with temporary_path.open("rb") as check:
        if check.read(4) != b"%PDF":
            temporary_path.unlink(missing_ok=True)
            raise ValueError("Downloaded file does not begin with a PDF header")
    temporary_path.replace(final_path)
    return final_path
