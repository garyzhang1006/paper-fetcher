"""Fetch arXiv submission metadata for the July 3-15, 2026 dataset.

The script uses the official arXiv API directly so the exported files are
reproducible without installing the project package.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"
NS = {"atom": ATOM, "arxiv": ARXIV, "opensearch": OPENSEARCH}

START_DATE = date(2026, 7, 3)
END_DATE = date(2026, 7, 15)
PAGE_SIZE = 2000
REQUEST_DELAY_SECONDS = 3.0
USER_AGENT = "codex-dataset-builder/1.0 (local arxiv dataset export)"
API_URL = "https://export.arxiv.org/api/query"
TARGET_PAPER_COUNT = 9000
LOW_QUALITY_MARKERS = re.compile(r"\b(withdrawn|retracted)\b", re.IGNORECASE)
EXCLUDED_PRIMARY_CATEGORY_PREFIXES = ("astro-ph.",)


@dataclass(frozen=True)
class DayWindow:
    day: date

    @property
    def query(self) -> str:
        stamp = self.day.strftime("%Y%m%d")
        return f"submittedDate:[{stamp}0000 TO {stamp}2359]"


def iter_days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def text(element: ET.Element, path: str) -> str | None:
    found = element.find(path, NS)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split())


def arxiv_extension(entry: ET.Element, tag: str) -> str | None:
    return text(entry, f"arxiv:{tag}")


def links(entry: ET.Element) -> tuple[str | None, str | None]:
    abs_url = None
    pdf_url = None
    for link in entry.findall("atom:link", NS):
        href = link.attrib.get("href")
        if link.attrib.get("rel") == "alternate":
            abs_url = href
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = href
    return abs_url, pdf_url


def parse_entry(entry: ET.Element, submitted_date: str) -> dict[str, object]:
    entry_id = text(entry, "atom:id") or ""
    versioned_id = entry_id.rsplit("/", 1)[-1]
    abs_url, pdf_url = links(entry)
    categories = [
        category.attrib["term"]
        for category in entry.findall("atom:category", NS)
        if "term" in category.attrib
    ]
    primary = entry.find("arxiv:primary_category", NS)
    return {
        "arxiv_id": versioned_id.split("v", 1)[0],
        "versioned_id": versioned_id,
        "title": text(entry, "atom:title"),
        "abstract": text(entry, "atom:summary"),
        "authors": [
            text(author, "atom:name")
            for author in entry.findall("atom:author", NS)
            if text(author, "atom:name")
        ],
        "categories": categories,
        "primary_category": primary.attrib.get("term") if primary is not None else None,
        "published_at": text(entry, "atom:published"),
        "updated_at": text(entry, "atom:updated"),
        "submitted_date": submitted_date,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
        "doi": arxiv_extension(entry, "doi"),
        "journal_ref": arxiv_extension(entry, "journal_ref"),
        "comment": arxiv_extension(entry, "comment"),
    }


def fetch_feed(query: str, start: int, max_results: int) -> ET.Element:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",
    }
    request = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=90) as response:
        return ET.fromstring(response.read())


def total_results(feed: ET.Element) -> int:
    value = feed.findtext("opensearch:totalResults", namespaces=NS)
    if value is None:
        raise RuntimeError("arXiv response omitted opensearch:totalResults")
    return int(value)


def feed_entries(feed: ET.Element) -> list[ET.Element]:
    entries = feed.findall("atom:entry", NS)
    if len(entries) == 1 and text(entries[0], "atom:title") == "Error":
        summary = text(entries[0], "atom:summary")
        raise RuntimeError(f"arXiv API returned error entry: {summary}")
    return entries


def fetch_day(window: DayWindow) -> tuple[int, list[dict[str, object]]]:
    first_feed = fetch_feed(window.query, start=0, max_results=PAGE_SIZE)
    total = total_results(first_feed)
    papers = [
        parse_entry(entry, window.day.isoformat())
        for entry in feed_entries(first_feed)
    ]

    next_start = PAGE_SIZE
    while next_start < total:
        time.sleep(REQUEST_DELAY_SECONDS)
        feed = fetch_feed(window.query, start=next_start, max_results=PAGE_SIZE)
        papers.extend(
            parse_entry(entry, window.day.isoformat())
            for entry in feed_entries(feed)
        )
        next_start += PAGE_SIZE

    if len(papers) != total:
        raise RuntimeError(
            f"{window.day.isoformat()} expected {total} papers, got {len(papers)}"
        )
    return total, papers


def quality_score(paper: dict[str, object]) -> tuple[int, int, int, int, str]:
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    searchable_text = f"{title} {abstract}"
    title_word_count = len(title.split())
    metadata_fields = ("doi", "journal_ref", "comment")
    metadata_count = sum(bool(paper.get(field)) for field in metadata_fields)
    category_count = len(paper.get("categories") or [])
    stable_tiebreaker = hashlib.sha256(
        str(paper.get("versioned_id") or "").encode("utf-8")
    ).hexdigest()
    return (
        not bool(LOW_QUALITY_MARKERS.search(searchable_text)),
        min(len(abstract.split()), 250),
        metadata_count,
        int(4 <= title_word_count <= 30) + min(category_count, 3),
        stable_tiebreaker,
    )


def category_quotas(
    category_sizes: dict[str, int], target: int
) -> dict[str, int]:
    total = sum(category_sizes.values())
    if target >= total:
        return dict(category_sizes)
    if target < len(category_sizes):
        raise ValueError("target must retain at least one paper per category")

    quotas = {category: 1 for category in category_sizes}
    remaining_target = target - len(quotas)
    remaining_population = total - len(quotas)
    remainders: list[tuple[float, str]] = []
    for category, size in category_sizes.items():
        share = remaining_target * (size - 1) / remaining_population
        extra = int(share)
        quotas[category] += extra
        remainders.append((share - extra, category))

    unassigned = target - sum(quotas.values())
    for _, category in sorted(remainders, reverse=True)[:unassigned]:
        quotas[category] += 1
    return quotas


def curate_papers(
    papers: list[dict[str, object]], target: int = TARGET_PAPER_COUNT
) -> list[dict[str, object]]:
    if target >= len(papers):
        return papers

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for paper in papers:
        category = str(paper.get("primary_category") or "(missing)")
        by_category[category].append(paper)

    quotas = category_quotas(
        {category: len(group) for category, group in by_category.items()},
        target,
    )
    retained_ids: set[str] = set()
    for category, group in by_category.items():
        ranked = sorted(group, key=quality_score, reverse=True)
        retained_ids.update(
            str(paper["versioned_id"])
            for paper in ranked[: quotas[category]]
        )

    return [
        paper
        for paper in papers
        if str(paper["versioned_id"]) in retained_ids
    ]


def exclude_primary_categories(papers: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove categories excluded from the final curated dataset."""
    return [
        paper
        for paper in papers
        if not str(paper.get("primary_category") or "").startswith(
            EXCLUDED_PRIMARY_CATEGORY_PREFIXES
        )
    ]


def daily_counts(papers: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(str(paper["submitted_date"]) for paper in papers)
    return [
        {"date": day.isoformat(), "paper_count": counts[day.isoformat()]}
        for day in iter_days(START_DATE, END_DATE)
    ]


def write_outputs(
    output_dir: Path,
    counts: list[dict[str, object]],
    papers: list[dict[str, object]],
    source_paper_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "papers.jsonl").open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper, ensure_ascii=False, sort_keys=True) + "\n")

    with (output_dir / "daily_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "paper_count"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(counts)

    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source": API_URL,
                "api_manual": "https://info.arxiv.org/help/api/user-manual.html",
                "date_range": {
                    "start": START_DATE.isoformat(),
                    "end": END_DATE.isoformat(),
                    "inclusive": True,
                    "timezone": "UTC",
                },
                "query_template": "submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]",
                "sort": {"sortBy": "submittedDate", "sortOrder": "ascending"},
                "curation": {
                    "method": "metadata-quality ranking with proportional primary-category quotas, followed by Astrophysics exclusion",
                    "target_paper_count": TARGET_PAPER_COUNT,
                    "post_curation_paper_count": len(papers),
                    "removed_paper_count": source_paper_count - len(papers),
                    "preserves_every_primary_category": False,
                    "excluded_primary_category_group": "Astrophysics",
                    "excluded_primary_category_prefix": "astro-ph.",
                },
                "source_paper_count": source_paper_count,
                "paper_count": len(papers),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    papers: list[dict[str, object]] = []
    for day in iter_days(START_DATE, END_DATE):
        if papers:
            time.sleep(REQUEST_DELAY_SECONDS)
        total, day_papers = fetch_day(DayWindow(day))
        papers.extend(day_papers)
        print(f"{day.isoformat()} {total}")

    source_paper_count = len(papers)
    papers = exclude_primary_categories(curate_papers(papers))
    write_outputs(
        Path(__file__).resolve().parent,
        daily_counts(papers),
        papers,
        source_paper_count,
    )
    print(f"source {source_paper_count}")
    print(f"retained {len(papers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
