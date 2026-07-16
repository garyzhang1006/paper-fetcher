"""Fetch arXiv submission metadata for the July 3-15, 2026 dataset.

The script uses the official arXiv API directly so the exported files are
reproducible without installing the project package.
"""

from __future__ import annotations

import csv
import json
import time
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


def write_outputs(output_dir: Path, counts: list[dict[str, object]], papers: list[dict[str, object]]) -> None:
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
    counts: list[dict[str, object]] = []
    papers: list[dict[str, object]] = []
    for day in iter_days(START_DATE, END_DATE):
        if counts:
            time.sleep(REQUEST_DELAY_SECONDS)
        total, day_papers = fetch_day(DayWindow(day))
        counts.append({"date": day.isoformat(), "paper_count": total})
        papers.extend(day_papers)
        print(f"{day.isoformat()} {total}")

    write_outputs(Path(__file__).resolve().parent, counts, papers)
    print(f"total {len(papers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
