"""Noninteractive entry point for one bounded arXiv update."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .db import Database
from .fetcher import ARXIV_CATEGORY_RE, FetchReport, fetch_recent_papers


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def category(value: str) -> str:
    if not ARXIV_CATEGORY_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"invalid arXiv category: {value!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded arXiv metadata update"
    )
    parser.add_argument("--db", default="data/arxiv_kg.sqlite3")
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        type=category,
        help="arXiv category; repeat for multiple categories (default: cs.LG)",
    )
    parser.add_argument("--max-results", type=positive_int, default=200)
    parser.add_argument(
        "--first-run-lookback-hours", type=positive_int, default=24
    )
    parser.add_argument("--overlap-hours", type=positive_int, default=24)
    parser.add_argument("--revision-max-results", type=positive_int, default=200)
    parser.add_argument(
        "--revision-first-run-lookback-hours", type=positive_int, default=24
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.categories = list(dict.fromkeys(args.categories or ["cs.LG"]))
    return args


def report_dict(report: FetchReport) -> dict[str, object]:
    values = asdict(report)
    for key, value in values.items():
        if isinstance(value, datetime):
            values[key] = value.isoformat()
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = fetch_recent_papers(
        Database(Path(args.db)),
        categories=args.categories,
        max_results=args.max_results,
        first_run_lookback_hours=args.first_run_lookback_hours,
        overlap_hours=args.overlap_hours,
        revision_max_results=args.revision_max_results,
        revision_first_run_lookback_hours=(
            args.revision_first_run_lookback_hours
        ),
    )
    print(json.dumps(report_dict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
