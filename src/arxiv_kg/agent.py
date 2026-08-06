"""One bounded daily agent for collection, extraction, graphing, and analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .bootstrap import iter_curated_papers
from .db import Database
from .extractor import RuleBasedFeatureExtractor
from .graph import GraphDocument, GraphSnapshot, build_graph
from .graph_analytics import compute_hubs, compute_trends, find_clusters
from .models import PaperRecord, SourceItem

LOGGER = logging.getLogger(__name__)
CATEGORY_RE = re.compile(r"^[A-Za-z]+(?:[.-][A-Za-z]+)+$")
SOURCE_NODE_TYPES = {
    "research_blog": "blog_post",
    "social_media": "social_post",
}
REPORT_TITLE_MARKERS = re.compile(
    r"\b(report|study|paper|survey|benchmark|evaluation|analysis)\b",
    flags=re.IGNORECASE,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def category(value: str) -> str:
    if not CATEGORY_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"invalid arXiv category: {value!r}")
    return value


def _fetch_report_dict(report: object) -> dict[str, object]:
    values = asdict(report)
    for key, value in values.items():
        if isinstance(value, datetime):
            values[key] = value.isoformat()
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete Project 1 research knowledge-graph agent"
    )
    parser.add_argument("--db", default="data/arxiv_kg.sqlite3")
    parser.add_argument("--dataset", default="dataset/papers.jsonl")
    parser.add_argument("--sources", default="config/sources.json")
    parser.add_argument("--output-dir", default="output/knowledge_graph")
    parser.add_argument(
        "--category",
        dest="categories",
        action="append",
        type=category,
        help="arXiv category; repeat for multiple categories (default: cs.LG)",
    )
    parser.add_argument("--max-results", type=positive_int, default=200)
    parser.add_argument("--first-run-lookback-hours", type=positive_int, default=24)
    parser.add_argument("--overlap-hours", type=positive_int, default=24)
    parser.add_argument("--revision-max-results", type=positive_int, default=200)
    parser.add_argument(
        "--revision-first-run-lookback-hours", type=positive_int, default=24
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="skip all network collection but still bootstrap, extract, and rebuild",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="do not import the curated JSONL before collection",
    )
    parser.add_argument(
        "--no-feeds",
        action="store_true",
        help="skip configured RSS and Atom sources",
    )
    parser.add_argument(
        "--no-arxiv",
        action="store_true",
        help="skip arXiv collection while retaining public feed collection",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.categories = list(dict.fromkeys(args.categories or ["cs.LG"]))
    return args


def _extract_missing_features(
    db: Database, extractor: RuleBasedFeatureExtractor
) -> int:
    papers = db.iter_papers_needing_features(
        extractor.name,
        extractor.version,
        extractor.prompt_version,
    )

    def records() -> Iterable[tuple[PaperRecord, Any]]:
        for paper in papers:
            yield paper, extractor.extract(
                title=paper.title,
                abstract=paper.abstract,
                paper_text="",
            )

    return db.save_features_batch(
        records(),
        extractor=extractor.name,
        extractor_version=extractor.version,
        prompt_version=extractor.prompt_version,
    )


def _paper_graph_documents(db: Database) -> Iterable[GraphDocument]:
    for paper, stored in db.iter_papers_with_features():
        yield GraphDocument(
            document_id=paper.arxiv_id,
            node_type="paper",
            source_id="arxiv",
            source_version=paper.versioned_id,
            title=paper.title,
            url=paper.abs_url,
            published_at=paper.published_at,
            categories=paper.categories,
            features=stored.features,
            extractor=stored.extractor,
            extractor_version=stored.extractor_version,
            prompt_version=stored.prompt_version,
        )


def _source_graph_document(
    item: SourceItem, extractor: RuleBasedFeatureExtractor
) -> GraphDocument:
    abstract = item.summary or item.content_text
    features = extractor.extract(
        title=item.title or item.source_name,
        abstract=abstract,
        paper_text=item.content_text,
    )
    if item.source_kind == "research_report":
        node_type = "report" if REPORT_TITLE_MARKERS.search(item.title or "") else "blog_post"
    else:
        node_type = SOURCE_NODE_TYPES[item.source_kind]
    return GraphDocument(
        document_id=item.item_id,
        node_type=node_type,
        source_id=item.source_id,
        source_version=item.content_sha256,
        title=item.title or item.source_name,
        url=item.canonical_url,
        published_at=item.published_at or item.updated_at or item.retrieved_at,
        source_topics=item.source_topics,
        features=features,
        extractor=extractor.name,
        extractor_version=extractor.version,
        prompt_version=extractor.prompt_version,
    )


def _snapshot_build_id(snapshot: GraphSnapshot) -> str:
    encoded = snapshot.model_dump_json(exclude_none=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _analysis(snapshot: GraphSnapshot) -> dict[str, Any]:
    hubs = compute_hubs(snapshot)[:200]
    clusters = sorted(
        find_clusters(snapshot),
        key=lambda item: (-len(item.member_ids), item.cluster_id),
    )[:200]
    trends = compute_trends(snapshot)
    return {
        "algorithm_version": snapshot.algorithm_version,
        "related_threshold": snapshot.related_threshold,
        "related_limit": snapshot.related_limit,
        "common_concept_ratio": snapshot.common_concept_ratio,
        "suppressed_concept_ids": snapshot.suppressed_concept_ids,
        "hubs": [item.model_dump(mode="json") for item in hubs],
        "clusters": [item.model_dump(mode="json") for item in clusters],
        "trends": trends.model_dump(mode="json"),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_agent(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    db = Database(Path(args.db))
    extractor = RuleBasedFeatureExtractor()
    bootstrap_report: dict[str, int] | None = None
    arxiv_report: dict[str, object] | None = None
    source_reports: list[dict[str, Any]] = []
    collection_errors: list[str] = []

    if not args.no_bootstrap:
        bootstrap_report = db.upsert_papers(iter_curated_papers(args.dataset))

    if not args.offline and not args.no_arxiv:
        from .fetcher import fetch_recent_papers

        try:
            fetched = fetch_recent_papers(
                db,
                categories=args.categories,
                max_results=args.max_results,
                first_run_lookback_hours=args.first_run_lookback_hours,
                overlap_hours=args.overlap_hours,
                revision_max_results=args.revision_max_results,
                revision_first_run_lookback_hours=args.revision_first_run_lookback_hours,
            )
            arxiv_report = _fetch_report_dict(fetched)
        except Exception as exc:
            LOGGER.exception("arXiv ingestion failed")
            collection_errors.append(f"arxiv: {exc}")

    if not args.offline and not args.no_feeds:
        from .feeds import fetch_feed, load_feed_sources

        for source in load_feed_sources(args.sources):
            if not source.enabled:
                continue
            try:
                items = fetch_feed(source)
                checkpoint = max(item.retrieved_at for item in items).isoformat()
                counts = db.upsert_source_items(
                    items,
                    checkpoint_key=f"feed:{source.source_id}:last_success",
                    checkpoint_value=checkpoint,
                )
                source_reports.append(
                    {"source_id": source.source_id, "received": len(items), **counts}
                )
            except Exception as exc:
                LOGGER.exception("Feed ingestion failed for %s", source.source_id)
                message = f"{source.source_id}: {exc}"
                collection_errors.append(message)
                source_reports.append(
                    {"source_id": source.source_id, "status": "error", "error": str(exc)}
                )

    extracted = _extract_missing_features(db, extractor)
    documents = [
        *_paper_graph_documents(db),
        *(
            _source_graph_document(item, extractor)
            for item in db.iter_source_items()
        ),
    ]
    snapshot = build_graph(documents)
    analysis = _analysis(snapshot)
    build_id = _snapshot_build_id(snapshot)
    graph_report = db.replace_graph(
        build_id=build_id,
        document_count=len(documents),
        nodes=snapshot.nodes,
        edges=snapshot.edges,
        analysis=analysis,
    )
    completed_at = datetime.now(UTC)
    report: dict[str, Any] = {
        "status": "degraded" if collection_errors else "ok",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "offline": args.offline,
        "bootstrap": bootstrap_report,
        "arxiv": arxiv_report,
        "feeds": source_reports,
        "collection_errors": collection_errors,
        "features_extracted": extracted,
        "graph": graph_report,
        "database_counts": db.counts(),
        "stored_sources": db.source_counts(),
        "trend_status": analysis["trends"]["status"],
    }
    output_dir = Path(args.output_dir)
    _write_json_atomic(output_dir / "summary.json", report)
    _write_json_atomic(output_dir / "trends.json", analysis["trends"])
    _write_json_atomic(
        output_dir / "hubs.json",
        {"build_id": build_id, "hubs": analysis["hubs"][:50]},
    )
    _write_json_atomic(
        output_dir / "clusters.json",
        {"build_id": build_id, "clusters": analysis["clusters"][:50]},
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run_agent(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
