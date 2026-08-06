"""Local web UI for fetching and browsing arXiv papers."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import Database
from .fetcher import ARXIV_CATEGORY_RE, fetch_recent_papers
from .models import PaperRecord

LOGGER = logging.getLogger(__name__)
MAX_BODY_BYTES = 16 * 1024
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def paper_to_dict(paper: PaperRecord) -> dict[str, Any]:
    return {
        "arxiv_id": paper.arxiv_id,
        "versioned_id": paper.versioned_id,
        "version": paper.version,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors,
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "published_at": paper.published_at.isoformat(),
        "updated_at": paper.updated_at.isoformat(),
        "abs_url": paper.abs_url,
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "journal_ref": paper.journal_ref,
        "comment": paper.comment,
    }


def validate_fetch_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("Choose at least one arXiv category")
    if len(categories) > 12:
        raise ValueError("Choose no more than 12 categories")

    cleaned_categories: list[str] = []
    for category in categories:
        if not isinstance(category, str) or not ARXIV_CATEGORY_RE.fullmatch(category):
            raise ValueError(f"Invalid arXiv category: {category!r}")
        if category not in cleaned_categories:
            cleaned_categories.append(category)

    max_results = payload.get("max_results", 200)
    lookback_hours = payload.get("lookback_hours", 24)
    scan_revisions = payload.get("scan_revisions", True)
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise ValueError("max_results must be an integer")
    if not 1 <= max_results <= 500:
        raise ValueError("max_results must be between 1 and 500")
    if isinstance(lookback_hours, bool) or not isinstance(lookback_hours, int):
        raise ValueError("lookback_hours must be an integer")
    if not 1 <= lookback_hours <= 720:
        raise ValueError("lookback_hours must be between 1 and 720")
    if not isinstance(scan_revisions, bool):
        raise ValueError("scan_revisions must be true or false")

    return {
        "categories": cleaned_categories,
        "max_results": max_results,
        "first_run_lookback_hours": lookback_hours,
        "revision_max_results": max_results,
        "revision_first_run_lookback_hours": lookback_hours,
        "scan_revisions": scan_revisions,
    }


class PaperFetcherServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], db_path: str | Path):
        super().__init__(address, PaperFetcherHandler)
        self.database = Database(db_path)


class PaperFetcherHandler(BaseHTTPRequestHandler):
    server: PaperFetcherServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in STATIC_FILES:
            self._serve_static(parsed.path)
            return
        if parsed.path == "/api/stats":
            self._send_json(HTTPStatus.OK, self.server.database.counts())
            return
        if parsed.path == "/api/papers":
            self._list_papers(parse_qs(parsed.query))
            return
        if parsed.path == "/api/graph":
            self._get_graph(parse_qs(parsed.query))
            return
        if parsed.path == "/api/placement":
            self._get_placement(parse_qs(parsed.query))
            return
        if parsed.path in {"/api/hubs", "/api/clusters", "/api/trends"}:
            key = parsed.path.removeprefix("/api/")
            self._get_analysis_section(key)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/fetch":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Route not found"})
            return

        try:
            payload = self._read_json_body()
            options = validate_fetch_payload(payload)
            report = fetch_recent_papers(self.server.database, **options)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            LOGGER.exception("arXiv fetch failed")
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "error": "arXiv fetch failed",
                    "detail": str(exc),
                },
            )
            return

        report_data = asdict(report)
        for key in ("start_utc", "revision_start_utc", "end_utc"):
            value = report_data[key]
            report_data[key] = value.isoformat() if value else None
        self._send_json(
            HTTPStatus.OK,
            {"report": report_data, "counts": self.server.database.counts()},
        )

    def _list_papers(self, query: dict[str, list[str]]) -> None:
        search = query.get("search", [""])[0].strip().casefold()
        category = query.get("category", [""])[0].strip()
        try:
            limit = self._bounded_query_int(
                query, "limit", default=100, maximum=500
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        papers, total = self.server.database.search_papers(
            search=search,
            category=category,
            limit=limit,
        )

        self._send_json(
            HTTPStatus.OK,
            {
                "papers": [paper_to_dict(paper) for paper in papers],
                "total": total,
                "categories": self.server.database.list_paper_categories(),
            },
        )

    def _get_graph(self, query: dict[str, list[str]]) -> None:
        try:
            limit = self._bounded_query_int(query, "limit", default=100, maximum=500)
            edge_limit = self._bounded_query_int(
                query, "edge_limit", default=200, maximum=1000
            )
            offset = self._bounded_query_int(
                query, "offset", default=0, maximum=1_000_000, minimum=0
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        node_type = query.get("node_type", [""])[0].strip() or None
        search = query.get("search", [""])[0].strip() or None
        relation = query.get("relation", [""])[0].strip() or None
        node_id = query.get("node_id", [""])[0].strip() or None
        nodes = self.server.database.list_graph_nodes(
            node_type=node_type,
            search=search,
            limit=limit,
            offset=offset,
        )
        edges = self.server.database.list_graph_edges(
            node_id=node_id,
            relation=relation,
            limit=edge_limit,
        )
        analysis = self.server.database.get_graph_analysis()
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ready" if analysis else "not_built",
                "build_id": analysis["build_id"] if analysis else None,
                "nodes": nodes,
                "edges": edges,
            },
        )

    def _get_placement(self, query: dict[str, list[str]]) -> None:
        identifier = query.get("id", [""])[0].strip()
        if not identifier:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "id query parameter is required"},
            )
            return
        node = self.server.database.get_graph_node(identifier)
        if node is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Graph document not found: {identifier}"},
            )
            return
        edges = self.server.database.list_graph_edges(
            node_id=node["node_id"], limit=200
        )
        neighbor_ids = {
            edge["target_id"] if edge["source_id"] == node["node_id"] else edge["source_id"]
            for edge in edges
        }
        neighbors = self.server.database.get_graph_nodes(neighbor_ids)
        self._send_json(
            HTTPStatus.OK,
            {"document": node, "relationships": edges, "neighbors": neighbors},
        )

    def _get_analysis_section(self, key: str) -> None:
        analysis = self.server.database.get_graph_analysis()
        if analysis is None:
            self._send_json(
                HTTPStatus.OK,
                {"status": "not_built", "build_id": None, key: []},
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ready",
                "build_id": analysis["build_id"],
                "built_at": analysis["built_at"],
                key: analysis.get(key, []),
            },
        )

    @staticmethod
    def _bounded_query_int(
        query: dict[str, list[str]],
        key: str,
        *,
        default: int,
        maximum: int,
        minimum: int = 1,
    ) -> int:
        raw = query.get(key, [str(default)])[0]
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        return value

    def _read_json_body(self) -> object:
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 1:
            raise ValueError("Request body is required")
        if length > MAX_BODY_BYTES:
            raise ValueError("Request body is too large")
        return json.loads(self.rfile.read(length))

    def _serve_static(self, path: str) -> None:
        filename, content_type = STATIC_FILES[path]
        content = files("arxiv_kg").joinpath("static", filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self._security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; connect-src 'self'",
        )

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: str | Path = "data/arxiv_kg.sqlite3",
) -> PaperFetcherServer:
    return PaperFetcherServer((host, port), db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the arXiv Paper Fetcher UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default="data/arxiv_kg.sqlite3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    server = create_server(args.host, args.port, args.db)
    print(f"Paper Fetcher UI: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
