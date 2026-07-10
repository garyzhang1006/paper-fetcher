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
            limit = int(query.get("limit", ["100"])[0])
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "limit must be an integer"})
            return
        limit = min(max(limit, 1), 500)

        matching: list[dict[str, Any]] = []
        for paper in self.server.database.iter_papers(limit=500):
            if category and category not in paper.categories:
                continue
            haystack = " ".join(
                [paper.title, paper.abstract, paper.arxiv_id, *paper.authors]
            ).casefold()
            if search and search not in haystack:
                continue
            matching.append(paper_to_dict(paper))
            if len(matching) >= limit:
                break

        self._send_json(
            HTTPStatus.OK,
            {"papers": matching, "total": len(matching)},
        )

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
