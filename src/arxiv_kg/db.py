"""SQLite persistence for papers, extracted features, and graph records."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .models import PaperFeatures, PaperRecord, SourceItem, StoredFeatureRecord
from .ids import canonical_key


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    versioned_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    affiliations_json TEXT NOT NULL,
                    categories_json TEXT NOT NULL,
                    primary_category TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    abs_url TEXT NOT NULL,
                    pdf_url TEXT,
                    doi TEXT,
                    journal_ref TEXT,
                    comment TEXT,
                    pdf_path TEXT,
                    text_path TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_features (
                    arxiv_id TEXT PRIMARY KEY,
                    source_versioned_id TEXT NOT NULL,
                    extractor TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    prompt_version TEXT,
                    extracted_at TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS pipeline_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_items (
                    item_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    title TEXT,
                    authors_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_topics_json TEXT NOT NULL DEFAULT '[]',
                    summary TEXT,
                    content_text TEXT NOT NULL,
                    canonical_url TEXT,
                    published_at TEXT,
                    updated_at TEXT,
                    retrieved_at TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    raw_metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_source_items_source_published
                    ON source_items(source_id, published_at DESC);

                CREATE TABLE IF NOT EXISTS kg_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    document_id TEXT,
                    properties_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kg_nodes_type_name
                    ON kg_nodes(node_type, canonical_name);

                CREATE TABLE IF NOT EXISTS kg_edges (
                    source_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    properties_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_id, relation, target_id),
                    FOREIGN KEY (source_id) REFERENCES kg_nodes(node_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES kg_nodes(node_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS kg_builds (
                    build_id TEXT PRIMARY KEY,
                    built_at TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    node_count INTEGER NOT NULL,
                    edge_count INTEGER NOT NULL,
                    analysis_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kg_edges_target
                    ON kg_edges(target_id);
                """
            )
            source_columns = {
                row[1] for row in con.execute("PRAGMA table_info(source_items)")
            }
            if "source_topics_json" not in source_columns:
                con.execute(
                    "ALTER TABLE source_items ADD COLUMN "
                    "source_topics_json TEXT NOT NULL DEFAULT '[]'"
                )
            graph_columns = {
                row[1] for row in con.execute("PRAGMA table_info(kg_nodes)")
            }
            if "document_id" not in graph_columns:
                con.execute("ALTER TABLE kg_nodes ADD COLUMN document_id TEXT")
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_nodes_document_id "
                "ON kg_nodes(document_id)"
            )

    def upsert_paper(self, paper: PaperRecord) -> str:
        """Insert or update a paper; return inserted, updated, or unchanged."""

        now = utc_now_iso()
        with self.connect() as con:
            return self._upsert_paper(con, paper, now)

    def upsert_papers(self, papers: Iterable[PaperRecord]) -> dict[str, int]:
        """Upsert a batch atomically and return status counts."""

        counts = {"inserted": 0, "updated": 0, "unchanged": 0}
        now = utc_now_iso()
        with self.connect() as con:
            for paper in papers:
                status = self._upsert_paper(con, paper, now)
                counts[status] += 1
        return counts

    @staticmethod
    def _upsert_paper(
        con: sqlite3.Connection, paper: PaperRecord, now: str
    ) -> str:
        existing = con.execute(
            "SELECT versioned_id, version, updated_at FROM papers WHERE arxiv_id = ?",
            (paper.arxiv_id,),
        ).fetchone()
        values = (
            paper.versioned_id,
            paper.version,
            paper.title,
            paper.abstract,
            json.dumps(paper.authors, ensure_ascii=False),
            json.dumps(paper.affiliations, ensure_ascii=False),
            json.dumps(paper.categories, ensure_ascii=False),
            paper.primary_category,
            paper.published_at.isoformat(),
            paper.updated_at.isoformat(),
            paper.abs_url,
            paper.pdf_url,
            paper.doi,
            paper.journal_ref,
            paper.comment,
            now,
        )

        if existing is None:
            con.execute(
                """
                INSERT INTO papers (
                    arxiv_id, versioned_id, version, title, abstract,
                    authors_json, affiliations_json, categories_json,
                    primary_category, published_at, updated_at, abs_url,
                    pdf_url, doi, journal_ref, comment, created_at,
                    last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (paper.arxiv_id, *values[:-1], now, now),
            )
            return "inserted"

        if existing["version"] > paper.version:
            con.execute(
                "UPDATE papers SET last_seen_at = ? WHERE arxiv_id = ?",
                (now, paper.arxiv_id),
            )
            return "unchanged"

        changed = (
            existing["versioned_id"] != paper.versioned_id
            or existing["updated_at"] != paper.updated_at.isoformat()
        )
        if not changed:
            con.execute(
                "UPDATE papers SET last_seen_at = ? WHERE arxiv_id = ?",
                (now, paper.arxiv_id),
            )
            return "unchanged"

        con.execute(
            """
            UPDATE papers SET
                versioned_id = ?, version = ?, title = ?, abstract = ?,
                authors_json = ?, affiliations_json = ?, categories_json = ?,
                primary_category = ?, published_at = ?, updated_at = ?,
                abs_url = ?, pdf_url = ?, doi = ?, journal_ref = ?,
                comment = ?, last_seen_at = ?, last_error = NULL,
                pdf_path = NULL, text_path = NULL
            WHERE arxiv_id = ?
            """,
            (*values, paper.arxiv_id),
        )
        return "updated"

    def upsert_source_item(self, item: SourceItem) -> str:
        """Insert or refresh one normalized feed item."""

        now = utc_now_iso()
        with self.connect() as con:
            return self._upsert_source_item(con, item, now)

    def upsert_source_items(
        self,
        items: Iterable[SourceItem],
        *,
        checkpoint_key: str,
        checkpoint_value: str,
    ) -> dict[str, int]:
        """Commit one source's normalized items and checkpoint atomically."""

        counts = {"inserted": 0, "updated": 0, "unchanged": 0}
        now = utc_now_iso()
        with self.connect() as con:
            for item in items:
                status = self._upsert_source_item(con, item, now)
                counts[status] += 1
            con.execute(
                """
                INSERT INTO pipeline_state(state_key, state_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_at = excluded.updated_at
                """,
                (checkpoint_key, checkpoint_value, now),
            )
        return counts

    @staticmethod
    def _upsert_source_item(
        con: sqlite3.Connection, item: SourceItem, now: str
    ) -> str:
        existing = con.execute(
            "SELECT content_sha256 FROM source_items WHERE item_id = ?",
            (item.item_id,),
        ).fetchone()
        values = (
            item.source_id,
            item.source_name,
            item.source_kind,
            item.external_id,
            item.title,
            json.dumps(item.authors, ensure_ascii=False),
            json.dumps(item.tags, ensure_ascii=False),
            item.summary,
            item.content_text,
            json.dumps(item.source_topics, ensure_ascii=False),
            item.canonical_url,
            item.published_at.isoformat() if item.published_at else None,
            item.updated_at.isoformat() if item.updated_at else None,
            item.retrieved_at.isoformat(),
            item.content_sha256,
            json.dumps(item.raw_metadata, ensure_ascii=False, sort_keys=True),
            now,
        )
        if existing is None:
            con.execute(
                """
                INSERT INTO source_items (
                    item_id, source_id, source_name, source_kind, external_id,
                    title, authors_json, tags_json, summary, content_text,
                    source_topics_json, canonical_url, published_at, updated_at, retrieved_at,
                    content_sha256, raw_metadata_json, created_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (item.item_id, *values[:-1], now, now),
            )
            return "inserted"
        if existing["content_sha256"] == item.content_sha256:
            con.execute(
                """
                UPDATE source_items SET
                    source_name = ?, source_kind = ?, retrieved_at = ?,
                    raw_metadata_json = ?, last_seen_at = ?
                WHERE item_id = ?
                """,
                (
                    item.source_name,
                    item.source_kind,
                    item.retrieved_at.isoformat(),
                    json.dumps(item.raw_metadata, ensure_ascii=False, sort_keys=True),
                    now,
                    item.item_id,
                ),
            )
            return "unchanged"
        con.execute(
            """
            UPDATE source_items SET
                source_id = ?, source_name = ?, source_kind = ?, external_id = ?,
                title = ?, authors_json = ?, tags_json = ?, summary = ?,
                content_text = ?, source_topics_json = ?, canonical_url = ?, published_at = ?,
                updated_at = ?, retrieved_at = ?, content_sha256 = ?,
                raw_metadata_json = ?, last_seen_at = ?
            WHERE item_id = ?
            """,
            (*values, item.item_id),
        )
        return "updated"

    def iter_source_items(self, limit: int | None = None) -> Iterator[SourceItem]:
        sql = "SELECT * FROM source_items ORDER BY COALESCE(published_at, retrieved_at) ASC"
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self.connect() as con:
            rows = con.execute(sql, params).fetchall()
        for row in rows:
            yield SourceItem(
                item_id=row["item_id"],
                source_id=row["source_id"],
                source_name=row["source_name"],
                source_kind=row["source_kind"],
                external_id=row["external_id"],
                title=row["title"],
                authors=json.loads(row["authors_json"]),
                tags=json.loads(row["tags_json"]),
                source_topics=json.loads(row["source_topics_json"]),
                summary=row["summary"],
                content_text=row["content_text"],
                canonical_url=row["canonical_url"],
                published_at=(
                    datetime.fromisoformat(row["published_at"])
                    if row["published_at"]
                    else None
                ),
                updated_at=(
                    datetime.fromisoformat(row["updated_at"])
                    if row["updated_at"]
                    else None
                ),
                retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
                content_sha256=row["content_sha256"],
                raw_metadata=json.loads(row["raw_metadata_json"]),
            )

    def set_paper_file(self, arxiv_id: str, *, kind: str, path: str | Path) -> None:
        if kind not in {"pdf", "text"}:
            raise ValueError("kind must be 'pdf' or 'text'")
        column = "pdf_path" if kind == "pdf" else "text_path"
        with self.connect() as con:
            con.execute(
                f"UPDATE papers SET {column} = ?, last_error = NULL WHERE arxiv_id = ?",
                (str(path), arxiv_id),
            )

    def set_paper_error(self, arxiv_id: str, message: str) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE papers SET last_error = ? WHERE arxiv_id = ?",
                (message[:1000], arxiv_id),
            )

    def get_paper(self, arxiv_id: str) -> PaperRecord | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        return self._paper_from_row(row) if row else None

    def get_paper_paths(self, arxiv_id: str) -> tuple[str | None, str | None]:
        with self.connect() as con:
            row = con.execute(
                "SELECT pdf_path, text_path FROM papers WHERE arxiv_id = ?",
                (arxiv_id,),
            ).fetchone()
        if row is None:
            raise KeyError(arxiv_id)
        return row["pdf_path"], row["text_path"]

    def iter_papers(self, limit: int | None = None) -> Iterator[PaperRecord]:
        sql = "SELECT * FROM papers ORDER BY published_at DESC"
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self.connect() as con:
            rows = con.execute(sql, params).fetchall()
        for row in rows:
            yield self._paper_from_row(row)

    def search_papers(
        self,
        *,
        search: str = "",
        category: str = "",
        limit: int = 100,
    ) -> tuple[list[PaperRecord], int]:
        """Search the complete saved corpus and return bounded rows plus total."""

        conditions: list[str] = []
        params: list[object] = []
        if search:
            conditions.append(
                "LOWER(title || ' ' || abstract || ' ' || arxiv_id || ' ' || "
                "authors_json) LIKE ?"
            )
            params.append(f"%{search.casefold()}%")
        if category:
            conditions.append(
                "EXISTS (SELECT 1 FROM json_each(papers.categories_json) "
                "WHERE json_each.value = ?)"
            )
            params.append(category)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connect() as con:
            total = con.execute(
                "SELECT COUNT(*) FROM papers" + where,
                tuple(params),
            ).fetchone()[0]
            rows = con.execute(
                "SELECT * FROM papers"
                + where
                + " ORDER BY published_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._paper_from_row(row) for row in rows], int(total)

    def list_paper_categories(self) -> list[str]:
        """Return every distinct saved arXiv category in display order."""

        with self.connect() as con:
            rows = con.execute(
                """
                SELECT DISTINCT json_each.value AS category
                FROM papers, json_each(papers.categories_json)
                ORDER BY category
                """
            ).fetchall()
        return [str(row["category"]) for row in rows]

    def iter_papers_needing_features(
        self,
        extractor: str,
        extractor_version: str,
        prompt_version: str | None,
        limit: int | None = None,
    ) -> Iterator[PaperRecord]:
        sql = """
            SELECT p.*
            FROM papers AS p
            LEFT JOIN paper_features AS f ON p.arxiv_id = f.arxiv_id
            WHERE f.arxiv_id IS NULL
               OR f.source_versioned_id != p.versioned_id
               OR f.extractor != ?
               OR f.extractor_version != ?
               OR COALESCE(f.prompt_version, '') != COALESCE(?, '')
            ORDER BY p.published_at ASC
        """
        params: list[object] = [extractor, extractor_version, prompt_version]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self.connect() as con:
            rows = con.execute(sql, tuple(params)).fetchall()
        for row in rows:
            yield self._paper_from_row(row)

    def save_features(
        self,
        *,
        paper: PaperRecord,
        features: PaperFeatures,
        extractor: str,
        extractor_version: str,
        prompt_version: str | None,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as con:
            self._save_features(
                con,
                paper=paper,
                features=features,
                extractor=extractor,
                extractor_version=extractor_version,
                prompt_version=prompt_version,
                now=now,
            )

    def save_features_batch(
        self,
        records: Iterable[tuple[PaperRecord, PaperFeatures]],
        *,
        extractor: str,
        extractor_version: str,
        prompt_version: str | None,
    ) -> int:
        """Persist one extractor run in a single transaction."""

        saved = 0
        now = utc_now_iso()
        with self.connect() as con:
            for paper, features in records:
                self._save_features(
                    con,
                    paper=paper,
                    features=features,
                    extractor=extractor,
                    extractor_version=extractor_version,
                    prompt_version=prompt_version,
                    now=now,
                )
                saved += 1
        return saved

    @staticmethod
    def _save_features(
        con: sqlite3.Connection,
        *,
        paper: PaperRecord,
        features: PaperFeatures,
        extractor: str,
        extractor_version: str,
        prompt_version: str | None,
        now: str,
    ) -> None:
        con.execute(
            """
            INSERT INTO paper_features (
                arxiv_id, source_versioned_id, extractor,
                extractor_version, prompt_version, extracted_at,
                features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                source_versioned_id = excluded.source_versioned_id,
                extractor = excluded.extractor,
                extractor_version = excluded.extractor_version,
                prompt_version = excluded.prompt_version,
                extracted_at = excluded.extracted_at,
                features_json = excluded.features_json
            """,
            (
                paper.arxiv_id,
                paper.versioned_id,
                extractor,
                extractor_version,
                prompt_version,
                now,
                features.model_dump_json(),
            ),
        )

    def get_stored_features(self, arxiv_id: str) -> StoredFeatureRecord | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM paper_features WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        if row is None:
            return None
        return StoredFeatureRecord(
            arxiv_id=row["arxiv_id"],
            source_versioned_id=row["source_versioned_id"],
            extractor=row["extractor"],
            extractor_version=row["extractor_version"],
            prompt_version=row["prompt_version"],
            extracted_at=datetime.fromisoformat(row["extracted_at"]),
            features=PaperFeatures.model_validate_json(row["features_json"]),
        )

    def iter_papers_with_features(
        self, limit: int | None = None
    ) -> Iterator[tuple[PaperRecord, StoredFeatureRecord]]:
        sql = """
            SELECT p.*, f.source_versioned_id AS f_source_versioned_id,
                   f.extractor AS f_extractor,
                   f.extractor_version AS f_extractor_version,
                   f.prompt_version AS f_prompt_version,
                   f.extracted_at AS f_extracted_at,
                   f.features_json AS f_features_json
            FROM papers p
            JOIN paper_features f ON p.arxiv_id = f.arxiv_id
            WHERE p.versioned_id = f.source_versioned_id
            ORDER BY p.published_at ASC
        """
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self.connect() as con:
            rows = con.execute(sql, params).fetchall()
        for row in rows:
            paper = self._paper_from_row(row)
            stored = StoredFeatureRecord(
                arxiv_id=paper.arxiv_id,
                source_versioned_id=row["f_source_versioned_id"],
                extractor=row["f_extractor"],
                extractor_version=row["f_extractor_version"],
                prompt_version=row["f_prompt_version"],
                extracted_at=datetime.fromisoformat(row["f_extracted_at"]),
                features=PaperFeatures.model_validate_json(row["f_features_json"]),
            )
            yield paper, stored

    def get_state(self, key: str) -> str | None:
        with self.connect() as con:
            row = con.execute(
                "SELECT state_value FROM pipeline_state WHERE state_key = ?", (key,)
            ).fetchone()
        return row["state_value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        self.set_states({key: value})

    def set_states(self, states: dict[str, str]) -> None:
        """Write related pipeline checkpoints in one transaction."""

        with self.connect() as con:
            now = utc_now_iso()
            for key, value in states.items():
                con.execute(
                    """
                    INSERT INTO pipeline_state(state_key, state_value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(state_key) DO UPDATE SET
                        state_value = excluded.state_value,
                        updated_at = excluded.updated_at
                    """,
                    (key, value, now),
                )

    def replace_graph(
        self,
        *,
        build_id: str,
        document_count: int,
        nodes: Iterable[object],
        edges: Iterable[object],
        analysis: Mapping[str, Any],
    ) -> dict[str, int | str]:
        """Atomically replace the readable graph and activate its analysis."""

        node_records = list(nodes)
        edge_records = list(edges)
        node_ids = {str(self._record_value(node, "node_id")) for node in node_records}
        if len(node_ids) != len(node_records):
            raise ValueError("Graph rebuild contains duplicate node IDs")
        if len(build_id) != 64 or any(
            character not in "0123456789abcdef" for character in build_id
        ):
            raise ValueError("Graph build ID must be a 64-character lowercase SHA-256")
        actual_document_count = sum(
            self._record_value(node, "node_type")
            in {"paper", "report", "blog_post", "social_post"}
            for node in node_records
        )
        if document_count != actual_document_count:
            raise ValueError(
                f"Graph document count {document_count} does not match "
                f"{actual_document_count} document nodes"
            )
        for edge in edge_records:
            source_id = str(self._record_value(edge, "source_id"))
            target_id = str(self._record_value(edge, "target_id"))
            if source_id not in node_ids or target_id not in node_ids:
                raise ValueError(
                    f"Graph edge {source_id!r} -> {target_id!r} references a missing node"
                )

        now = utc_now_iso()
        analysis_json = json.dumps(
            dict(analysis), ensure_ascii=False, sort_keys=True, default=self._json_default
        )
        with self.connect() as con:
            con.execute("DELETE FROM kg_edges")
            con.execute("DELETE FROM kg_nodes")
            for node in sorted(
                node_records,
                key=lambda value: self._record_value(value, "node_id"),
            ):
                con.execute(
                    """
                    INSERT INTO kg_nodes (
                        node_id, node_type, name, canonical_name, document_id,
                        properties_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._record_value(node, "node_id"),
                        self._record_value(node, "node_type"),
                        self._record_value(node, "name"),
                        self._record_value(node, "canonical_name"),
                        self._record_value(node, "properties").get("document_id"),
                        json.dumps(
                            self._record_value(node, "properties"),
                            ensure_ascii=False,
                            sort_keys=True,
                            default=self._json_default,
                        ),
                        now,
                    ),
                )
            for edge in sorted(
                edge_records,
                key=lambda value: (
                    self._record_value(value, "source_id"),
                    self._record_value(value, "relation"),
                    self._record_value(value, "target_id"),
                ),
            ):
                con.execute(
                    """
                    INSERT INTO kg_edges (
                        source_id, relation, target_id, properties_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        self._record_value(edge, "source_id"),
                        self._record_value(edge, "relation"),
                        self._record_value(edge, "target_id"),
                        json.dumps(
                            self._edge_properties(edge),
                            ensure_ascii=False,
                            sort_keys=True,
                            default=self._json_default,
                        ),
                        now,
                    ),
                )
            con.execute(
                """
                INSERT INTO kg_builds (
                    build_id, built_at, document_count, node_count,
                    edge_count, analysis_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(build_id) DO UPDATE SET
                    built_at = excluded.built_at,
                    document_count = excluded.document_count,
                    node_count = excluded.node_count,
                    edge_count = excluded.edge_count,
                    analysis_json = excluded.analysis_json
                """,
                (
                    build_id,
                    now,
                    document_count,
                    len(node_records),
                    len(edge_records),
                    analysis_json,
                ),
            )
            con.execute(
                """
                INSERT INTO pipeline_state(state_key, state_value, updated_at)
                VALUES ('active_graph_build_id', ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_at = excluded.updated_at
                """,
                (build_id, now),
            )
            con.execute(
                """
                DELETE FROM kg_builds
                WHERE build_id NOT IN (
                    SELECT build_id FROM kg_builds
                    ORDER BY built_at DESC, build_id DESC
                    LIMIT 30
                )
                """
            )
        return {
            "build_id": build_id,
            "documents": document_count,
            "nodes": len(node_records),
            "edges": len(edge_records),
        }

    def list_graph_nodes(
        self,
        *,
        node_type: str | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit = min(max(limit, 1), 500)
        offset = max(offset, 0)
        conditions: list[str] = []
        params: list[object] = []
        if node_type:
            conditions.append("node_type = ?")
            params.append(node_type)
        if search:
            conditions.append("canonical_name LIKE ?")
            params.append(f"%{canonical_key(search)}%")
        sql = "SELECT * FROM kg_nodes"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY node_type, canonical_name LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self.connect() as con:
            rows = con.execute(sql, tuple(params)).fetchall()
        return [self._graph_node_from_row(row) for row in rows]

    def list_graph_edges(
        self,
        *,
        node_id: str | None = None,
        relation: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        limit = min(max(limit, 1), 1000)
        conditions: list[str] = []
        params: list[object] = []
        if node_id:
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([node_id, node_id])
        if relation:
            conditions.append("relation = ?")
            params.append(relation)
        sql = "SELECT * FROM kg_edges"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY relation, source_id, target_id LIMIT ?"
        params.append(limit)
        with self.connect() as con:
            rows = con.execute(sql, tuple(params)).fetchall()
        return [self._graph_edge_from_row(row) for row in rows]

    def get_graph_node(self, identifier: str) -> dict[str, Any] | None:
        """Find a node by stable node ID or original document ID."""

        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM kg_nodes WHERE node_id = ?", (identifier,)
            ).fetchone()
            if row is None:
                row = con.execute(
                    "SELECT * FROM kg_nodes WHERE document_id = ?",
                    (identifier,),
                ).fetchone()
        return self._graph_node_from_row(row) if row else None

    def get_graph_nodes(self, node_ids: Iterable[str]) -> list[dict[str, Any]]:
        identifiers = sorted(set(node_ids))[:500]
        if not identifiers:
            return []
        placeholders = ", ".join("?" for _ in identifiers)
        with self.connect() as con:
            rows = con.execute(
                f"SELECT * FROM kg_nodes WHERE node_id IN ({placeholders}) "
                "ORDER BY node_type, canonical_name",
                tuple(identifiers),
            ).fetchall()
        return [self._graph_node_from_row(row) for row in rows]

    def get_graph_analysis(self) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT b.* FROM kg_builds b
                JOIN pipeline_state s ON s.state_value = b.build_id
                WHERE s.state_key = 'active_graph_build_id'
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "build_id": row["build_id"],
            "built_at": row["built_at"],
            "document_count": row["document_count"],
            "node_count": row["node_count"],
            "edge_count": row["edge_count"],
            **json.loads(row["analysis_json"]),
        }

    def counts(self) -> dict[str, int]:
        with self.connect() as con:
            return {
                "papers": con.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
                "source_items": con.execute(
                    "SELECT COUNT(*) FROM source_items"
                ).fetchone()[0],
                "features": con.execute(
                    "SELECT COUNT(*) FROM paper_features"
                ).fetchone()[0],
                "nodes": con.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0],
                "edges": con.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0],
            }

    def source_counts(self) -> list[dict[str, str | int]]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT source_id, source_name, source_kind, COUNT(*) AS item_count
                FROM source_items
                GROUP BY source_id, source_name, source_kind
                ORDER BY source_id
                """
            ).fetchall()
        return [
            {
                "source_id": row["source_id"],
                "source_name": row["source_name"],
                "source_kind": row["source_kind"],
                "item_count": row["item_count"],
            }
            for row in rows
        ]

    @staticmethod
    def _record_value(record: object, key: str) -> Any:
        if isinstance(record, Mapping):
            return record[key]
        return getattr(record, key)

    @classmethod
    def _edge_properties(cls, edge: object) -> dict[str, Any]:
        properties = dict(cls._record_value(edge, "properties"))
        if isinstance(edge, Mapping):
            weight = edge.get("weight")
        else:
            weight = getattr(edge, "weight", None)
        if weight is not None:
            properties.setdefault("weight", weight)
        return properties

    @staticmethod
    def _json_default(value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    @staticmethod
    def _graph_node_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "name": row["name"],
            "canonical_name": row["canonical_name"],
            "properties": json.loads(row["properties_json"]),
        }

    @staticmethod
    def _graph_edge_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "source_id": row["source_id"],
            "relation": row["relation"],
            "target_id": row["target_id"],
            "properties": json.loads(row["properties_json"]),
        }

    @staticmethod
    def _paper_from_row(row: sqlite3.Row) -> PaperRecord:
        return PaperRecord(
            arxiv_id=row["arxiv_id"],
            versioned_id=row["versioned_id"],
            version=row["version"],
            title=row["title"],
            abstract=row["abstract"],
            authors=json.loads(row["authors_json"]),
            affiliations=json.loads(row["affiliations_json"]),
            categories=json.loads(row["categories_json"]),
            primary_category=row["primary_category"],
            published_at=datetime.fromisoformat(row["published_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            abs_url=row["abs_url"],
            pdf_url=row["pdf_url"],
            doi=row["doi"],
            journal_ref=row["journal_ref"],
            comment=row["comment"],
        )
