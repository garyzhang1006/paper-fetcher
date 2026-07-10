"""SQLite persistence for papers, extracted features, and graph records."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .models import PaperFeatures, PaperRecord, StoredFeatureRecord


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

                CREATE TABLE IF NOT EXISTS kg_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
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
                """
            )

    def upsert_paper(self, paper: PaperRecord) -> str:
        """Insert or update a paper; return inserted, updated, or unchanged."""

        now = utc_now_iso()
        with self.connect() as con:
            existing = con.execute(
                "SELECT versioned_id, updated_at FROM papers WHERE arxiv_id = ?",
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
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO pipeline_state(state_key, state_value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_at = excluded.updated_at
                """,
                (key, value, utc_now_iso()),
            )

    def counts(self) -> dict[str, int]:
        with self.connect() as con:
            return {
                "papers": con.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
                "features": con.execute(
                    "SELECT COUNT(*) FROM paper_features"
                ).fetchone()[0],
                "nodes": con.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0],
                "edges": con.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0],
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
