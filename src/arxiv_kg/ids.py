"""Canonical identifiers and conservative text normalization."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_VERSION_RE = re.compile(r"v(?P<version>\d+)$", flags=re.IGNORECASE)


def split_arxiv_version(versioned_id: str) -> tuple[str, int]:
    """Return ``(base_id, version)`` for modern and legacy arXiv IDs."""

    clean = versioned_id.strip().split("arxiv.org/abs/")[-1]
    match = _VERSION_RE.search(clean)
    if match is None:
        return clean, 1
    return clean[: match.start()], int(match.group("version"))


def normalize_text_name(value: str) -> str:
    """Normalize spacing/case while retaining readable Unicode characters."""

    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def canonical_key(value: str) -> str:
    """Create a comparison key; do not display this key to users."""

    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def stable_node_id(node_type: str, canonical_name: str) -> str:
    """Create a compact deterministic ID from a node type and canonical name."""

    digest = hashlib.sha256(canonical_name.encode("utf-8")).hexdigest()[:16]
    return f"{node_type.casefold()}:{digest}"
