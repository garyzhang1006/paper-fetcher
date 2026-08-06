"""Bounded normalization for public HTTPS RSS and Atom feeds."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

import requests
from pydantic import ValidationError

from .models import FeedSourceConfig, SourceItem

MAX_FEED_BYTES = 1_000_000
REQUEST_TIMEOUT_SECONDS = 30
TOTAL_REQUEST_SECONDS = 60
MAX_REDIRECTS = 5
USER_AGENT = (
    "paper-fetcher/0.1 (+https://github.com/garyzhang1006/paper-fetcher)"
)


class FeedFetchError(RuntimeError):
    """Network or HTTP failure that must prevent a success checkpoint."""


class FeedParseError(ValueError):
    """Incomplete or unsafe feed content that must not be treated as success."""


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if clean:
            self.parts.append(clean)


def _plain_text(value: str | None) -> str:
    if not value:
        return ""
    parser = _PlainTextParser()
    parser.feed(value)
    parser.close()
    return " ".join(parser.parts)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _child(element: ElementTree.Element, *names: str) -> ElementTree.Element | None:
    wanted = {name.casefold() for name in names}
    return next(
        (child for child in element if _local_name(child.tag) in wanted),
        None,
    )


def _children(element: ElementTree.Element, *names: str) -> list[ElementTree.Element]:
    wanted = {name.casefold() for name in names}
    return [child for child in element if _local_name(child.tag) in wanted]


def _raw_element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def _element_text(element: ElementTree.Element | None) -> str:
    return _plain_text(_raw_element_text(element))


def _optional_text(element: ElementTree.Element | None) -> str | None:
    value = _element_text(element)
    return value or None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    clean = value.strip()
    try:
        parsed = parsedate_to_datetime(clean)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _canonical_url(entry: ElementTree.Element, feed_format: str) -> str | None:
    if feed_format == "rss":
        return _optional_text(_child(entry, "link"))
    links = _children(entry, "link")
    alternate = next(
        (link for link in links if link.attrib.get("rel", "alternate") == "alternate"),
        links[0] if links else None,
    )
    value = alternate.attrib.get("href", "").strip() if alternate is not None else ""
    return value or None


def _authors(entry: ElementTree.Element, feed_format: str) -> list[str]:
    if feed_format == "rss":
        return [
            value
            for element in _children(entry, "creator", "author")
            if (value := _element_text(element))
        ]
    return [
        value
        for author in _children(entry, "author")
        if (value := _element_text(_child(author, "name")))
    ]


def _tags(entry: ElementTree.Element, feed_format: str) -> list[str]:
    output: list[str] = []
    for category in _children(entry, "category"):
        value = (
            category.attrib.get("term", "").strip()
            if feed_format == "atom"
            else _element_text(category)
        )
        if value:
            output.append(value)
    return output


def _unique_strings(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def _stable_item_id(source_id: str, external_id: str) -> str:
    digest = hashlib.sha256(
        f"{source_id}\0{external_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"source_item:{digest}"


def _content_hash(values: dict[str, object]) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_entry(
    entry: ElementTree.Element,
    *,
    source: FeedSourceConfig,
    feed_format: str,
    retrieved_at: datetime,
    position: int,
) -> SourceItem:
    guid_element = _child(entry, "guid" if feed_format == "rss" else "id")
    guid = _element_text(guid_element)
    canonical_url = _canonical_url(entry, feed_format)
    external_id = guid or canonical_url
    if not external_id:
        raise FeedParseError(
            f"Feed {source.source_id!r} item {position} has neither a GUID nor "
            "a canonical URL"
        )

    title = _optional_text(_child(entry, "title"))
    summary_element = _child(entry, "description" if feed_format == "rss" else "summary")
    summary = _optional_text(summary_element)
    content_element = _child(entry, "encoded", "content")
    content_text = _element_text(content_element) or summary or ""
    authors = _unique_strings(_authors(entry, feed_format))
    tags = _unique_strings(_tags(entry, feed_format))

    if feed_format == "rss":
        published_at = _parse_datetime(_raw_element_text(_child(entry, "pubdate")))
        updated_at = _parse_datetime(_raw_element_text(_child(entry, "updated")))
    else:
        published_at = _parse_datetime(_raw_element_text(_child(entry, "published")))
        updated_at = _parse_datetime(_raw_element_text(_child(entry, "updated")))

    semantic_values: dict[str, object] = {
        "source_id": source.source_id,
        "source_kind": source.source_kind,
        "external_id": external_id,
        "title": title,
        "authors": authors,
        "tags": tags,
        "source_topics": source.default_topics,
        "summary": summary,
        "content_text": content_text,
        "canonical_url": canonical_url,
        "published_at": published_at.isoformat() if published_at else None,
        "updated_at": updated_at.isoformat() if updated_at else None,
    }
    raw_metadata = {
        "entry_format": feed_format,
        "feed_url": source.feed_url,
    }
    if guid:
        raw_metadata["guid"] = guid

    try:
        return SourceItem(
            item_id=_stable_item_id(source.source_id, external_id),
            source_id=source.source_id,
            source_name=source.name,
            source_kind=source.source_kind,
            external_id=external_id,
            title=title,
            authors=authors,
            tags=tags,
            source_topics=source.default_topics,
            summary=summary,
            content_text=content_text,
            canonical_url=canonical_url,
            published_at=published_at,
            updated_at=updated_at,
            retrieved_at=retrieved_at,
            content_sha256=_content_hash(semantic_values),
            raw_metadata=raw_metadata,
        )
    except ValidationError as exc:
        raise FeedParseError(
            f"Feed {source.source_id!r} item {position} is invalid: {exc}"
        ) from exc


def parse_feed(
    content: bytes,
    source: FeedSourceConfig,
    *,
    retrieved_at: datetime | None = None,
) -> list[SourceItem]:
    """Parse one complete RSS or Atom payload into bounded typed records."""

    if len(content) > MAX_FEED_BYTES:
        raise FeedParseError(
            f"Feed {source.source_id!r} exceeds maximum response size of "
            f"{MAX_FEED_BYTES} bytes"
        )
    if not content.strip():
        raise FeedParseError(f"Feed {source.source_id!r} returned an empty response")
    lowered = content.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise FeedParseError("Feed XML document type declarations are not allowed")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise FeedParseError(
            f"Feed {source.source_id!r} contains malformed XML: {exc}"
        ) from exc

    root_name = _local_name(root.tag)
    if root_name == "rss":
        channel = _child(root, "channel")
        entries = _children(channel, "item") if channel is not None else []
        feed_format = "rss"
    elif root_name == "feed":
        entries = _children(root, "entry")
        feed_format = "atom"
    else:
        raise FeedParseError(
            f"Feed {source.source_id!r} has unsupported feed root {root_name!r}"
        )
    if not entries:
        raise FeedParseError(f"Feed {source.source_id!r} contains no items")

    retrieved = retrieved_at or datetime.now(UTC)
    if retrieved.tzinfo is None:
        raise FeedParseError("retrieved_at must include a timezone")
    retrieved = retrieved.astimezone(UTC)
    output: list[SourceItem] = []
    seen: set[str] = set()
    if len(entries) > source.max_items:
        raise FeedParseError(
            f"Feed {source.source_id!r} contains {len(entries)} items, exceeding "
            f"the configured complete-feed limit of {source.max_items}; checkpoint "
            "was not advanced"
        )
    for position, entry in enumerate(entries, start=1):
        item = _normalize_entry(
            entry,
            source=source,
            feed_format=feed_format,
            retrieved_at=retrieved,
            position=position,
        )
        if item.item_id not in seen:
            seen.add(item.item_id)
            output.append(item)
    if not output:
        raise FeedParseError(f"Feed {source.source_id!r} produced no usable items")
    return output


def _read_bounded_response(
    response: Any,
    source: FeedSourceConfig,
    *,
    deadline: float,
) -> bytes:
    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            if int(declared_length) > MAX_FEED_BYTES:
                raise FeedFetchError(
                    f"Feed {source.source_id!r} exceeds maximum response size of "
                    f"{MAX_FEED_BYTES} bytes"
                )
        except ValueError:
            pass

    blocks: list[bytes] = []
    total = 0
    for block in response.iter_content(chunk_size=64 * 1024):
        if time.monotonic() > deadline:
            raise FeedFetchError(
                f"Feed {source.source_id!r} exceeded total download time of "
                f"{TOTAL_REQUEST_SECONDS} seconds"
            )
        if not block:
            continue
        total += len(block)
        if total > MAX_FEED_BYTES:
            raise FeedFetchError(
                f"Feed {source.source_id!r} exceeds maximum response size of "
                f"{MAX_FEED_BYTES} bytes"
            )
        blocks.append(block)
    return b"".join(blocks)


def _validate_public_https_url(url: str, source: FeedSourceConfig) -> None:
    """Reject credential-bearing or non-public destinations before connecting."""

    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise FeedFetchError(
            f"Feed {source.source_id!r} destination must use HTTPS"
        )
    if parsed.username or parsed.password:
        raise FeedFetchError(
            f"Feed {source.source_id!r} destination must not contain credentials"
        )
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise FeedFetchError(
            f"Feed {source.source_id!r} destination does not resolve exclusively "
            "to public IP addresses"
        )
    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None and not literal_address.is_global:
        raise FeedFetchError(
            f"Feed {source.source_id!r} destination does not resolve exclusively "
            "to public IP addresses"
        )
    try:
        addresses = {
            ipaddress.ip_address(result[4][0])
            for result in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or 443,
                type=socket.SOCK_STREAM,
            )
        }
    except (OSError, ValueError) as exc:
        raise FeedFetchError(
            f"Feed {source.source_id!r} host could not be resolved safely: {exc}"
        ) from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise FeedFetchError(
            f"Feed {source.source_id!r} destination does not resolve exclusively "
            "to public IP addresses"
        )


def _fetch_with_session(
    source: FeedSourceConfig,
    *,
    retrieved_at: datetime | None,
    session: Any,
) -> list[SourceItem]:
    deadline = time.monotonic() + TOTAL_REQUEST_SECONDS
    current_url = source.feed_url
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            if time.monotonic() > deadline:
                raise FeedFetchError(
                    f"Feed {source.source_id!r} exceeded total request time of "
                    f"{TOTAL_REQUEST_SECONDS} seconds"
                )
            _validate_public_https_url(current_url, source)
            with session.get(
                current_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": (
                        "application/rss+xml, application/atom+xml, "
                        "application/xml, text/xml"
                    ),
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
                stream=True,
                allow_redirects=False,
            ) as response:
                response_url = getattr(response, "url", current_url)
                _validate_public_https_url(response_url, source)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise FeedFetchError(
                            f"Feed {source.source_id!r} returned a redirect without "
                            "a Location header"
                        )
                    if redirect_count == MAX_REDIRECTS:
                        raise FeedFetchError(
                            f"Feed {source.source_id!r} exceeded {MAX_REDIRECTS} redirects"
                        )
                    next_url = urljoin(current_url, location)
                    _validate_public_https_url(next_url, source)
                    current_url = next_url
                    continue
                response.raise_for_status()
                content = _read_bounded_response(
                    response,
                    source,
                    deadline=deadline,
                )
                break
        else:  # pragma: no cover - loop exits through break or explicit error
            raise FeedFetchError(
                f"Feed {source.source_id!r} did not return content"
            )
    except FeedFetchError:
        raise
    except Exception as exc:
        raise FeedFetchError(
            f"Feed {source.source_id!r} fetch failed: {exc}"
        ) from exc
    return parse_feed(content, source, retrieved_at=retrieved_at)


def fetch_feed(
    source: FeedSourceConfig,
    *,
    retrieved_at: datetime | None = None,
    session: Any | None = None,
) -> list[SourceItem]:
    """Fetch and normalize one source without credentials or unbounded reads."""

    if not source.enabled:
        return []
    if session is not None:
        return _fetch_with_session(
            source,
            retrieved_at=retrieved_at,
            session=session,
        )
    with requests.Session() as owned_session:
        return _fetch_with_session(
            source,
            retrieved_at=retrieved_at,
            session=owned_session,
        )


def load_feed_sources(path: str | Path) -> list[FeedSourceConfig]:
    """Load a strict source list and reject ambiguous duplicate identities."""

    source_path = Path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"feed source configuration not found: {source_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"feed source configuration is invalid JSON: {source_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
        raise ValueError("feed source configuration must contain a sources list")
    if not payload["sources"]:
        raise ValueError("feed source configuration contains no sources")

    sources = [FeedSourceConfig.model_validate(item) for item in payload["sources"]]
    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            raise ValueError(f"duplicate source_id in feed configuration: {source.source_id}")
        seen.add(source.source_id)
    return sources
