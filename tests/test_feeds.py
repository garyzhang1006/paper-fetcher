from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from arxiv_kg.feeds import (
    MAX_FEED_BYTES,
    FeedFetchError,
    FeedParseError,
    fetch_feed,
    load_feed_sources,
    parse_feed,
)
from arxiv_kg.models import FeedSourceConfig


FIXTURES = Path(__file__).parent / "fixtures"
RETRIEVED_AT = datetime(2026, 8, 6, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(
        "arxiv_kg.feeds.socket.getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 1, 6, "", ("93.184.216.34", 443))
        ],
    )


def source(kind: str = "research_blog", max_items: int = 100) -> FeedSourceConfig:
    return FeedSourceConfig(
        source_id="test-source",
        name="Test Source",
        source_kind=kind,
        feed_url="https://example.org/feed.xml",
        homepage_url="https://example.org/",
        max_items=max_items,
        default_topics=["machine learning"],
    )


def test_parse_rss_normalizes_content_identity_and_provenance():
    items = parse_feed(
        (FIXTURES / "research_blog.rss.xml").read_bytes(),
        source(),
        retrieved_at=RETRIEVED_AT,
    )

    assert len(items) == 2
    first = items[0]
    assert first.external_id == "post-42"
    assert first.item_id == "source_item:f9612c76ec07a4f3252c6fa4"
    assert first.title == "Graph agents for scientific discovery"
    assert first.authors == ["Ada Researcher"]
    assert first.tags == ["Machine Learning", "Agents"]
    assert first.source_topics == ["machine learning"]
    assert first.summary == "A concise source-provided summary."
    assert first.content_text == (
        "We describe a graph agent. Evidence stays linked to its source."
    )
    assert first.published_at == datetime(2026, 8, 5, 14, 30, tzinfo=UTC)
    assert first.updated_at is None
    assert first.retrieved_at == RETRIEVED_AT
    assert first.raw_metadata == {
        "entry_format": "rss",
        "feed_url": "https://example.org/feed.xml",
        "guid": "post-42",
    }
    assert len(first.content_sha256) == 64
    assert items[1].external_id == "https://research.example.org/blog/url-identity"
    assert items[1].title is None


def test_parse_atom_normalizes_social_item_and_utc_times():
    item = parse_feed(
        (FIXTURES / "social.atom.xml").read_bytes(),
        source("social_media"),
        retrieved_at=RETRIEVED_AT,
    )[0]

    assert item.source_kind == "social_media"
    assert item.external_id == "tag:example.social,2026:post-9"
    assert item.canonical_url == "https://example.social/posts/9"
    assert item.authors == ["ML Community"]
    assert item.tags == ["benchmark"]
    assert item.summary == "Discussion of a new benchmark."
    assert item.content_text == (
        "Participants compare reported results and limitations."
    )
    assert item.published_at == datetime(2026, 8, 5, 15, tzinfo=UTC)
    assert item.updated_at == datetime(2026, 8, 5, 16, tzinfo=UTC)


def test_content_hash_ignores_retrieval_time_but_changes_with_content():
    content = (FIXTURES / "social.atom.xml").read_bytes()
    first = parse_feed(content, source("social_media"), retrieved_at=RETRIEVED_AT)[0]
    later = parse_feed(
        content,
        source("social_media"),
        retrieved_at=datetime(2026, 8, 7, 12, tzinfo=UTC),
    )[0]
    changed = parse_feed(
        content.replace(b"Participants compare", b"Researchers compare"),
        source("social_media"),
        retrieved_at=RETRIEVED_AT,
    )[0]

    assert first.item_id == later.item_id == changed.item_id
    assert first.content_sha256 == later.content_sha256
    assert first.content_sha256 != changed.content_sha256


def test_duplicate_tags_do_not_create_false_content_update():
    content = (FIXTURES / "research_blog.rss.xml").read_bytes()
    duplicated = content.replace(
        b"<category>Agents</category>",
        b"<category>Agents</category><category>agents</category>",
    )

    first = parse_feed(content, source(), retrieved_at=RETRIEVED_AT)[0]
    duplicate = parse_feed(duplicated, source(), retrieved_at=RETRIEVED_AT)[0]

    assert duplicate.tags == ["Machine Learning", "Agents"]
    assert duplicate.content_sha256 == first.content_sha256


def test_parser_rejects_saturated_feed_without_silent_truncation():
    with pytest.raises(FeedParseError, match="checkpoint was not advanced"):
        parse_feed(
            (FIXTURES / "research_blog.rss.xml").read_bytes(),
            source(max_items=1),
            retrieved_at=RETRIEVED_AT,
        )


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"<rss><channel><item></item></channel></rss>", "neither a GUID nor"),
        (b"<rss><channel></channel></rss>", "contains no items"),
        (b"<html></html>", "unsupported feed root"),
        (b"<rss>", "malformed XML"),
        (b"<!DOCTYPE rss><rss><channel/></rss>", "document type declarations"),
    ],
)
def test_parser_rejects_incomplete_or_unsafe_feed(payload, message):
    with pytest.raises(FeedParseError, match=message):
        parse_feed(payload, source(), retrieved_at=RETRIEVED_AT)


def test_parser_rejects_oversized_response():
    with pytest.raises(FeedParseError, match="maximum response size"):
        parse_feed(b"x" * (MAX_FEED_BYTES + 1), source(), retrieved_at=RETRIEVED_AT)


def test_parser_rejects_ambiguous_retrieval_timezone():
    with pytest.raises(FeedParseError, match="retrieved_at must include a timezone"):
        parse_feed(
            (FIXTURES / "research_blog.rss.xml").read_bytes(),
            source(),
            retrieved_at=datetime(2026, 8, 6, 12),
        )


def test_source_configuration_requires_https_and_unique_ids(tmp_path):
    with pytest.raises(ValidationError, match="must use HTTPS"):
        FeedSourceConfig(
            source_id="bad",
            name="Bad",
            source_kind="research_blog",
            feed_url="http://example.org/feed",
        )
    for private_url in (
        "https://127.0.0.1/feed",
        "https://169.254.169.254/latest/meta-data/",
        "https://localhost/feed",
    ):
        with pytest.raises(ValidationError, match="public host"):
            FeedSourceConfig(
                source_id="bad",
                name="Bad",
                source_kind="research_blog",
                feed_url=private_url,
            )

    config = tmp_path / "sources.json"
    config.write_text(
        '{"sources": ['
        '{"source_id":"same","name":"One","source_kind":"research_blog",'
        '"feed_url":"https://example.org/one"},'
        '{"source_id":"same","name":"Two","source_kind":"social_media",'
        '"feed_url":"https://example.org/two"}'
        ']}'
    )
    with pytest.raises(ValueError, match="duplicate source_id"):
        load_feed_sources(config)


def test_checked_in_source_config_covers_all_required_source_kinds():
    root = Path(__file__).parents[1]
    sources = load_feed_sources(root / "config/sources.json")
    assert {item.source_kind for item in sources if item.enabled} == {
        "research_blog",
        "research_report",
        "social_media",
    }


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.content = content
        self.status_code = status_code
        self.url = url or "https://example.org/feed.xml"
        self.headers = headers or {"Content-Type": "application/rss+xml"}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class FakeSession:
    def __init__(self, response: FakeResponse | Exception | list[FakeResponse]):
        self.responses = response if isinstance(response, list) else [response]
        self.requests = []

    def get(self, url: str, **kwargs):
        self.requests.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_fetch_feed_uses_bounded_request_and_parses_response():
    session = FakeSession(FakeResponse((FIXTURES / "research_blog.rss.xml").read_bytes()))
    items = fetch_feed(source(), retrieved_at=RETRIEVED_AT, session=session)

    assert len(items) == 2
    assert session.requests[0][0] == "https://example.org/feed.xml"
    assert session.requests[0][1]["timeout"] > 0
    assert "paper-fetcher" in session.requests[0][1]["headers"]["User-Agent"]
    assert session.requests[0][1]["stream"] is True
    assert session.requests[0][1]["allow_redirects"] is False


def test_fetch_feed_rejects_http_failure_and_bad_redirect():
    with pytest.raises(FeedFetchError, match="fetch failed"):
        fetch_feed(
            source(),
            retrieved_at=RETRIEVED_AT,
            session=FakeSession(FakeResponse(b"", status_code=429)),
        )

    with pytest.raises(FeedFetchError, match="destination must use HTTPS"):
        fetch_feed(
            source(),
            retrieved_at=RETRIEVED_AT,
            session=FakeSession(
                FakeResponse(
                    (FIXTURES / "research_blog.rss.xml").read_bytes(),
                    url="http://example.org/feed.xml",
                )
            ),
        )


def test_redirect_is_validated_before_second_request():
    session = FakeSession(
        FakeResponse(
            b"",
            status_code=302,
            headers={"Location": "https://127.0.0.1/private"},
        )
    )

    with pytest.raises(FeedFetchError, match="public IP addresses"):
        fetch_feed(source(), retrieved_at=RETRIEVED_AT, session=session)

    assert len(session.requests) == 1
