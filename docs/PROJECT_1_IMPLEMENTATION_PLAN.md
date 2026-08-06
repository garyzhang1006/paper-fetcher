# Project 1 implementation plan

## Observable success

Project is complete only when current repository can:

1. Ingest arXiv papers, research-blog or report feeds, and a public social-media feed.
2. Run one scheduled daily command that collects, extracts, rebuilds graph, analyzes it, and persists results.
3. Populate typed paper, report, social-post, topic, method, research-goal, dataset, metric, and category nodes.
4. Store evidence-backed semantic edges and bounded related-document edges.
5. Return document placement, hubs, clusters, hot topics, and emerging topics through API and UI.
6. Bootstrap all 7,751 curated papers from `dataset/papers.jsonl` without loading file into memory.
7. Rebuild graph deterministically, pass full test suite, and publish matching source, artifacts, and documentation.

## Frozen interfaces

### Source ingestion

`FeedSourceConfig` fields:

- `source_id`, `name`, `source_kind`
- `feed_url`, optional `homepage_url`
- `enabled`, `max_items`, and bounded source-declared `default_topics`

Supported source kinds are `research_blog`, `research_report`, and `social_media`.

`SourceItem` fields:

- stable `item_id` from `source_id` plus feed GUID or canonical URL
- source identity and kind
- optional title, authors, tags, and source-provided summary
- content text, canonical URL, publication/update/retrieval times
- content SHA-256 and raw provenance metadata

Feed ingestion accepts bounded HTTPS RSS or Atom only. Missing optional values remain empty. Missing both feed GUID and canonical URL rejects item. Each feed owns an independent success checkpoint.

### Graph ontology

Document node types:

- `paper`, `report`, `blog_post`, `social_post`

Concept node types and relations:

- domain to `topic` through `ABOUT_TOPIC`
- research task to `research_goal` through `PURSUES_GOAL`
- method through `USES_METHOD`
- dataset through `EVALUATES_ON`
- metric through `REPORTS_METRIC`
- arXiv category through `IN_CATEGORY`

Semantic edges require extractor evidence, except arXiv categories and
configured feed topics whose basis is source metadata. Free keywords never
become graph edges.

Related-document edges use IDF-weighted overlap from supported concepts. Candidate pairs come from inverted postings, concepts present in more than 10 percent of documents are suppressed, threshold is 0.20, and each document keeps at most 10 neighbors.

### Graph analysis

- Hubs: degree, weighted degree, and deterministic PageRank.
- Clusters: connected components over `RELATED_TO` edges with score at least 0.35.
- Cluster labels: top supported concepts by IDF-weighted member frequency.
- Trends: per-source-normalized unique-document shares across equal recent and baseline windows.
- Emerging topics: at least three recent documents plus positive smoothed log growth.
- Static datasets use maximum document publication date as `as_of`; insufficient history returns an explicit status.

### Persistence and API

Graph rebuild is atomic. Failed rebuild leaves prior snapshot readable. Durable outputs are SQLite plus bounded JSON summaries under `output/knowledge_graph/`.

Required endpoints:

- `GET /api/graph`
- `GET /api/placement`
- `GET /api/hubs`
- `GET /api/clusters`
- `GET /api/trends`

All limits are bounded and validated. Unknown documents return 404.

## Verified public sources

- Google DeepMind RSS: research blog
- Hugging Face blog RSS: technical reports and articles
- Reddit r/MachineLearning Atom feed: public social-media source

Authenticated social APIs and private feeds are outside current scope. Documentation must call this public-feed coverage, not universal social-media collection.

## Verification gates

- RSS and Atom fixture tests, malformed and oversized response tests, checkpoint failure tests
- database insert/update/idempotence tests
- graph ontology, evidence, stale-edge, collision, and atomic-rebuild tests
- bounded-similarity, hub, PageRank, cluster, and trend fixture tests
- API validation and integration tests
- clean 7,751-paper bootstrap with nonzero paper/topic/method/goal nodes and edges
- repeated build produces same semantic graph and analysis
- daily workflow has cron plus manual dispatch and runs tests before network
- final full suite passes after last change
