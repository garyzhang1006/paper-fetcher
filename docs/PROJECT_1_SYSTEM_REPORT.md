# Project 1 system report

Verified August 6, 2026 in the repository's tracked SQLite database.

## Requirement coverage

| Project requirement | Implemented behavior | Observed evidence |
|---|---|---|
| Collect new research every day | One `paper-fetcher-agent` command collects arXiv plus every enabled public RSS or Atom feed. GitHub Actions runs it daily at 13:17 UTC and supports manual dispatch. | Live arXiv forward run and all three public feeds completed. |
| Include arXiv, research blogs, reports, and social media | arXiv uses the existing version-aware API client. Feed adapters normalize research articles and public social posts with source identity, URL, timestamps, content hash, and raw provenance. Explicit report-like Hugging Face titles become report nodes; other entries become blog posts. | 100 DeepMind items, 835 Hugging Face items, and 25 Reddit r/MachineLearning posts are stored. |
| Represent papers, reports, and topics as nodes | Graph document nodes are `paper`, `report`, `blog_post`, and `social_post`. Concept nodes are `topic`, `method`, `research_goal`, `dataset`, `metric`, and `category`. | Active graph contains 7,794 papers, 29 reports, 906 blog posts, 25 social posts, and 197 concept/category nodes. |
| Connect shared topics, methods, and research goals | Extracted `ABOUT_TOPIC`, `USES_METHOD`, and `PURSUES_GOAL` edges require evidence. Configured feed topics and arXiv categories use explicit source metadata. Dataset and metric relations are also typed. `RELATED_TO` uses bounded IDF-weighted supported-concept overlap. | Active graph contains 59,256 edges across all seven relation types, and no document node is isolated. |
| Summarize each document accurately | Default rules extractor copies the first abstract or feed-summary sentence. It does not invent a new claim. Summary provenance includes source version and extractor version. | 7,794 current paper feature rows use rules extractor 1.4. Feed summaries were extracted during graph build. |
| Place documents in the right graph area | Methods, goals, datasets, metrics, and extracted domains require exact evidence. ArXiv categories and configured feed topics use source metadata. Free keywords never create semantic edges. | Ontology and evidence tests pass. `/api/placement?id=...` returns document, typed relationships, and neighboring nodes; all 8,754 document nodes have at least one edge. |
| Find hubs and clusters | Analysis computes weighted PageRank, degree, weighted degree, and connected components over `RELATED_TO` edges scoring at least 0.35. Cluster labels use supported concepts. | Bounded hub and cluster artifacts were generated from active build. |
| Track hot and emerging topics | Equal recent and baseline windows use mean per-source document shares. Emerging score requires at least three recent documents and positive smoothed log growth. | Trend status is `ok` with seven-day windows ending August 6, 2026. No topic met the three-document emerging threshold in the active mixed-source snapshot. |
| Make research easy to understand, search, and explore | Local UI combines paper search, document placement explanation, hot topics, emerging topics, hubs, and clusters. Five bounded JSON APIs expose same data. | HTTP integration tests cover graph empty state, placement, active build metadata, and bound validation. |

## Active graph snapshot

- Build ID: `95ac2bfcd067acf6cf2b7ce03fd686920ebfb14a95b3364d0c38095bcbae00c2`
- Documents: 8,754
- Nodes: 8,951
- Edges: 59,256
- Trend status: `ok`
- Durable database: `data/arxiv_kg.sqlite3`
- Bounded summaries: `output/knowledge_graph/`

### Node counts

| Node type | Count |
|---|---:|
| paper | 7,794 |
| report | 29 |
| blog post | 906 |
| social post | 25 |
| category | 148 |
| dataset | 11 |
| method | 11 |
| metric | 10 |
| research goal | 10 |
| topic | 7 |

### Edge counts

| Relation | Count |
|---|---:|
| `RELATED_TO` | 37,656 |
| `IN_CATEGORY` | 13,180 |
| `ABOUT_TOPIC` | 3,056 |
| `PURSUES_GOAL` | 1,972 |
| `REPORTS_METRIC` | 1,624 |
| `USES_METHOD` | 1,636 |
| `EVALUATES_ON` | 132 |

## Daily operation and recovery

Each feed commits normalized items and its success checkpoint in one SQLite
transaction. Graph replacement is also atomic. A failed graph build leaves the
previous snapshot and analysis readable.

The first August 6 arXiv smoke run found a 3,021-paper backlog and stopped at
the configured 200-paper safety cap. It did not advance the checkpoint. Because
this repository intentionally keeps a bounded corpus, both `cs.LG` checkpoints
were then rebased to August 6 instead of importing the full backlog. The tracked
database records this operator decision and its reason. A forward 24-hour
overlap run then completed successfully with zero new submissions and no missed
checkpoint transition. Papers not already stored from July 13 through the
rebase time were deliberately skipped.

## Interpretation limits

Public RSS or Atom coverage does not represent all research blogs or all social
media. Reddit coverage is one public subreddit feed. Authenticated and private
sources are outside current scope.

The rules extractor recognizes a documented, finite vocabulary. An absent edge
can mean the concept was absent or outside that vocabulary. Extractive summaries
avoid unsupported generated claims, but no human-rated summary quality study has
been run. Trend labels describe measured source-normalized corpus movement. They
do not establish research quality, causal importance, or future impact.

The optional OpenAI extractor remains available for structured extraction, but
the active graph and counts in this report use the deterministic rules backend.
