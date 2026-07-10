# arXiv Paper Fetcher

Tutorial 1 source for fetching arXiv metadata, storing revisions in SQLite,
and validating the first knowledge-graph data models.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Tests

```bash
pytest
```

Package source lives in `src/arxiv_kg/`. The fetcher stores paper metadata and
idempotent checkpoints in SQLite, while `download_pdf` validates the arXiv host
and PDF header before replacing the destination file.
