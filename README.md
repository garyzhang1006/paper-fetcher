# arXiv Paper Fetcher

Tutorial 1 source for fetching arXiv metadata, storing revisions in SQLite,
validating the first knowledge-graph data models, and browsing saved papers in
a local web UI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[test]'
```

## Run the UI

```bash
paper-fetcher
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The UI lets you choose
arXiv categories, set a first-run lookback window, fetch revisions, search
saved titles/authors/abstracts, and filter the local library by category.

Paper metadata and pipeline checkpoints are stored in
`data/arxiv_kg.sqlite3`. Override the path or port when needed:

```bash
paper-fetcher --db /path/to/papers.sqlite3 --port 9000
```

## Tests

```bash
pytest
```

Package source lives in `src/arxiv_kg/`. The fetcher stores paper metadata and
idempotent checkpoints in SQLite, while `download_pdf` validates the arXiv host
and PDF header before replacing the destination file.
