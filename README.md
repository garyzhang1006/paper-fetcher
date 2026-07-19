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
Its defaults match Notebook 1's bounded live example: `cs.LG`, a 24-hour
lookback, a 200-paper submission cap, and a 200-paper revision cap.

Paper metadata and pipeline checkpoints are stored in
`data/arxiv_kg.sqlite3`. Override the path or port when needed:

```bash
paper-fetcher --db /path/to/papers.sqlite3 --port 9000
```

## Run one bounded update

The noninteractive command uses the notebook's safe live defaults: `cs.LG`, a
24-hour first-run lookback, a 24-hour overlap, and separate 200-result caps for
new submissions and revisions.

```bash
paper-fetcher-daily
```

Options can be changed explicitly without editing code:

```bash
paper-fetcher-daily \
  --db data/arxiv_kg.sqlite3 \
  --category cs.LG \
  --category cs.RO \
  --max-results 200 \
  --revision-max-results 200
```

The command exits nonzero if an API page fails or either query exceeds its cap.
Rows saved before failure are safe because upserts are idempotent, while atomic
checkpoints remain at the last complete run.

## Daily GitHub Actions update

`.github/workflows/daily-arxiv-fetch.yml` runs every day at 06:17 in the
`America/Chicago` IANA timezone and also supports manual runs. It installs the
package, runs the full offline test suite, performs one bounded fetch, then
commits `data/arxiv_kg.sqlite3` only after success. Repository Actions settings
must allow `GITHUB_TOKEN` write access for the final push.

The persisted checkpoint covers skipped days because the next query begins at
the last successful checkpoint minus the overlap. A capped or failed run does
not move either checkpoint. Checkpoints are scoped to each category set, so
adding a category uses its full first-run lookback instead of inheriting another
category's checkpoint. The revision query is sorted by last-updated time, so a
new version of an older paper is refreshed even when its original submission is
outside the new-paper window.

## Tests

```bash
pytest
```

Package source lives in `src/arxiv_kg/`. The fetcher stores paper metadata and
idempotent checkpoints in SQLite, while `download_pdf` validates the arXiv host
and PDF header before replacing the destination file.

## Tutorial 2: feature extraction

`notebooks/02_feature_extractor.ipynb` is a complete offline tutorial for
turning stored papers into typed semantic features. It includes a deterministic
rules baseline, gold-label precision/recall/F1 evaluation, evidence coverage,
PDF text extraction, section-aware prompt selection, experimental validity
envelopes, extractor versioning, and all exercise solutions.

Install notebook dependencies and start JupyterLab:

```bash
python -m pip install '.[test,notebooks]'
jupyter lab notebooks/02_feature_extractor.ipynb
```

The optional OpenAI backend is disabled in the notebook. To install its SDK:

```bash
python -m pip install '.[llm]'
```

The rules backend requires no API key or network access. Feature JSON is stored
with source paper version, extractor version, and prompt version so changed
papers or extraction configurations are selected for reprocessing.

Extract evidence-backed validity envelopes for the curated 8,406-paper dataset:

```bash
paper-fetcher-validity --expected-count 8406
```

Output is stored in date shards under `dataset/validity_envelopes/`. This pass
uses abstracts only, so unavailable table, page, seed, and compute-budget fields
remain empty rather than being guessed. Boundary statements are labeled
paper-level unless the abstract explicitly ties them to a claim. Absolute
reported values stay separate from comparative effect sizes.
