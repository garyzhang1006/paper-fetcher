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

## Manual GitHub Actions update

Scheduled fetching is disabled. `.github/workflows/daily-arxiv-fetch.yml` runs
only when manually started with GitHub Actions `workflow_dispatch`. It installs
the package, runs the full offline test suite, performs one bounded fetch, then
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

Extract evidence-backed validity envelopes for the curated 7,751-paper dataset:

```bash
paper-fetcher-validity --expected-count 7751
```

Output is stored in date shards under `dataset/validity_envelopes/`. This pass
uses abstracts only, so unavailable table, page, seed, and compute-budget fields
remain empty rather than being guessed. Boundary statements are labeled
paper-level unless the abstract explicitly ties them to a claim. Absolute
reported values stay separate from comparative effect sizes.

## Primary-category classifier

Train a CPU neural network that predicts each paper's `primary_category` from
its title and abstract:

```bash
python -m pip install '.[ml]'
paper-fetcher-classify
```

The classifier follows the attached PyTorch classification tutorial's full
workflow. It fits word and character TF-IDF features on training papers only,
uses weighted cross-entropy on raw logits, chooses the best epoch using a
validation set, and evaluates the test set once. Fixed seeds make CPU split and
training order reproducible; accelerator kernels can still vary. Categories
with fewer than five examples are excluded by default because they cannot
support meaningful train, validation, and test subsets.

See [`classification/PSEUDOCODE.md`](classification/PSEUDOCODE.md) for the
complete data, training, evaluation, artifact, and prediction flow.

Outputs are written to `data/category_classifier/`:

- `metrics.json`: loss, accuracy, macro-F1, weighted F1, top-3 accuracy,
  calibration error, per-category metrics, confidence-filtered accuracy,
  confusion pairs, and high-confidence mistakes;
- `learning_curves.png`: training and validation curves;
- `model.pt`: best validation-selected PyTorch weights;
- `vectorizer.pkl`: fitted text feature pipeline; and
- `labels.json`: output-index to arXiv-category mapping.

Classify a new paper with saved artifacts:

```bash
paper-fetcher-predict \
  --title "Paper title" \
  --abstract "Paper abstract"
```

Only load `model.pt` and `vectorizer.pkl` artifacts produced by a trusted run.
Confidence is model output, not certainty. Test scores estimate performance on
papers similar to this dataset; they do not prove perfect classification on new
domains or missing categories.
