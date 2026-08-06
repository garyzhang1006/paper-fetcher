# Machine Learning Research Knowledge Graph

Daily research agent for collecting arXiv papers, public research feeds, and a
public machine-learning social feed. It extracts grounded summaries and typed
concepts, builds an evidence-backed knowledge graph, identifies hubs and
clusters, and measures hot or emerging topics over time.

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
saved titles/authors/abstracts, filter the local library by category, explain a
document's graph placement, and inspect hubs, clusters, hot topics, and emerging
topics.
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

## Run the complete research agent

The Project 1 command imports the curated corpus as a stream, collects recent
arXiv updates, reads every enabled public RSS or Atom source, extracts missing
paper features, rebuilds the graph atomically, computes graph analysis, and
writes bounded JSON summaries:

```bash
paper-fetcher-agent \
  --db data/arxiv_kg.sqlite3 \
  --dataset dataset/papers.jsonl \
  --sources config/sources.json \
  --output-dir output/knowledge_graph \
  --category cs.LG
```

Use `--offline` to skip network collection while still importing, extracting,
building, and analyzing. The complete command exits nonzero if arXiv or an
enabled feed fails. Each feed's items and success checkpoint commit together.
Successful sources still rebuild a degraded snapshot and write their state; the
report records each failed source. A failed graph rebuild leaves the previous
graph readable.

For bounded recovery, `--no-arxiv` runs configured feeds without changing the
arXiv checkpoint, while `--no-feeds` runs arXiv without public feeds. Scheduled
runs use neither switch and collect every configured source.

Configured public feeds are:

- Google DeepMind Blog RSS, stored as research blog posts;
- Hugging Face Blog RSS, whose explicit report-like titles become reports and
  whose other entries become blog posts; and
- Reddit r/MachineLearning Atom, stored as social posts.

The complete-feed limit is 1,000 entries per configured source. Exceeding that
limit fails loudly without advancing the source checkpoint; entries are never
silently truncated.

This is bounded public-feed coverage. It does not ingest private feeds,
authenticated social APIs, or every research blog.

## Daily GitHub Actions update

`.github/workflows/daily-arxiv-fetch.yml` runs every day at 13:17 UTC and also
supports manual `workflow_dispatch`. It installs the package, runs the full
offline test suite and Notebook 2, then runs `paper-fetcher-agent`. Only a
complete or degraded agent run commits valid `data/arxiv_kg.sqlite3` state and
bounded summaries under `output/knowledge_graph/`. A degraded run then fails
the workflow visibly. Repository Actions settings must allow
`GITHUB_TOKEN` write access for the final push.

The persisted checkpoint covers skipped days because the next query begins at
the last successful checkpoint minus the overlap. A capped or failed run does
not move either checkpoint. Checkpoints are scoped to each category set, so
adding a category uses its full first-run lookback instead of inheriting another
category's checkpoint. The revision query is sorted by last-updated time, so a
new version of an older paper is refreshed even when its original submission is
outside the new-paper window.

## Knowledge graph contract

Document nodes are `paper`, `report`, `blog_post`, or `social_post`. Extractor
evidence supports these concept relationships:

- domain to `topic` through `ABOUT_TOPIC`;
- research task to `research_goal` through `PURSUES_GOAL`;
- method through `USES_METHOD`;
- dataset through `EVALUATES_ON`; and
- metric through `REPORTS_METRIC`.

Paper category edges use arXiv source metadata through `IN_CATEGORY`.
Configured feed topics use source metadata through `ABOUT_TOPIC`, ensuring a
feed document has an auditable placement even when the fixed extractor
vocabulary finds no narrower concept. Free keywords never create semantic
edges. `RELATED_TO` uses IDF-weighted concept overlap, suppresses concepts found
in more than 10 percent of documents, requires score 0.20, and keeps at most 10
neighbors per document.

Hubs include degree, weighted degree, and deterministic PageRank. Clusters are
connected components using `RELATED_TO` score at least 0.35. Trend analysis
compares equal recent and baseline windows using per-source document shares, so
a high-volume feed cannot dominate by volume alone. Emerging topics require at
least three recent documents and positive smoothed log growth. The API and UI
show recent-versus-baseline counts and log2 growth for direct evolution checks.

Read graph outputs through:

- `GET /api/graph`
- `GET /api/placement?id=<paper-or-feed-item-id>`
- `GET /api/hubs`
- `GET /api/clusters`
- `GET /api/trends`

See [`docs/PROJECT_1_IMPLEMENTATION_PLAN.md`](docs/PROJECT_1_IMPLEMENTATION_PLAN.md)
for frozen interfaces and verification gates. Current live counts, requirement
coverage, checkpoint history, and interpretation limits are in the
[`Project 1 system report`](docs/PROJECT_1_SYSTEM_REPORT.md).

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
python3.13 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install '.[ml]'
paper-fetcher-classify
```

Python 3.13 is the tested ML runtime used by repository CI. On the current Mac,
Python 3.14.5 with PyTorch 2.13 imports unusually slowly, so keep classifier
work in the separate `.venv-ml` environment.

The classifier follows the attached PyTorch classification tutorial's full
workflow. It fits word and character TF-IDF features on training papers only,
uses weighted cross-entropy on raw logits, chooses the best epoch using a
validation set, and computes current-run test metrics after checkpoint
selection. Earlier development viewed results from the same test partition, so
it is not untouched across project history. Fixed seeds make CPU split and
training order reproducible; accelerator kernels can still vary. Categories
with fewer than five examples are excluded by default because they cannot
support meaningful train, validation, and test subsets.

See [`classification/PSEUDOCODE.md`](classification/PSEUDOCODE.md) for the
complete data, training, evaluation, artifact, and prediction flow. Current
results and error analysis are in the
[`classification benchmark`](classification/BENCHMARK.md), the
[`four-page report`](output/pdf/arxiv-classification-progress-report.pdf), and
the
[`executed evidence notebook`](output/jupyter-notebook/arxiv-category-classifier-report.executed.ipynb).

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
