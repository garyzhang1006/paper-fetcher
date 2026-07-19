# arXiv submissions dataset, 2026-07-03 to 2026-07-15

This folder contains a curated 8,406-paper sample from all arXiv API records
matching UTC `submittedDate` windows from 2026-07-03 through 2026-07-15,
inclusive. It began as a 9,000-paper metadata-quality sample, then all papers
whose primary category was in Astrophysics were removed.

Files:

- `papers.jsonl`: one JSON object per paper.
- `daily_counts.csv`: daily paper counts.
- `metadata.json`: source URL, query template, range, total count, and fetch time.
- `fetch_arxiv_dataset.py`: reproducible fetch script.
- `validity_envelopes/`: one abstract-level validity record per paper, split
  into daily JSONL shards with a hashed manifest.

The script uses one UTC day query at a time:

```text
submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]
```

It compares parsed paper records with arXiv's `opensearch:totalResults` for
each day and exits with an error if a day is incomplete.

After validating the complete API response, the script ranks papers by
observable metadata quality. Withdrawn or retracted markers rank lowest,
followed by abstract depth, optional publication metadata, sensible title
length, and category tagging. Proportional primary-category quotas retain at
least one paper from every category. A stable arXiv-ID hash breaks ties, making
selection deterministic for an unchanged source snapshot. The final pass removes
all papers whose primary category is in Astrophysics. This is a metadata quality
filter, not a judgment of scientific validity.

`daily_counts.csv` reports retained papers. `metadata.json` records source,
retained, and removed totals plus the exact curation method.

## Experimental validity envelopes

Run `paper-fetcher-validity --expected-count 8406` from the repository root to
rebuild `validity_envelopes/`. The extractor preserves explicit result claims
and records supported comparators, contexts, metrics, numeric values, effect
sizes, uncertainty, conditions, and boundary statements. Papers without an explicit abstract-level
result claim receive `no_supported_claim` and an empty envelope list.

Boundary statements are labeled paper-level because proximity in an abstract
does not prove that a limitation applies to one particular claim. Confidence is
an extraction-support heuristic, not the probability that a claim is true.

No full-text PDFs are stored for these 8,406 papers. The output therefore does
not claim table, figure, page, seed, or compute-budget evidence.
