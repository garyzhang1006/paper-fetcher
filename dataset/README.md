# arXiv submissions dataset, 2026-07-03 to 2026-07-15

This folder contains a curated 9,000-paper sample from all arXiv API records
matching UTC `submittedDate` windows from 2026-07-03 through 2026-07-15,
inclusive.

Files:

- `papers.jsonl`: one JSON object per paper.
- `daily_counts.csv`: daily paper counts.
- `metadata.json`: source URL, query template, range, total count, and fetch time.
- `fetch_arxiv_dataset.py`: reproducible fetch script.

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
selection deterministic for an unchanged source snapshot. This is a metadata
quality filter, not a judgment of scientific validity.

`daily_counts.csv` reports retained papers. `metadata.json` records source,
retained, and removed totals plus the exact curation method.
