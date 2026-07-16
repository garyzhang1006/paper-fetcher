# arXiv submissions dataset, 2026-07-03 to 2026-07-15

This folder contains arXiv API metadata for all papers matching UTC
`submittedDate` windows from 2026-07-03 through 2026-07-15, inclusive.

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

The July 15 count is a same-day snapshot taken at the `fetched_at` timestamp in
`metadata.json`; arXiv may add more submissions after that time.
