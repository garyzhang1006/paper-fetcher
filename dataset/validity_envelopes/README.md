# Abstract-level experimental validity envelopes

Each date-sharded JSONL file contains one record for every source paper from
that UTC submission date. Join records to `../papers.jsonl` with `arxiv_id` and
`source_versioned_id`.

An envelope preserves an explicit result claim plus any comparator, evaluation
context, metric, reported numeric value, effect size, uncertainty, or condition found in the claim.
Explicit limitations elsewhere in the abstract are labeled as paper-level
boundaries. Empty fields mean the abstract did not state that detail.
`no_supported_claim` means the conservative rules found no explicit result
claim; it does not mean the paper has no results.

Confidence measures how many supported detail types were extracted. It is not a
calibrated probability and does not estimate whether the scientific claim is true.

These records do not claim table, figure, page, seed, or compute-budget evidence
because the repository does not contain the 9,000 full-text PDFs. See
`manifest.json` for counts, hashes, extractor version, and source provenance.

Regenerate from the repository root:

```bash
paper-fetcher-validity --expected-count 9000
```
