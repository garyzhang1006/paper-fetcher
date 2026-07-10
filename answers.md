ARXIV FETCHER  |  NOTEBOOK 01 ANSWER KEY

Answers renumbered to the notebook
Each heading below matches the section number and title in 01_daily_arxiv_fetcher.ipynb.
A question is placed under the notebook section where it appears. Answers state repository
behavior first, then cite exact source or test lines. Tied to repository commit c700846;
no live arXiv request was needed.


1. The fetcher is a stateful agent component

1.1  What is new or revised since the last successful run, including overlap, and can it be
     stored without duplication?

The daily fetcher reads the submission and revision checkpoints for the selected categories,
backs up by the safety overlap, and searches forward to the current UTC time. The submittedDate
field surfaces newly posted papers, and the LastUpdatedDate field surfaces older papers that
were revised today. Every result is validated into a typed PaperRecord keyed on its base arXiv
ID, so a repeated version resolves to the same row with no change and a newer version overwrites
that row's revision fields in place.

Repo evidence: src/arxiv_kg/fetcher.py:38-42, 51-68, 71-96, 99-146, 189-226;
src/arxiv_kg/models.py:17-39; src/arxiv_kg/db.py:112-180


2. Parse paper versions

2.1  Why would using 2107.05580v3 as the primary key cause trouble when v4 appears?

Keying on the versioned ID would register 2107.05580v3 and 2107.05580v4 as two independent
papers, producing two rows for one logical work and leaving any reference to v3 pointing at
superseded content. The repository trims the trailing v and its digits, stores 2107.05580 as
the arxiv_id primary key, and records the concrete v3 or v4 label in versioned_id, which keeps
every revision of the paper collapsed onto a single row.

Repo evidence: src/arxiv_kg/ids.py:9-19; src/arxiv_kg/models.py:17-26; src/arxiv_kg/db.py:43-46;
tests/test_fetcher.py:58-61, 64-82


7. Run the offline fetcher tests

7.1  Why does the saturated-query test keep the first rows but expect no checkpoint?

The first max_results papers are genuine results, so the fetcher commits them to the database as
they stream in. The presence of one extra record beyond the cap signals that the submitted-date
interval was truncated mid-window, so the fetcher raises at that point and holds the checkpoint at
its previous value. On the following run the query reopens from that same checkpoint and re-reads
the already-stored rows, which the idempotent upsert absorbs without creating duplicates.

Repo evidence: src/arxiv_kg/fetcher.py:160-187, 223-226; src/arxiv_kg/db.py:25-37;
tests/test_fetcher.py:95-118


9. Scheduling

9.1  What happens if the job is skipped for three days?

The next run reopens at the last successful checkpoint minus the overlap window and closes at the
current UTC time, so its interval spans all three skipped days in a single pass. When that widened
interval carries more papers than the configured cap, the run aborts and preserves the old
checkpoint for a later retry. When it fits under the cap, the fetcher stores every recovered paper
and advances the checkpoint once the pass completes.

Repo evidence: src/arxiv_kg/fetcher.py:125-146, 223-226; tests/test_fetcher.py:151-187;
README.md:67-73

9.2  What happens if the API returns more papers than the cap?

The query deliberately requests cap + 1 results so that a saturated page carries a detectable
sentinel row. When that extra paper materializes, the fetcher treats the interval as truncated,
retains every safely stored row, raises a RuntimeError, and holds both checkpoints at their prior
values. The revision scan applies the identical guard and relaxes it only when the surplus result
already predates its own cutoff.

Repo evidence: src/arxiv_kg/fetcher.py:148-187, 189-221; tests/test_fetcher.py:95-118, 317-343

9.3  What happens if papers save successfully but the final API page fails?

The papers written before the failure remain in the local SQLite database because every upsert
commits on its own transaction. The exception propagates before the final atomic checkpoint write,
so both checkpoints continue to reference the last fully successful run and the unfinished interval
is retried on the next invocation. Under GitHub Actions the failed job also declines to commit the
database back to main, which keeps the published state consistent with the recorded checkpoint.

Repo evidence: src/arxiv_kg/db.py:25-37, 112-180; src/arxiv_kg/fetcher.py:223-226;
tests/test_fetcher.py:121-148, 346-375; .github/workflows/daily-arxiv-fetch.yml:37-63

9.4  What happens when an older paper receives v2 today?

The submitted-date query can overlook the paper because its original submission date falls outside
the current window. The revision query covers that case by sorting on LastUpdatedDate and scanning
back to its own cutoff, so it catches the fresh v2 the day it posts. Once found, split_arxiv_version
resolves v2 to the same base ID, and the upsert rewrites the stored row's version number, title,
abstract, URLs, and timestamps to match the revision.

Repo evidence: src/arxiv_kg/fetcher.py:189-226; src/arxiv_kg/ids.py:12-19;
src/arxiv_kg/db.py:156-180; tests/test_fetcher.py:254-288


10. Codex prompt card

10.1  Explain the current checkpoint, overlap, and cap-sentinel behavior; is Step 10 complete?

Step 10 is complete in the current repo. The implementation and offline tests cover checkpoints,
overlap, the cap sentinel, and a three-day gap without touching the live arXiv API.

Checkpoint: submission and revision checkpoints are stored as UTC ISO timestamps in pipeline_state.
Their keys are scoped to an order-independent category set, and both are written in one transaction
after successful query completion.

Overlap: for an existing checkpoint, the next start time equals checkpoint minus overlap_hours. The
same rule is applied independently to the revision checkpoint.

Cap sentinel: each query asks for one more result than its configured cap. A recent extra row proves
truncation and raises before checkpoints move; an older revision sentinel proves the cutoff was
reached and is not a false saturation.

Offline gap test: test_three_day_gap_after_checkpoint_misses_nothing uses FakeClient. It runs on
day 0, skips days 1 and 2, runs on day 3, checks both gap papers, and confirms the checkpoint
advances only after the successful run.

Why missed intervals are prevented: only a complete, uncapped run can replace the old checkpoint.
Any cap, network failure, conversion error, or database error exits before set_states, so the next
run begins from the same old checkpoint minus overlap and retries the uncertain interval.

Repo evidence: src/arxiv_kg/fetcher.py:38-42, 99-226; src/arxiv_kg/db.py:79-83, 342-367;
tests/test_fetcher.py:37-55, 151-187, 291-343, 378-405; .github/workflows/daily-arxiv-fetch.yml:34-49


11. Exercises

11.1  Add cs.RO to a query and test the exact expression.

With the notebook's dates and categories, the repo's exact tested query is:

    ((cat:cs.LG OR cat:stat.ML OR cat:cs.RO) AND
     submittedDate:[202606230000 TO 202606240000])

Repo evidence: src/arxiv_kg/fetcher.py:45-62; tests/test_fetcher.py:218-228

11.2  Draw the row state after v1, duplicate v1, then v2.

All three operations resolve to the same base-ID row. Repeating v1 advances only the last_seen_at
timestamp while leaving the stored content intact, and the arrival of v2 rewrites the
revision-sensitive metadata and blanks the file paths that were derived from v1.

    Operation    Upsert result   Rows   Stored state
    ---------    -------------   ----   ------------
    Insert v1    inserted        1      arxiv_id=2606.90001; versioned_id=2606.90001v1; version=1
    Repeat v1    unchanged       1      Same logical row and metadata; last_seen_at refreshes
    Receive v2   updated         1      versioned_id becomes ...v2; version=2; revised fields
                                        replace v1; pdf_path/text_path=NULL

Repo evidence: src/arxiv_kg/db.py:112-180; tests/test_fetcher.py:64-82

11.3  Modify a fake result so v2 has a new title; verify the stored title changes.

The test assigns v2.title the value "Paper 1: Revised With a New Title", runs the fetch, and then
reads the row back from the database. That row now reports version 2 together with the exact revised
title, because the upsert refreshes stored metadata whenever versioned_id or updated_at moves forward.

Repo evidence: src/arxiv_kg/db.py:156-180; tests/test_fetcher.py:231-251

11.4  Explain why a PDF path must be cleared after a paper revision.

A path recorded for v1 continues to reference v1's rendered content. Leaving it in place once v2
arrives would let the database assert that a superseded PDF or text extraction represents the current
paper, which corrupts every downstream consumer that trusts the field. Setting pdf_path and text_path
to NULL forces the pipeline to fetch and reprocess v2 before any later stage reads the file.

Repo evidence: src/arxiv_kg/db.py:167-180; tests/test_fetcher.py:64-82

11.5  Design a log message that helps an instructor diagnose a capped run without exposing secrets.

The repo already has a useful, secret-free message. It names the limit that was hit, tells the
instructor what to change, and says exactly what happened to the checkpoint:

    ERROR The new-paper query reached --max-results. Increase the limit or
    reduce the lookback window; the checkpoint was not advanced.

The revision-saturation variant names --revision-max-results, notes that the time cutoff was not
reached, and confirms that the checkpoints stayed fixed at their prior values. Neither string carries
tokens, credentials, request headers, or paper text.

Repo evidence: src/arxiv_kg/fetcher.py:180-187, 217-221


Exit ticket (two sentences)

Overlap can resurface a paper the previous run already stored, yet the base arXiv ID serves as the
primary key, so the upsert resolves the repeat to an unchanged or updated result on the one existing
row. A run that fails or saturates the cap holds the checkpoint at its prior value, because the
following run must replay that same interval to recover every paper it might not have processed.

Repo evidence: src/arxiv_kg/db.py:43-46, 112-180; src/arxiv_kg/fetcher.py:111-119, 180-187, 217-226
