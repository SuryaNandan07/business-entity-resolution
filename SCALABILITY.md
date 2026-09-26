# Phase 4A.1 full training catalog retrieval

Entry points (run from the repository root with the project dependencies):

```powershell
python -m src.profile_catalog
python -m src.scalable_holdout2 verify
python -m src.scalable_holdout2 candidates
python -m src.scalable_holdout2 run
python -m unittest discover -s tests -v
```

The existing local `venv` launcher references a missing Python 3.11 installation.
For this session, the bundled Codex Python runtime and `.runtime-deps` were used.
Set `PYTHONPATH` to `.runtime-deps` for that runtime. The evaluation entry point
sets native computation to at most four threads before importing model libraries.

## Storage and semantics

`output/cache/train_catalog/records.sqlite` contains all normalized S2/S3 training
records, read sequentially in chunks of 10,000. No full target DataFrame is used.
An integer primary key maps to one entity ID and its record. SQLite's page cache
is bounded at 32 MiB. All persistent intermediates are on the project's drive.

Postings use unsigned 32-bit arrays. A batch cache contains only keys requested
by that batch's queries, but each key is counted against every catalog record.
Main blocking buckets are never truncated. Rescue postings are discarded when
their exact document count exceeds the original 200/300/200 limits; subsequent
occurrences continue contributing to diagnostic counts. Query trigram selection
uses the original `(posting_count, trigram)` ordering and four-key limit.

This is an exact query-driven index, not a universal all-key posting index.
The normalized catalog is reusable for every query batch. A new key set requires
one sequential scan of that cache; an identical key set reuses its posting cache.
Neither key selection nor ranking uses ground-truth labels.

Metadata records source paths, file sizes, nanosecond modification times, schema,
normalization/retrieval code hashes, configuration, UTC creation time and build
timing. Completed caches publish atomically; partial builds cannot be loaded.
Source hashes are not computed for the approximately 1 GB of input TSVs; size
and mtime checks are therefore not protection against deliberate same-stat edits.
Pickle files are trusted local caches and must not be imported from other sources.
All caches and generated reports are excluded by the existing `output/` rule.

## Resource and evaluation controls

The Windows watchdog checks resident memory, private committed memory and system
headroom every 200 ms. It terminates the process at the conservative 5.5 GiB
process limit or below 2 GiB available system memory. This sampling is a safeguard,
not an operating-system-enforced hard allocation quota. Checks also run at chunk
and query boundaries. Large stages execute sequentially.

Holdout 2 is exactly S1 rows 60,001–80,000 and is checked against the first 60,000
S1 IDs plus the saved Phase 3A split. This corrects the older Holdout 2 overlap
check, which omitted Holdout 1. Both models consume the same persisted Top-50
features, with frozen thresholds 0.670 and 0.585. No fitting is performed.

Per-query progress is committed to an internal SQLite checkpoint, allowing the
retrieval run to resume. Ranking retains only the best 50 candidates and bounds
prepared-record caching to 2,000 records. Inference reads 10,000 pairs at a time.
There is no competition-file access or final submission output.

The unchanged Python ranker is also profiled on initial full-catalog queries.
For accelerated ranking, an additional query-key cache stores unrestricted
name-token, address-token, name-bigram and exact-name integer postings, plus
per-record frozen IDF totals and bigram counts. Vector arithmetic screens scores;
every contender within a conservative floating-point error band of the 50th
score is retained. The original Python scorer then computes all final scores and
entity-ID tie breaks. Approximate scores never become model features. The band
includes rounding to 12 decimal places and is at least 1e-9, much larger than
ordinary positive-sum roundoff. Ranking-index fingerprints include the frozen
frequency file and scoring implementation. Feature checkpoints also invalidate
when those inputs change.

Ranking postings are built in committed 10,000-record SQLite segments and then
flattened into a binary file with key offsets. Totals are written through bounded
memory maps. The completed reader loads one requested posting block at a time;
it does not retain the full ranking posting corpus. Construction resumes from
the last committed chunk after interruption. The earlier in-memory ranking-cache
attempt was stopped by the system-headroom guard at roughly 1.44 GiB resident
memory; it never approached the 6 GB Python limit. Its failure is recorded under
`output/scalability/resource_stop.json` and the memory-sample logs.

## Reports

Reports under `output/scalability/` distinguish the bounded reference profile
from full-catalog optimized measurements. Reference profiling uses 100,000
targets and 100 queries; isolated rescue timings are diagnostic replays, not
additive pipeline stages. The equivalence check uses the first 200 development
S1 rows, all of their true target matches and 20,000 distractors per source.

The 1.7M-query estimate extrapolates measured Holdout 2 throughput. It excludes
new-batch posting construction and is not a guaranteed competition runtime.
Model adoption remains a review decision; HGB is not automatically replaced.

## Current inference block

Use the `candidates` stage while waiting for an approved Python environment.
It resumes valid indexes and per-S1 checkpoints, generates the unchanged Top-50
features, measures candidate recall only after selection finishes, and writes
`output/scalability/holdout2_candidates.json` without loading either model.
Windows Application Control blocked scikit-learn's native `arrayfuncs` module.
The user has explicitly deferred frozen-model inference; do not run `run` until
an approved environment is supplied. The completed candidate/feature checkpoint
can then be reused for both frozen models without repeating retrieval.

`python -m src.scalability_report` assembles the candidate-only review report.
Model metrics and inference timings remain unavailable, and HGB remains selected.
