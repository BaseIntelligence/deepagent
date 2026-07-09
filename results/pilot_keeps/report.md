# SWE-Forge Benchmark Report

- Shipped tasks: **5**
- Overall: **PASS**

## Headline A - gold solvability
- gold = **100%** (5/5) -- PASS
- deterministic across reruns: True

## Headline B - frontier solve-rate
- stated threshold: **0.5100**
- measured frontier solve-rate: **0.2167** (< threshold and > 0)

## Per-model panel solve-rates

| model | tier | tasks | solves/k | solve-rate |
| --- | --- | --- | --- | --- |
| openai/gpt-4o-mini | weak | 5 | 0/30 | 0.0000 |
| anthropic/claude-sonnet-4-6 | mid | 5 | 0/30 | 0.0000 |
| anthropic/claude-opus-4-8 | frontier | 5 | 4/30 | 0.1333 |
| openai/gpt-5.5 | frontier | 5 | 9/30 | 0.3000 |

- per-tier (pooled): weak=0.0000, mid=0.0000, frontier=0.2167 (weak <= mid <= frontier: True; gold 100% >= frontier: True)

## IRT difficulty / discrimination
- difficulty: mean=1.3076, min=1.1254, max=1.6018
- discrimination: mean=4.8973, min=3.9917, max=5.5436

## Breakdown
- by generator: bug_combination=3, multi_file=2
- by language: python=5
- breakdown sums to shipped total: True

## Counts reconciliation
- tasks/*/ = 5, jsonl = 5, parquet = 5 (reconciled: True)

## Provenance audit
- completeness: 5/5 complete (PASS)
- consistency: 5/5 consistent (PASS)

## Measured harvest yield & pipeline funnel (honest)

**Scope (user mid-mission decision):** the pilot is an **engine demonstrator**.
The genuinely-hard-AND-100%-verifiable "keep" sweet spot is empirically rare and
expensive per keep, so the "done" bar is **>=1 genuine band-KEEP task (target
1-5)**, not 10-30. The full multi-language (Python/JS-TS/Go) + six-generator
engine capability is proven by the sealed M2-M5 milestones, independent of this
shipped set. **No gate and no keep band was ever loosened to inflate counts.**

- **Shipped (kept) set: 5 tasks** — all `mahmoud/boltons`, Python, across
  **2 generators** (`bug_combination`=3, `multi_file`=2). Single-language by
  outcome, not by design (see the funnel below).
- **Whole-run funnel (monotone `sourced >= env >= synth >= oracle-pass >= keep == exported`):**
  the self-budgeted background harvest processed **560 candidates** across 24
  advanced batches. Per batch each stage is non-increasing
  (`sourced=24 >= env=24 >= synth=24 >= oracle_pass >= keep`); the exported count
  equals the kept count equals the `tasks/<id>/` count equals the jsonl/parquet
  row count = **5** (deduped by deterministic `task_id`).
- **Disposition ledger (`dispositions.jsonl`, 360 rows from the billed batches
  that post-date the ledger):** `oracle_reject=279`, `calib_drop=77`, `kept=4`
  (the 5th keep, seed-0, predates the ledger). Of the 81 oracle-passers,
  **77 were band-DROPPED by the UNCHANGED keep band** — 75 as too-easy
  (frontier pass@k > 0.5) and 2 as too-hard (frontier pass@k == 0). These drops
  are the calibration band working correctly, not a shortfall.
- **Per-language funnel reality (honest):**
  - **Python:** 79 oracle-passers -> 5 kept. The band-populating path
    (`boltons` large-symbol difficulty rungs) is the only one that landed in-band.
  - **Go** (`spf13/cast` + `gorilla/mux`, pure-logic structural faults):
    reached **oracle-pass twice** but both were **band-DROPPED at calibration**
    (too easy for the frontier panel); no Go candidate landed in the keep band
    within budget. Reported honestly rather than forcing a keep.
  - **JS/TS** (`qs` pr_mirror `#555/#559`): all attempts **rejected at the oracle
    gates** (never reached calibration) — the oracle working as designed.
- **LLM usage / cost accounting (teacher + panel, threaded per-call):** the
  billed batches consumed **~349.7M tokens** for **~$1061.92** of measured
  API cost; the driver's cumulative budget accounting (including prior-run
  continuity) reached **$1301.90** against the authorized **$1400** ceiling,
  where the run self-stopped (`status=budget_exhausted`). Non-zero token + cost
  figures are attributable to teacher (synthesis/oracle proposals) and panel
  (weak/mid/frontier rollout) calls.
- **Measured keep-rate:** ~5 keeps / 560 candidates ~= **0.9%**, ~= **$260 per
  shipped keep** — consistent with the "rare and expensive" engine-demonstrator
  finding that drove the scope decision.

_Gold (Headline A) was independently re-measured on this finalized snapshot:
`evaluate.sh` in Docker, 2 fresh `--rm` runs per task, 5/5 `{"score":1}`,
deterministic, Phase-1 broken-fail + regression-green for all 5._
