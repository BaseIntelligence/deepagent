# prod_hard_deepswe_med — M27 DeepSWE-median product hardness

**Status:** FAIL-CLOSED under-yield (certified **N=2** < min **5**).
**Role:** Intended current PRODUCT hardness root under M27 floors.
**Historical softer band (M25/M26):** `datasets/prod_hard_keep` (retained for audit; not deleted).

## Floors (product defaults)

| Floor | Default |
|------:|--------:|
| source files | >= 4 |
| source hunks | >= 14 |
| gold added lines | >= 400 |
| F2P nodes | >= 5 |
| dual-truth | HarborDocker sol=1 / null=0 |
| alignment + intrinsic | non-easy (M25) |

## Certified keeps (N=2)

| task_id | files | hunks | added | f2p | floors_ok |
|---------|------:|------:|------:|----:|-----------|
| realpr-itemadapter-101 | 4 | 15 | 726 | 43 | True |
| realpr-werkzeug-3116 | 16 | 74 | 582 | 28 | True |

## p50 vs DeepSWE sample

| metric | product p50 (N=2) | DeepSWE sample p50 (~48) | product goal |
|--------|--------------------:|-------------------------:|--------------|
| files | 10.0 | 6 | >=4 |
| hunks | 44.5 | 14 | >=14 floor per pack |
| added | 654.0 | 640 | p50>=400 |
| f2p | 35.5 | n/a | >=5 floor per pack |

Structural p50 on dual-truth survivors can sit in-band, but **wave success requires N>=5**. This tree is not a completed product wave.

## Honesty / funnel

- `source_track=real_pr`, live materials only, **no fixture pad**.
- Materials: `live_materials_m22` + sibling live materials + fresh materialize (werkzeug/httpcore/oauthlib/structlog/rq/scrapy/quart/...).
- Four generate passes under M27 floors; dual-truth survivors: realpr-itemadapter-101, realpr-werkzeug-3116.
- Dominant rejects: empty F2P cohort, green suite unclean/zero passers, hardness floors on near-band packs.
- Docker cert concurrency 1; ship `ok=false` when certified < min_packs.

## Artifacts

- `median_stats.json`
- `ship_summary.json`, `pack_manifest.json`, `gate_audit.jsonl`, `oracle_evidence.json`, `report.md`
- `generate_pass*.log`

