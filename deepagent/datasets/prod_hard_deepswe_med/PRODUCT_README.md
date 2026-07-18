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

Dual-truth (HarborDocker sol=1/null=0) + prompt alignment + intrinsic non-easy still required.

## Certified keeps (N=2)

| task_id | files | hunks | added | f2p | floors_ok |
|---------|------:|------:|------:|----:|:---------:|
| realpr-itemadapter-101 | 4 | 15 | 726 | 43 | True |
| realpr-werkzeug-3116 | 16 | 74 | 582 | 28 | True |

## p50 vs DeepSWE sample

| metric | product p50 | DeepSWE sample p50 |
|--------|------------:|-------------------:|
| source files | 10.0 | ~6 |
| source hunks | 44.5 | ~14 |
| gold added | 654.0 | ~640 |
| f2p nodes | 35.5 | (n/a sample) |

Gates: p50(added)>=400 → **True**; p50(files)>=4 → **True**.
Wave OK requires also N>=5 dual-truth keeps (currently **False**).

## Campaign (m27c retry)

1. free -g; Docker dual-truth concurrency 1; reclaimed stale unit harbor jobs only.
2. Priority materials union from live_materials_m22/m27*/expand/recent + historical keep candidates.
3. `packaging-1120` considered: **not re-admitted** (files=3 < multi-file floor 4).
4. Serial envbuild+HarborDocker cert on priority then expanded recent packs.
5. Expanded live mine (list_pulls + classify_pr_files) found additional struct-ok materials (werkzeug/jinja/flask/pydantic/tornado/rq/marshmallow/...), but dual-run green/F2P rejected all except the prior two keeps.
6. Host dual-run install hardened (no `pip -U` poison; pytest-xprocess/simplejson/…) — werkzeug siblings still green-unclean on `RequestRedirect`; only itemadapter-101 + werkzeug-3116 dual-truth.
7. Fail-closed after priority + expanded funnel honestly exhausts with N=2.
8. DeepSWE-style prompt rewrite on keeps; `median_stats.json` updated; no fixture pad.

## Artifacts

- `median_stats.json`
- `ship_summary.json`, `pack_manifest.json`, `gate_audit.jsonl`, `oracle_evidence.json`
- `generate_m27c_retry.log`, `generate_m27c_cert2.log`
- `refresh_instructions_m27c.log`
