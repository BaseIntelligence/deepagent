# prod_hard_deepswe_med - M27 DeepSWE-median product hardness

**Status:** certified **N=5** (min 5, target 10). **ok_for_product_wave=True**.
**Role:** Current PRODUCT hardness root under M27 floors (hybrid multi-file + isolation + green-flake).
**Historical softer band (M25/M26):** datasets/prod_hard_keep (retained for audit; not deleted).

## Floors (product defaults)

| Floor | Default |
|------:|--------:|
| source files | >= 4 OR hybrid (files>=3 AND added>=500 AND hunks>=14) |
| source hunks | >= 14 |
| gold added lines | >= 400 |
| F2P nodes | >= 5 |

Dual-truth (HarborDocker sol=1/null=0) + prompt alignment + intrinsic non-easy still required.

## Certified keeps (N=5)

| task_id | files | hunks | added | f2p | floors_ok |
|---------|------:|------:|------:|----:|:---------:|
| realpr-itemadapter-101 | 4 | 15 | 726 | 43 | True |
| realpr-packaging-1120 | 3 | 24 | 882 | 9 | True |
| realpr-werkzeug-2608 | 16 | 83 | 479 | 5 | True |
| realpr-werkzeug-2637 | 5 | 24 | 468 | 13 | True |
| realpr-werkzeug-3116 | 16 | 74 | 582 | 28 | True |

## p50 vs DeepSWE sample

| metric | product p50 | DeepSWE sample p50 |
|--------|------------:|-------------------:|
| source files | 5.0 | ~6 |
| source hunks | 24.0 | ~14 |
| gold added | 582.0 | ~640 |
| f2p nodes | 13.0 | (n/a sample) |

Gates: p50(added)>=400 -> **True**; p50(files)>=3 hybrid-ok -> **True**; N>=5 -> **True**.

## Campaign (m27c after m27f)

1. Final M27 floors (hybrid multi-file, isolation, green-flake).
2. Re-admit packaging-1120 under hybrid (3 files, added=882, hunks=24) with dual-truth.
3. Priority dual-truth densify on struct-ok materials with isolation + green-flake.
4. Agent Dockerfile installs pytest-xprocess for werkzeug ProcessStarter under Docker oracle.
5. Certified adds: packaging-1120, werkzeug-2637, werkzeug-2608 (ambient 2-node windows fileurl P2P trim; F2P floor held).
6. Expanded mine attempted; GH secondary rate-limit blocked late materialize. No fixture pad. prod_hard_keep left historical.

## Artifacts

- median_stats.json
- ship_summary.json, pack_manifest.json, gate_audit.jsonl, oracle_evidence.json
- densify logs: generate_m27c_densify*.log
- dual-truth evidence under evidence/oracle_* and densify work dirs
