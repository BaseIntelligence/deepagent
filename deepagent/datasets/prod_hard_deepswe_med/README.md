---
pretty_name: DeepAgent Hardness — DeepSWE-median (M27)
tags:
  - code
  - software-engineering
  - harbor
  - deepagent
  - real-pr
license: mit
task_categories:
  - text-generation
  - other
size_categories:
  - n<1K
---

# DeepAgent — current product hardness (DeepSWE-median / M27)

**Hugging Face revision:** `test` on [`BaseIntelligence/deepagent`](https://huggingface.co/datasets/BaseIntelligence/deepagent).  
**Local product root:** `datasets/prod_hard_deepswe_med`  
**Band:** `deepswe_median_m27` · certified **N=5** (min 5, target 10)

This revision is the **current product hardness corpus**. It calibrates structural difficulty to the **DeepSWE median band** so thin API-only packs no longer flatten dual-model ranking.

## Supersession — M25 / M26 `prod_hard_keep` soft band

| Item | Status |
|------|--------|
| **Current product (this revision)** | `datasets/prod_hard_deepswe_med` · DeepSWE-median floors · band `deepswe_median_m27` |
| **Historical softer band (superseded on HF `test`)** | `datasets/prod_hard_keep` · M25/M26 · N=10 · retained **locally for audit only** |

The previous HF `test` tree (M25 N=10: attrs/httpx/qs/werkzeug-thin class among others) is **superseded**. Those packs remain on disk under `datasets/prod_hard_keep` for audit and offline floor-bite reports; they are **not** the live `test` product after this upload.

Do **not** treat model dual-solve alone as a product drop (M25 intrinsic policy still holds). Hardness refusals remain dual-truth fail, prompt–verifier misalignment, structural floors, and high-confidence intrinsic `EASY_REQUEST`.

## DeepSWE-median structural floors (product defaults)

| Floor | Default |
|------:|--------:|
| source files | ≥ **4** **OR** hybrid: files ≥ 3 **AND** gold added ≥ 500 **AND** hunks ≥ 14 |
| source hunks | ≥ **14** |
| gold added lines | ≥ **400** |
| F2P nodes | ≥ **5** |
| dual-truth | HarborDocker sol=1 / null=0 |
| alignment | prompt–verifier fail-closed |
| intrinsic | non-easy (request+gold; model scores are leaderboard-only) |

DeepSWE public sample reference (≈48 packs): files p50≈6, hunks p50≈14, added p50≈640.

## Certified keeps (N=5)

| task_id | files | hunks | added | f2p |
|---------|------:|------:|------:|----:|
| `realpr-itemadapter-101` | 4 | 15 | 726 | 43 |
| `realpr-packaging-1120` | 3 | 24 | 882 | 9 |
| `realpr-werkzeug-2608` | 16 | 83 | 479 | 5 |
| `realpr-werkzeug-2637` | 5 | 24 | 468 | 13 |
| `realpr-werkzeug-3116` | 16 | 74 | 582 | 28 |

Product p50: files=5.0 · hunks=24.0 · added=582.0 · f2p=13.0.

## Layout

```text
pack_manifest.json
README.md                 # this dataset card
PRODUCT_README.md
PROVENANCE.md
report.md
median_stats.json
ship_summary.json
tasks/<task_id>/
  task.toml
  instruction.md
  environment/Dockerfile
  tests/…
  solution/…
```

## Pull / eval (product)

```bash
deepagent pull --repo-id BaseIntelligence/deepagent --revision test --out datasets/hf_pull_test --json
deepagent eval --product-root datasets/hf_pull_test \
  --max-packs 15 --k 1 --n-concurrent 5 --hard-stop-usd 600 \
  --out datasets/panel_prod_hard_deepswe_med_n5 --json
```

Default median eval pair (M27): `x-ai/grok-4.5` + `moonshotai/kimi-k2.7-code`.  
Dual-solve pack rate quality gate on the median product: **≤ 30%**.

## Secrets

No API tokens, Bearer headers, or `.env` contents are shipped in this tree. Auth for upload/pull is local-only (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`).
