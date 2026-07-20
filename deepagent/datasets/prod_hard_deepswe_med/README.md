---
pretty_name: DeepAgent
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

# DeepAgent — current product hardness (DeepSWE-median / M28 diversity)

**Hugging Face stable pin:** revision **`main`** on [`BaseIntelligence/deepagent`](https://huggingface.co/datasets/BaseIntelligence/deepagent).  
**Automation / dev mirror:** revision **`test`** (may stay in sync at the same N=9 product).  
**Local product root:** `datasets/prod_hard_deepswe_med`  
**Band:** `deepswe_median_m27` · certified **N=9** (M28 densify hard gate N≥8 PASS; target 15 shortfall honest)

**`main` = current stable product pin.** Full-folder M30 promotion ships the same certified hardness tree that automation previously published only on `test`. Pull `main` for the stable public pin; `test` remains available for CI/dev mirrors and is expected to stay N=9-aligned unless a later wave moves it first.

This revision is the **current product hardness corpus**. Structural difficulty stays on the **DeepSWE median** floors (M27). **M28** adds coverage volume under a **repo diversity cap** so one upstream no longer dominates the set.

> **Not M16 N=10.** Historical `datasets/test_n10` / early M16 trees are **not** the live product. Current certified hardness is **N=9** under M27 floors + M28 diversity.

## M28 coverage + diversity policy

| Policy | Value |
|--------|------:|
| Certified N | **9** |
| Unique upstream repos | **7** |
| Max packs per repo | **≤ 2** |
| Diversity gate | PASS (N≥8, unique_repos≥5, max_packs_per_repo≤2) |
| Floors | unchanged M27 (see below) |
| Fixture pad | **false** |

`packs_per_repo`: scrapy/itemadapter 1 · pypa/packaging 2 · Textualize/rich 1 · pallets-eco/wtforms 1 · pallets/werkzeug 2 · oauthlib/oauthlib 1 · pallets/click 1.

Preferred targets N≥12/15 remain aspirational; shortfall is honest (many floor-ok materials still fail Docker dual-truth). See `coverage_stats.json` and `PRODUCT_README.md`.

## Supersession

| Item | Status |
|------|--------|
| **Current stable product (HF `main`)** | `datasets/prod_hard_deepswe_med` · M27 floors + **M28 diversity (max 2/repo)** · **N=9** |
| **HF `test` mirror** | Same N=9 product after M29/M30; automation default write target (may stay in sync) |
| **M16 N=10 / `test_n10` claims** | **Superseded** — early M16 live wave and any card/docs that called N=10 the current product are historical only |
| Prior M27 HF `test` (N=5, denser werkzeug) | Superseded by this diversified tree (drops weak 3rd-werkzeug keep; adds non-werkzeug dual-truth keeps) |
| Prior M28c/M29d HF `test` (same N=9 ids) | Promoted to **`main`** by M30a full-folder upload (card `pretty_name: DeepAgent`) |
| Historical softer band | `datasets/prod_hard_keep` (M25/M26) retained **locally for audit only** — not HF `main` |

Do **not** treat model dual-solve alone as a product drop (M25 intrinsic policy). Hardness refusals remain dual-truth fail, prompt–verifier misalignment, structural floors, and high-confidence intrinsic `EASY_REQUEST`.

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
| diversity (M28) | **≤ 2 certified packs per upstream repo** |

DeepSWE public sample reference (≈48 packs): files p50≈6, hunks p50≈14, added p50≈640.

## Certified keeps (N=9)

| task_id | repo | files | hunks | added | f2p |
|---------|------|------:|------:|------:|----:|
| `realpr-itemadapter-101` | scrapy/itemadapter | 4 | 15 | 726 | 43 |
| `realpr-packaging-1120` | pypa/packaging | 3 | 24 | 882 | 9 |
| `realpr-packaging-1267` | pypa/packaging | 3 | 22 | 1200 | 144 |
| `realpr-rich-3930` | Textualize/rich | 26 | 29 | 12223 | 78 |
| `realpr-wtforms-923` | pallets-eco/wtforms | 9 | 29 | 483 | 24 |
| `realpr-werkzeug-2637` | pallets/werkzeug | 5 | 24 | 468 | 13 |
| `realpr-werkzeug-3116` | pallets/werkzeug | 16 | 74 | 582 | 28 |
| `realpr-oauthlib-889` | oauthlib/oauthlib | 9 | 17 | 620 | 6 |
| `realpr-click-3442` | pallets/click | 18 | 138 | 960 | 20 |

Product p50: files=9.0 · hunks=24.0 · added=726.0 · f2p=24.0.

## Layout

```text
pack_manifest.json
README.md                 # this dataset card
PRODUCT_README.md
PROVENANCE.md
report.md
median_stats.json
coverage_stats.json
ship_summary.json
gate_audit_summary.json
tasks/<task_id>/
  task.toml
  instruction.md
  environment/Dockerfile
  tests/…
  solution/…
```

## Pull / eval (product)

Prefer the stable pin:

```bash
deepagent pull --repo-id BaseIntelligence/deepagent --revision main --out datasets/hf_pull_main --json
deepagent eval --product-root datasets/hf_pull_main \
  --max-packs 15 --k 1 --n-concurrent 5 --hard-stop-usd 600 \
  --model x-ai/grok-4.5 --model moonshotai/kimi-k2.7-code \
  --out datasets/panel_prod_hard_deepswe_med_m28 --json
```

Automation / dev mirror (same N=9 after M29/M30):

```bash
deepagent pull --repo-id BaseIntelligence/deepagent --revision test --out datasets/hf_pull_test --json
```

Default median eval pair: `x-ai/grok-4.5` + `moonshotai/kimi-k2.7-code`.  
Dual-solve pack rate quality gate on the median product: **≤ 30%**.

## Secrets

No API tokens, Bearer headers, or `.env` contents are shipped in this tree. Auth for upload/pull is local-only (`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`).
