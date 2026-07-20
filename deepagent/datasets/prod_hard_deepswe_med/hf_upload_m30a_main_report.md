# m30a — HF full-folder upload to revision **main** (N=9)

**Feature:** `m30a-hf-upload-main-n9`  
**Fulfills:** VAL-PUB-007  
**UTC:** 2026-07-20

## What shipped

| Field | Value |
|---|---|
| Local src | `datasets/prod_hard_deepswe_med` (staged via `datasets/_m30a_hf_stage`) |
| Stage note | Clean tree from m29d pattern: tasks/* (committed Docker image tags), pack_manifest, coverage/median stats, PRODUCT/PROVENANCE, refreshed dataset card; excluded generate logs, `_m*work*`, `.env` |
| Repo | `BaseIntelligence/deepagent` |
| Revision | **`main`** (user-requested stable product pin; full folder, not card-only) |
| Pack count | **9** |
| Band | `deepswe_median_m27` + M28 diversity (max 2/repo) |
| unique_repos | **7** |
| max packs/repo | **2** |
| Upload commit | `150bf67e18450bda7fa8760e73234ae2d2e3d7ff` |
| Commit URL | https://huggingface.co/datasets/BaseIntelligence/deepagent/commit/150bf67e18450bda7fa8760e73234ae2d2e3d7ff |
| Message | `upload_folder complete` |

## Task ids (local == main pull == exact)

1. `realpr-click-3442`
2. `realpr-itemadapter-101`
3. `realpr-oauthlib-889`
4. `realpr-packaging-1120`
5. `realpr-packaging-1267`
6. `realpr-rich-3930`
7. `realpr-werkzeug-2637`
8. `realpr-werkzeug-3116`
9. `realpr-wtforms-923`

- Local ↔ pull `tasks/*` set match: **yes** (exact)
- Remote/pull `pack_manifest`: count=9, certified_n=9, task_ids exact match
- Pull out: `/tmp/da_m30a_hf_pull_main` pack_count=9
- Sample pack layout (itemadapter-101): task.toml, instruction.md, environment/, tests/, solution/ present
- Meta present on pull: coverage_stats.json, median_stats.json, PRODUCT_README.md, PROVENANCE.md, README.md

## Card / pin semantics

- Dataset card `README.md` YAML front-matter: `pretty_name: DeepAgent`
- Documents **N=9**, M28 diversity **max 2 packs/repo**, unique_repos=7, M27 floors table
- Explicit: **`main` = current stable product pin**
- Explicit: revision **`test`** may stay in sync as automation/dev mirror (also N=9)
- Supersession of **M16 N=10 / `test_n10`** claims (historical only; not current product)

## Secrets

- Pre-upload local text scan on stage: **0 hits** (`secrets_scan_m30a.json`)
- Scanner targets credential shapes (hf_/gho_/env assignments/PEM); oauthlib domain prose about Bearer token types is not a hit
- Upload/pull CLI JSON: no token fields (auth via gitignored `.env` only)
- Stage excludes: generate logs, work dirs, `.env`

## Spot-check: revision `test` not wiped

| Field | Value |
|---|---|
| Pull out | `/tmp/da_m30a_hf_pull_test_spot` |
| pack_count | **9** |
| task_ids | exact same 9 ids as main/local |
| Result | **test still N=9** (not wiped by main upload) |

## Commands

```bash
cd /projects/Agent-SWE/deepagent
set -a && source .env && set +a

.venv/bin/deepagent upload --src datasets/_m30a_hf_stage \
  --repo-id BaseIntelligence/deepagent --revision main --dry-run --json

.venv/bin/deepagent upload --src datasets/_m30a_hf_stage \
  --repo-id BaseIntelligence/deepagent --revision main --json

.venv/bin/deepagent pull --repo-id BaseIntelligence/deepagent \
  --revision main --out /tmp/da_m30a_hf_pull_main --json

.venv/bin/deepagent pull --repo-id BaseIntelligence/deepagent \
  --revision test --out /tmp/da_m30a_hf_pull_test_spot --json
```

## VAL-PUB-007 checklist

| Check | Result |
|---|---|
| HF `main` has exactly 9 product task ids matching local | **PASS** |
| `pack_manifest` count=9 on main | **PASS** |
| Card on main reflects N=9 + M28 max2/repo + M27 floors (not M16 N=10) | **PASS** |
| Secrets scan clean (0) | **PASS** |
| Full folder upload (not card-only) | **PASS** |
| `test` still N=9 | **PASS** |
