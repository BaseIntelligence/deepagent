# m29d — HF full re-upload N=9 + refreshed card

**Feature:** `m29d-hf-full-reupload-n9-card`  
**Fulfills:** VAL-PUB-005  
**UTC:** 2026-07-20

## What shipped

| Field | Value |
|---|---|
| Local src | `datasets/prod_hard_deepswe_med` (staged via `datasets/_m29d_hf_stage`) |
| Stage note | Full folder pack tree + corpus meta; excluded `generate*.log`, `_m28*` work dirs, `.env`; restored committed `tests/Dockerfile` image tags |
| Repo | `BaseIntelligence/deepagent` |
| Revision | **`test` only** (not `main`; not card-only) |
| Pack count | **9** |
| Band | `deepswe_median_m27` + M28 diversity (max 2/repo) |
| unique_repos | **7** |
| max packs/repo | **2** |
| Upload SHA | `c17bc27918d2e4ac1196ac3af049f1c7b58a161d` |
| Commit URL | https://huggingface.co/datasets/BaseIntelligence/deepagent/commit/c17bc27918d2e4ac1196ac3af049f1c7b58a161d |
| Message | `upload_folder complete` |

## Task ids (local == remote list == pull)

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
- Remote/pull `pack_manifest`: count=9, task_ids exact match
- Pull out: `/tmp/da_m29d_hf_pull` pack_count=9
- Sample pack layout (itemadapter-101): task.toml, instruction.md, environment/, tests/, solution/ present
- Meta present on pull: coverage_stats.json, median_stats.json, PRODUCT_README.md, PROVENANCE.md, README.md

## Card / supersession

- Dataset card `README.md` YAML front-matter: `pretty_name: DeepAgent`
- Documents **N=9**, M28 diversity **max 2 packs/repo**, unique_repos=7
- M27 floors table (files≥4 OR hybrid; hunks≥14; added≥400; F2P≥5; dual-truth)
- Explicit supersession of **M16 N=10 / `test_n10`** claims (historical only; not current product)
- Re-asserts M28c N=9 diversified product after public-surface card refresh

## Secrets

- Pre-upload local text scan on stage: **0 hits**
- Pull tree scan: **0 hits**
- Upload/pull CLI JSON: no token fields (auth via gitignored `.env` only)
- Stage excludes: generate logs, `_m28*` work dirs, `.env`

## Commands

```bash
cd /projects/Agent-SWE/deepagent
set -a && source .env && set +a

.venv/bin/deepagent upload --src datasets/_m29d_hf_stage \
  --repo-id BaseIntelligence/deepagent --revision test --dry-run --json

.venv/bin/deepagent upload --src datasets/_m29d_hf_stage \
  --repo-id BaseIntelligence/deepagent --revision test --json

.venv/bin/deepagent pull --repo-id BaseIntelligence/deepagent \
  --revision test --out /tmp/da_m29d_hf_pull --json
```

Services.yaml: `upload_prod_hard_deepswe_med_m29`, `pull_test_m29`.
