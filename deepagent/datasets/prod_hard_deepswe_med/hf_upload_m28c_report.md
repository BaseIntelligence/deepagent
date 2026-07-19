# m28c — HF upload diversified DeepSWE-median product

**Feature:** `m28c-hf-upload-diversified-median`  
**Fulfills:** VAL-DCOV-006  
**UTC:** 2026-07-19

## What shipped

| Field | Value |
|---|---|
| Local src | `datasets/prod_hard_deepswe_med` (staged via `datasets/_m28c_hf_stage`, logs excluded) |
| Repo | `BaseIntelligence/deepagent` |
| Revision | **`test` only** (not `main`) |
| Pack count | **9** |
| Band | `deepswe_median_m27` + M28 diversity |
| unique_repos | **7** |
| max packs/repo | **2** |
| Upload SHA | `3904cbbd0418fcc53233149dd49f2636de0c9465` |
| Commit URL | https://huggingface.co/datasets/BaseIntelligence/deepagent/commit/3904cbbd0418fcc53233149dd49f2636de0c9465 |
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

- Local ↔ pull `tasks/*` set match: **yes**
- Remote/pull `pack_manifest`: count=9, band=`deepswe_median_m27`, task_ids exact match
- Pull `/tmp/da_m28c_hf_pull`: pack_count=9, task_ids exact match
- Prior third-werkzeug keep `realpr-werkzeug-2608` **gone** from `test`
- Sample pack layout (itemadapter-101): task.toml, instruction.md, environment/, tests/, solution/ present

## Card / diversity

- Dataset card `README.md`: YAML front-matter + **M28 diversity (max 2/repo) + N=9** + M27 floors table
- `coverage_stats.json` / `PRODUCT_README.md` uploaded; N=9, unique_repos=7, max_packs_per_repo=2
- Supersedes M27d HF `test` N=5 denser-werkzeug tree

## Secrets

- Pre-upload local text scan: **0 hits**
- Pull tree scan: **0 hits**
- Upload/pull CLI JSON: no token fields (auth via gitignored `.env` only)

## Commands

```bash
cd /projects/Agent-SWE/deepagent
set -a && source .env && set +a

.venv/bin/deepagent upload --src datasets/_m28c_hf_stage \
  --repo-id BaseIntelligence/deepagent --revision test --dry-run --json

.venv/bin/deepagent upload --src datasets/_m28c_hf_stage \
  --repo-id BaseIntelligence/deepagent --revision test --json

.venv/bin/deepagent pull --repo-id BaseIntelligence/deepagent \
  --revision test --out /tmp/da_m28c_hf_pull --json
```

Services.yaml: `upload_prod_hard_deepswe_med`, `pull_test_revision`.
