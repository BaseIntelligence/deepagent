# m27d — HF upload DeepSWE-median product

**Feature:** `m27d-hf-upload-deepswe-median`  
**Fulfills:** VAL-DMED-006  
**UTC:** 2026-07-19

## What shipped

| Field | Value |
|---|---|
| Local src | `datasets/prod_hard_deepswe_med` |
| Repo | `BaseIntelligence/deepagent` |
| Revision | **`test` only** (not `main`) |
| Pack count | **5** |
| Band | `deepswe_median_m27` |
| Upload SHA | `60b77df5408cc3a399c463de58103fc6ac7dfcdf` |
| Commit URL | https://huggingface.co/datasets/BaseIntelligence/deepagent/commit/60b77df5408cc3a399c463de58103fc6ac7dfcdf |
| Message | `upload_folder complete` |

## Task ids (local == remote list == pull)

1. `realpr-itemadapter-101`
2. `realpr-packaging-1120`
3. `realpr-werkzeug-2608`
4. `realpr-werkzeug-2637`
5. `realpr-werkzeug-3116`

- Local ↔ remote `tasks/*` set match: **yes**
- Remote `pack_manifest.json`: `count=5`, `band=deepswe_median_m27`, task_ids exact match
- Pull `/tmp/da_m27d_hf_pull`: pack_count=5, task_ids exact match
- Sample pack layout (itemadapter-101): task.toml, instruction.md, environment/, tests/, solution/ present

## Card / supersession

- Dataset card: corpus `README.md` (YAML front-matter + DeepSWE-median floors table)
- Notes M25/M26 soft band `datasets/prod_hard_keep` **superseded on HF `test`**
- Soft-band packs retained locally for audit; pruned from remote by uploader `delete_patterns=["*"]` mirror
- Old soft task ids (attrs/httpx/qs/werkzeug-thin class) **gone** from remote `test`

## Secrets

- Pre-upload local text scan: **0 hits**
- Pull tree scan: **0 hits**
- Upload/pull CLI JSON: no token fields (auth via gitignored `.env` only)

## Commands

```bash
cd /projects/Agent-SWE/deepagent
set -a && source .env && set +a

.venv/bin/deepagent upload --src datasets/prod_hard_deepswe_med \
  --repo-id BaseIntelligence/deepagent --revision test --dry-run --json

.venv/bin/deepagent upload --src datasets/prod_hard_deepswe_med \
  --repo-id BaseIntelligence/deepagent --revision test --json

.venv/bin/deepagent pull --repo-id BaseIntelligence/deepagent \
  --revision test --out /tmp/da_m27d_hf_pull --json
```

Services.yaml: `upload_prod_hard_deepswe_med`, `pull_test_revision`.
