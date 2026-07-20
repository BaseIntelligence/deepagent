# m31a — HF Dataset Viewer table + GitHub-style card + banner

**Feature:** `m31a-hf-viewer-jsonl-and-github-style-card`  
**Fulfills:** VAL-PUB-008, VAL-PUB-009  
**UTC:** 2026-07-20

## Problem

Hub Dataset Viewer failed with heuristics-could-not-detect-any-supported-data-files (Harbor pack tree only; no JSONL/Parquet at a discoverable path / configs). User also requested the GitHub product-first README (banner + N=9 story) adapted for the HF card.

## What shipped

| Field | Value |
|---|---|
| Local src / stage | `datasets/prod_hard_deepswe_med` staged via `datasets/_m31a_hf_stage` (m30 clean-pack pattern + new viewer assets) |
| Repo | `BaseIntelligence/deepagent` |
| Revisions | **`main`** (stable pin) **and** **`test`** (automation mirror) |
| Pack count | **9** (unchanged product ids) |
| Viewer table | `data/packs.jsonl` + `data/packs.parquet` (exactly **9** rows) |
| Banner | `banner.png` + `assets/banner.png` (from `/projects/Agent-SWE/assets/banner.png`) |
| Card | `README.md` YAML `configs` / `data_files` → `data/packs.jsonl`; GitHub-style body + scoreboard |

### Viewer columns (no gold leaks)

`task_id`, `repository_url`, `base_commit`, `language`, `license`, `source_files`, `source_hunks`, `gold_added_lines`, `f2p_nodes`, `instruction` (from each pack’s `instruction.md`), `pack_path`, `source_track`

**Not included:** `solution.patch` / `test.patch` bodies, secrets, `diff --git` gold text.

### Task ids (local == main pull == test pull)

1. `realpr-click-3442`
2. `realpr-itemadapter-101`
3. `realpr-oauthlib-889`
4. `realpr-packaging-1120`
5. `realpr-packaging-1267`
6. `realpr-rich-3930`
7. `realpr-werkzeug-2637`
8. `realpr-werkzeug-3116`
9. `realpr-wtforms-923`

## HF commits (key uploads)

Upload used `HfApi.upload_file` per path (full `upload_folder` / multi-op `create_commit` hung on this host under Xet/LFS processing; single-file path is reliable). Product Harbor trees already present from M30; M31 added/refreshed viewer + card + banner on both revisions.

### main (representative)

| Path | Commit |
|---|---|
| `data/packs.jsonl` | `97133a85cca21861d07adca08737b2a7e298b796` |
| `data/packs.parquet` | `a46eee9b949ad858112fc737723ff67f5786e3ba` |
| `banner.png` | `a3d5db1c8db6867a0af97dea9f3d9d62ae5e1b56` |
| `assets/banner.png` | `a5324c9a3cb6150f0f00b4d926f7ff77b865ce32` |
| `README.md` (card) | `5fe4e7834ba88a8ae34da8eea7a5f22e05503575` |

### test (representative)

| Path | Commit |
|---|---|
| `data/packs.jsonl` | `0c127f92fd11ee16e96ca67747b7290a5a1de510` |
| `data/packs.parquet` | `8c9f46b71fcd73ce58a1bd386da7fe5371024b7e` |
| `banner.png` | `2f2ae9966a4d9eec3a2ae4b6d87c19b4442758ed` |
| `assets/banner.png` | `f96cf8aadee5f21ad6f4bb37eb84dd2f88391d22` |
| `README.md` (card) | `64cce6112b30dced02f551ca548ff0b34a62cf77` |

## Verification

| Check | main | test |
|---|---|---|
| `list_repo_files` has `data/packs.jsonl` | **yes** | **yes** |
| `list_repo_files` has `data/packs.parquet` | **yes** | **yes** |
| `list_repo_files` has `banner.png` | **yes** | **yes** |
| `assets/banner.png` | **yes** | **yes** |
| Card YAML `configs` + `data_files` → `data/packs.jsonl` | **yes** | **yes** |
| Card `![DeepAgent Banner](banner.png)` | **yes** | **yes** |
| Card product-first N=9 + floors + scoreboard 0.33/0.11/0.11 | **yes** | **yes** |
| Pull spot-check rows == 9, ids exact match local | **yes** | **yes** |
| No `solution.patch` / `diff --git` in jsonl | **yes** | **yes** |
| `task.toml` count still 9 (full product tree retained) | **9** | **9** |
| `n_repo_files` | **196** | **196** |
| Parquet load via pandas | **9 rows** | **9 rows** |
| Secrets scan stage | **0 hits** (`secrets_scan_m31a.json`) | same stage |

Pull out dirs: `/tmp/da_m31a_hf_pull_main`, `/tmp/da_m31a_hf_pull_test` (targeted `hf_hub_download` of viewer/card/banner/manifest).

Optional `datasets.load_dataset(...)` smoke: HuggingFace `datasets` package **not installed** in the product venv (and would be shadowed by the local `datasets/` directory name). Viewer paths + pandas parquet load cover the table contract.

**Note:** SOCKS proxy env (`ALL_PROXY` / Oxylabs) can block HF content CDN downloads; verification ran with proxy env cleared. Uploads themselves succeed with token auth.

## Local product artifacts (committed)

Under `datasets/prod_hard_deepswe_med/`:

- `data/packs.jsonl`, `data/packs.parquet`
- updated `README.md` (HF card with configs + GitHub-style body)
- `banner.png`, `assets/banner.png`
- `hf_upload_m31a_viewer_card_report.md` (this file)
- `secrets_scan_m31a.json`

Stage tree (filesystem, typically not required as a git product path): `datasets/_m31a_hf_stage/`.

## VAL checklist

| Assertion | Result |
|---|---|
| **VAL-PUB-008** viewer supported table on **main** and **test**, YAML configs, N=9 rows, no gold patch bodies | **PASS** |
| **VAL-PUB-009** GitHub-style card + banner on **main** and **test**, HF adaptations, N=9 product story, main pin | **PASS** |

## Secrets

- Pre-upload local text scan on stage: **0 hits**
- Never printed `HF_TOKEN` / Hub auth headers
- No gold patches embedded in viewer table
