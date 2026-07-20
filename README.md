<div align="center">

# DeepAgent

**Hard, Docker-verifiable software-engineering benchmarks from real merged PRs**

[![License](https://img.shields.io/github/license/BaseIntelligence/deepagent)](https://github.com/BaseIntelligence/deepagent/blob/main/LICENSE)
[![HF](https://img.shields.io/badge/HF-BaseIntelligence%2Fdeepagent-yellow.svg)](https://huggingface.co/datasets/BaseIntelligence/deepagent/tree/test)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![CLI](https://img.shields.io/badge/CLI-deepagent-black.svg)](deepagent/README.md)

![DeepAgent Banner](assets/banner.png)

</div>

DeepAgent ships **real_pr Harbor hardness packs**: live-mined multi-file pull requests, clone@SHA agent images, held-out verifier tests, and Docker dual-truth (solution reward = 1, null reward = 0). Primary product work runs through the **`deepagent`** CLI under [`deepagent/`](deepagent/).

---

## Current product (authoritative)

| Field | Value |
|---|---|
| **Product root** | [`deepagent/datasets/prod_hard_deepswe_med`](deepagent/datasets/prod_hard_deepswe_med/) |
| **N** | **9** certified packs |
| **unique_repos** | **7** |
| **max packs / repo** | **2** (M28 diversity) |
| **HF dataset** | [`BaseIntelligence/deepagent`](https://huggingface.co/datasets/BaseIntelligence/deepagent) revision **`test`** |
| **Primary CLI** | `deepagent` (`generate` / `upload` / `pull` / `eval` / `oracle`) |
| **Scoreboard** | [`deepagent/datasets/panel_prod_hard_deepswe_med_m28`](deepagent/datasets/panel_prod_hard_deepswe_med_m28/) |

### M27 DeepSWE-median hardness floors

Every keep must pass:

- **Multi-file:** source files ≥ **4**, **or** hybrid files ≥ **3** + gold added ≥ **500** + hunks ≥ **14**
- source **hunks ≥ 14**
- gold **added lines ≥ 400**
- **F2P nodes ≥ 5**
- HarborDocker **dual-truth** (sol = 1, null = 0) + prompt–verifier alignment
- live-mined `source_track=real_pr` only (no fixture pad, no hybrid motors)

See [`deepagent/datasets/prod_hard_deepswe_med/PRODUCT_README.md`](deepagent/datasets/prod_hard_deepswe_med/PRODUCT_README.md) and `coverage_stats.json`.

### Pack IDs (N=9)

`realpr-click-3442` · `realpr-itemadapter-101` · `realpr-oauthlib-889` · `realpr-packaging-1120` · `realpr-packaging-1267` · `realpr-rich-3930` · `realpr-werkzeug-2637` · `realpr-werkzeug-3116` · `realpr-wtforms-923`

### Scoreboard (M28 diversified panel)

Durable dual-model matrix on the current product (observational ranking only; dual-solve rate is the hardness quality gate ≤ 0.30):

| Model | pass@1 (k=1) |
|---|---|
| `x-ai/grok-4.5` | **3/9 ≈ 0.33** |
| `moonshotai/kimi-k2.7-code` | **1/9 ≈ 0.11** |
| dual_solve rate | **≈ 0.11** (1/9) |

Evidence: [`deepagent/datasets/panel_prod_hard_deepswe_med_m28/SUMMARY.md`](deepagent/datasets/panel_prod_hard_deepswe_med_m28/SUMMARY.md) and `scoreboard.json`.

### Historical (not current product)

| Path | Note |
|---|---|
| `deepagent/datasets/test_n10` | **Historical** M16 wave N=10 — not the live hardness product |
| `deepagent/datasets/prod_hard_keep` | Softer M25/M26 band — audit only |
| `deepagent/datasets/deepagent_v1` | Older Real-PR product archive N=20 |
| `deepagent/fixtures/real_pr_ship` | Unit shortlist only — never product N |

```mermaid
flowchart LR
  Mine[Live mine real_pr] --> Dual[Dual-run F2P/P2P]
  Dual --> Oracle[HarborDocker sol=1 null=0]
  Oracle --> Product[prod_hard_deepswe_med N=9]
  Product --> HF[HF BaseIntelligence/deepagent @test]
  Product --> Eval[deepagent eval Pier + Harbor]
  Eval --> Board[panel_prod_hard_deepswe_med_m28]
```

---

## Quick start (Real-PR product)

```bash
git clone https://github.com/BaseIntelligence/deepagent.git
cd deepagent/deepagent          # package root for the Real-PR factory
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]"
cp .env.example .env            # fill placeholders; never commit .env
```

Primary loop (paths relative to `deepagent/`):

```bash
# Live-mine hard real_pr under M27 floors (diversity max 2/repo)
deepagent generate \
  --target 10 --min-packs 5 --max-packs 15 \
  --out datasets/prod_hard_deepswe_med \
  --live-mine --oracle docker --panel offline --pier scripted

# Push pack trees to HF revision test
deepagent upload \
  --src datasets/prod_hard_deepswe_med \
  --repo-id BaseIntelligence/deepagent \
  --revision test

# Pull from HF
deepagent pull \
  --repo-id BaseIntelligence/deepagent \
  --revision test \
  --out datasets/hf_pull_test

# Dual-model Pier + Harbor eval (n_concurrent 1..5; hard-stop $600)
deepagent eval \
  --product-root datasets/prod_hard_deepswe_med \
  --max-packs 9 --k 1 --n-concurrent 5 \
  --hard-stop-usd 600 \
  --model x-ai/grok-4.5 \
  --model moonshotai/kimi-k2.7-code \
  --out datasets/panel_prod_hard_deepswe_med_m28

# HarborDocker dual-truth on one pack
deepagent oracle --pack-dir datasets/prod_hard_deepswe_med/tasks/realpr-click-3442
```

Full factory docs: [`deepagent/README.md`](deepagent/README.md).

Compatibility entry `swe-factory` remains for longer historical stages (`ship-deepagent`, `real-pr-pool`, `ledger`, …). Prefer **`deepagent`** for product work.

---

## Environment and GitHub 429 mitigation

Set secrets in `deepagent/.env` (gitignored). Placeholders only — never commit real tokens.

| Variable | Purpose |
|---|---|
| `GITHUB_TOKEN` / `GH_TOKEN` | GitHub REST + Search auth. Prefer `export GITHUB_TOKEN="$(gh auth token)"` after `gh auth login` |
| `OXYLABS_PROXY_URL` | **SOCKS** residential proxy URL for GitHub API rate-limit relief (e.g. `socks5h://user:pass@host:port`) |
| `ALL_PROXY` / `HTTPS_PROXY` | Optional chain so git/HTTPS clients share the same proxy |
| `HF_TOKEN` | Hugging Face upload/pull for `BaseIntelligence/deepagent` |
| `OPENROUTER_API_KEY` | Live panel / Pier model eval spend |
| `FACTORY_BUDGET_USD` | Hard spend cap (default `600`) |

**Notes:**

- Live mine hits `api.github.com`. Authenticated REST (`GITHUB_TOKEN` / `gh auth`) plus optional **SOCKS** via `OXYLABS_PROXY_URL` (and/or `ALL_PROXY` / `HTTPS_PROXY`) are the primary anti-429 path.
- The **Oxylabs realtime Web Scraper API** (`realtime.oxylabs.io`, page-shaped) is **optional** and **not required** for GitHub REST/Search mining.
- Never log proxy userinfo, Bearer tokens, `gho_`, or `hf_*` values. Never commit `.env`.

---

## Synthetic feature-deletion tasks (secondary path)

Besides live Real-PR mining, DeepAgent can generate **synthetic feature-deletion** tasks from a local checkout: remove a testable behavior, keep tests as the reward signal, and store the inverse repair as the oracle. The synthetic path is a scaling tool; **hardness product truth remains real_pr Harbor packs**.

Install the root package (exposes `swe-forge`):

```bash
cd /path/to/deepagent          # monorepo root
pip install -e ".[dev]"

git clone https://github.com/owner/repo.git ./target-repo

swe-forge synthetic generate \
  --repo-path ./target-repo \
  --repo owner/repo \
  --source-file src/package/module.py \
  --symbol target_function \
  --fail-to-pass "pytest tests/test_target.py -v" \
  --pass-to-pass "pytest tests/ -v" \
  --install-command "pip install -e ." \
  --output-folder ./synthetic_tasks \
  --output-jsonl ./synthetic_tasks.jsonl \
  --overwrite
```

Workspace layout and evaluation details: [docs/architecture/synthetic-feature-deletion.md](docs/architecture/synthetic-feature-deletion.md), [docs/architecture/workspace-format.md](docs/architecture/workspace-format.md).

### Brief methodology note

The synthetic feature-deletion idea is loosely inspired by public Cursor writing on Composer-style training (remove a behavior, restore it under tests). DeepAgent is **not affiliated with Cursor**. Real-PR Harbor hardness is the product north star; Composer is not the primary story.

---

## Development

### Real-PR factory (`deepagent/`)

```bash
cd deepagent
ruff format --check src tests && ruff check src tests
mypy src
pytest tests -q -m "not integration" -n 0   # never -n auto on this host
```

### Root forge package (`swe-forge`)

```bash
# monorepo root
ruff format src/ tests/
ruff check src/ tests/
mypy src/
pytest tests/ -v
```

---

## Repository layout

```text
.
├── README.md                 # this file (public product surface)
├── assets/
├── docs/architecture/        # synthetic workspace + eval docs
├── src/swe_forge/            # synthetic / forge package (CLI: swe-forge)
├── tests/
└── deepagent/                # Real-PR factory + product datasets
    ├── README.md
    ├── src/swe_factory/      # CLI: deepagent (+ swe-factory compat)
    ├── datasets/
    │   ├── prod_hard_deepswe_med/           # CURRENT product N=9
    │   ├── panel_prod_hard_deepswe_med_m28/ # CURRENT scoreboard
    │   ├── test_n10/                        # historical M16 only
    │   └── prod_hard_keep/                  # historical softer band
    └── fixtures/             # unit shortlists only
```

---

## What you get in a Real-PR pack

Each pack under `deepagent/datasets/prod_hard_deepswe_med/tasks/<task_id>/`:

```text
task.toml                 # schema, repository_url, base_commit_hash
instruction.md            # agent-facing problem (no gold leak)
environment/Dockerfile    # agent image @ base SHA
tests/
  Dockerfile
  test.sh / grader.py
  config.json             # fail_to_pass / pass_to_pass
  test.patch              # held-out verifier tests
solution/
  solution.patch          # multi-file product sources
  solve.sh
```

Corpus-level: `pack_manifest.json`, `PRODUCT_README.md`, `PROVENANCE.md`, `coverage_stats.json`, `median_stats.json`, Docker dual-truth under `evidence/`.

---

## Platform integration

DeepAgent feeds Platform challenge validators with deterministic repository-repair tasks. Validators sample packs, run agent patches in isolated Docker workspaces, and turn fail-to-pass completion into challenge scores.

---

## Documentation

| Doc | Role |
|---|---|
| [deepagent/README.md](deepagent/README.md) | Real-PR CLI, honesty floors, ship loop |
| [prod_hard PRODUCT_README](deepagent/datasets/prod_hard_deepswe_med/PRODUCT_README.md) | Current N=9 product facts |
| [M28 panel SUMMARY](deepagent/datasets/panel_prod_hard_deepswe_med_m28/SUMMARY.md) | Grok vs Kimi 2.7-code matrix |
| [docs/architecture/](docs/architecture/README.md) | Synthetic workspace + eval |

---

## License

Apache-2.0 (root). The Real-PR factory package under `deepagent/` is MIT unless noted in that tree.
