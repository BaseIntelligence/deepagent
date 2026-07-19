# DeepAgent prod_hard_deepswe_med M28 diversified scoreboard SUMMARY

**Authority:** This SUMMARY.md, `scoreboard.json`, `report.json`, and mission
`library/m28-coverage-diversity.md` are the **authoritative current product**
Grok 4.5 vs Kimi k2.7-code matrix for the **post-M28b diversified** DeepSWE-median
hardness band (N=9, max 2 packs/repo). Fulfills VAL-DCOV-007/008.

Supersedes M27e `datasets/panel_prod_hard_deepswe_med_n5` for the **current**
median product root after densify (M27e remains historical N=5 pre-diversity).
M26 `datasets/panel_prod_hard_m26_n5` remains soft-band `prod_hard_keep` history only.

Handoff `salientSummary` may be wiped by endfeature-proxy; do not re-run the
multi-hour live wave solely to repair handoff stubs. Use these durable files,
the retained jobs dir, and mission library notes instead.

## Wave
- product_root: `/projects/Agent-SWE/deepagent/datasets/prod_hard_deepswe_med` (DeepSWE-median diversified N=9)
- out: `datasets/panel_prod_hard_deepswe_med_m28`
- fidelity: `pier_miniswe_harbor`
- models: `x-ai/grok-4.5`, `moonshotai/kimi-k2.7-code` (explicit `--model` override; defaults still k2.6 for historical panels)
- k: 1
- n_concurrent: 5 (true M20 pool; actual_max_inflight=5; concurrent_pool=True; pool_workers=5)
- hard_stop_usd: 600.0
- jobs_dir: `/tmp/harbor-deepagent-jobs-prod-m28` (`--no-reclaim`)
- n_packs_requested: 9
- n_packs_scored: **9** (equals product N)
- budget_stop: False
- spend_usd: **23.63724622000000059**
- remaining_usd: 576.36275377999999941
- wall_s: 1776.562
- invented_rewards: False
- started_at: 2026-07-19T16:07:06.262834+00:00
- finished_at: 2026-07-19T16:36:42.824863+00:00
- skip_preflight: true (see SKIP_PREFLIGHT_NOTE.md)

## Diversity (product N)
- unique_repos: **7**
- max_packs_per_repo: **2**
- pack_ids:
  - realpr-click-3442
  - realpr-itemadapter-101
  - realpr-oauthlib-889
  - realpr-packaging-1120
  - realpr-packaging-1267
  - realpr-rich-3930
  - realpr-werkzeug-2637
  - realpr-werkzeug-3116
  - realpr-wtforms-923

## Dual-truth / skip-preflight note
- Pier oracle preflight is a **known false-negative** (solution apply_failed) on these Harbor packs.
- HarborDocker dual-truth (sol=1, null=0) is already certified for all 9 packs under `datasets/prod_hard_deepswe_med/evidence/docker/`.
- Live wave used `--skip-preflight` + `--no-reclaim` (same honesty path as M15/M16/M19/M23/M26/M27). See `SKIP_PREFLIGHT_NOTE.md`.

## M25 / M28 policy
- Scoreboard is **leaderboard / observational ranking only**.
- Dual-model success does **not** auto-drop hardness product packs (M25).
- Product hardness = dual-truth + alignment + **DeepSWE-median structural floors** + diversity (max 2/repo) + intrinsic non-easy.
- Soft-panel `decision=drop` is historical band stamp only — not a product curate gate.

## Quality gate: dual-solve pack rate (VAL-DCOV-007)
- dual_solve_packs: `['realpr-wtforms-923']`
- dual_solve_pack_count: 1 / 9
- **dual_solve_rate: 0.1111111111111111**
- gate: must be ≤ 0.30 (fail if ≥ 0.40)
- gate_pass: **True**

## Overall pass@1 (k=1) — observational ranking
- `x-ai/grok-4.5`: **3/9 = 0.3333333333333333**
- `moonshotai/kimi-k2.7-code`: **1/9 = 0.1111111111111111**
- Ranking: Observational on N=9: **Grok 4.5 above Kimi k2.7-code** (0.333 > 0.111).
- **Not a hard gate** if small-N noise inverts; dual-solve rate is the hardness quality gate.

## Per-pack matrix (pass@1) — all diversified median product IDs

| pack | grok-4.5 | kimi-k2.7-code | frontier | dual | panel_verdict | decision_rule |
|------|----------|----------------|----------|------|---------------|---------------|
| realpr-click-3442 | 0.0 | 0.0 | 0.0 | N | drop | solve-none |
| realpr-itemadapter-101 | 0.0 | 0.0 | 0.0 | N | drop | solve-none |
| realpr-oauthlib-889 | 0.0 | 0.0 | 0.0 | N | drop | solve-none |
| realpr-packaging-1120 | 0.0 | 0.0 | 0.0 | N | drop | solve-none |
| realpr-packaging-1267 | 0.0 | 0.0 | 0.0 | N | drop | solve-none |
| realpr-rich-3930 | 1.0 | 0.0 | 0.5 | N | keep | in-band-high-discrimination |
| realpr-werkzeug-2637 | 0.0 | 0.0 | 0.0 | N | drop | solve-none |
| realpr-werkzeug-3116 | 1.0 | 0.0 | 0.5 | N | keep | in-band-high-discrimination |
| realpr-wtforms-923 | 1.0 | 1.0 | 1.0 | Y | drop | solve-all |

Note: `panel_verdict` is the historical soft-panel keep-band stamp in the eval engine.
It is **not** a product hardness drop. M25 forbids product drops from dual solve-all alone.

## Spend / ledger
- ledger: `datasets/panel_prod_hard_deepswe_med_m28/ledger.jsonl`
- ledger_summary: `datasets/panel_prod_hard_deepswe_med_m28/ledger_summary.json`
- settled_exact_usd: 23.63724622000000059
- by_model: grok-4.5 ≈ 10.29; kimi-k2.7-code ≈ 13.35
- settled_call_count: 18 (9 packs × 2 models × k=1)
- under_cap vs 600: **True**
- has_unknown_billing: False
- invented_rewards: False

## Host mem
- before: available 102.51 / total 377.26 GiB (used 274.74)
- after: available 108.57 / total 377.26 GiB (used 268.68)

## Jobs retained
- jobs_dir: `/tmp/harbor-deepagent-jobs-prod-m28`
- trials under `_trials/` retained for trajectory audit (`--no-reclaim`)

## Artifacts
- report.json (raw live wave)
- scoreboard.json
- SUMMARY.md
- ledger.jsonl + ledger_summary.json
- SKIP_PREFLIGHT_NOTE.md
- secrets_scan.txt
- _free_before.txt / _free_after.txt
- eval_run.log (runtime; do not treat pid residue as evidence)
- **Do not commit** `eval_run.pid`

## Secrets
- secrets_scan hit_count=0 on panel out (see secrets_scan.txt)
