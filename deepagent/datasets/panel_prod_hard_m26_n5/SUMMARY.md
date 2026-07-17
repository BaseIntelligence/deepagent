# DeepAgent prod_hard_keep scoreboard refresh (M26) SUMMARY

**Authority:** This SUMMARY.md, `scoreboard.json`, and mission
`library/m26-scoreboard-refresh.md` are the **authoritative current product**
Grok vs Kimi matrix for VAL-DREFRESH. M23
`datasets/panel_prod_hard_bench10_n5` remains **historical** only.

Handoff `salientSummary` may be wiped by endfeature-proxy; do not re-run the
multi-hour live wave solely to repair handoff stubs. Use these durable files,
the retained jobs dir, and mission library notes instead.

## Wave
- product_root: `/projects/Agent-SWE/deepagent/datasets/prod_hard_keep` (post-M25 N=10; werkzeug restored)
- out: `datasets/panel_prod_hard_m26_n5`
- fidelity: `pier_miniswe_harbor`
- models: x-ai/grok-4.5, moonshotai/kimi-k2.6
- k: 1
- n_concurrent: 5 (true M20 pool; actual_max_inflight=5; concurrent_pool=True)
- hard_stop_usd: 600.0
- jobs_dir: `/tmp/harbor-deepagent-jobs-prod-m26` (`--no-reclaim`)
- n_packs_requested: 10
- n_packs_scored: **10**
- budget_stop: False
- stop_reason: None
- spend_usd: 45.7685217899999983
- remaining_usd: 554.2314782100000017
- wall_s: 2438.201
- invented_rewards: False
- started_at: 2026-07-17T14:58:22.125896+00:00
- finished_at: 2026-07-17T15:39:00.326421+00:00

## Dual-truth / skip-preflight note
- Pier oracle preflight is a **known false-negative** (solution apply_failed) on these Harbor packs.
- HarborDocker dual-truth (sol=1, null=0) is already certified for all 10 packs under `datasets/prod_hard_keep/evidence/docker/`.
- Live wave used `--skip-preflight` + `--no-reclaim` (same honesty path as M15/M16/M19/M23). See `SKIP_PREFLIGHT_NOTE.md`.

## M25 / M26 policy (informational only)
- This refresh is a **leaderboard / scoreboard only**.
- Dual-model success (frontier=1.0 / EASY_SOLVE_ALL) **does not** auto-drop hardness product packs.
- Prod hardness continues under dual-truth + alignment + floors + intrinsic request/patch (M25).
- EasyDetect was re-run labels-only: `easy_detect_labels.json` (all `should_drop_hardness=False` for solve-alls).
- **No product pack was dropped solely because both models solved.**

## Overall pass@1 (k=1)
- `x-ai/grok-4.5`: **5/10 = 0.500**
- `moonshotai/kimi-k2.6`: **5/10 = 0.500**
- aggregate (both models): **10/20 = 0.500**

## Per-pack matrix (pass@1) — all 10 current prod IDs
| pack | grok-4.5 | kimi-k2.6 | frontier | panel_verdict |
|------|----------|-----------|----------|---------------|
| realpr-itemadapter-101 | 1.0 | 0.0 | 0.5 | keep |
| realpr-attrs-1323 | 0.0 | 1.0 | 0.5 | keep |
| realpr-httpx-3672 | 0.0 | 0.0 | 0.0 | drop |
| realpr-packaging-1120 | 0.0 | 0.0 | 0.0 | drop |
| realpr-attrs-1457 | 0.0 | 0.0 | 0.0 | drop |
| realpr-qs-487 | 1.0 | 1.0 | 1.0 | drop |
| realpr-qs-488 | 0.0 | 0.0 | 0.0 | drop |
| realpr-werkzeug-2979 | 1.0 | 1.0 | 1.0 | drop |
| realpr-werkzeug-3006 | 1.0 | 1.0 | 1.0 | drop |
| realpr-werkzeug-3101 | 1.0 | 1.0 | 1.0 | drop |

Note: `panel_verdict` is the historical soft-panel keep-band stamp in the eval engine.
It is **not** a product hardness drop. M25 forbids product drops from dual solve-all alone.

## EasyDetect labels-only (M26 post-scoreboard)
- `realpr-itemadapter-101`: reason_code=`one_sided_discrimination_keep` all_models_solved=False should_drop_hardness=False
- `realpr-attrs-1323`: reason_code=`one_sided_discrimination_keep` all_models_solved=False should_drop_hardness=False
- `realpr-httpx-3672`: reason_code=`not_easy_pass_matrix` all_models_solved=False should_drop_hardness=False
- `realpr-packaging-1120`: reason_code=`not_easy_pass_matrix` all_models_solved=False should_drop_hardness=False
- `realpr-attrs-1457`: reason_code=`not_easy_pass_matrix` all_models_solved=False should_drop_hardness=False
- `realpr-qs-487`: reason_code=`solve_all_easy_policy_drop` all_models_solved=True should_drop_hardness=False
- `realpr-qs-488`: reason_code=`not_easy_pass_matrix` all_models_solved=False should_drop_hardness=False
- `realpr-werkzeug-2979`: reason_code=`solve_all_easy_policy_drop` all_models_solved=True should_drop_hardness=False
- `realpr-werkzeug-3006`: reason_code=`solve_all_easy_policy_drop` all_models_solved=True should_drop_hardness=False
- `realpr-werkzeug-3101`: reason_code=`solve_all_easy_policy_drop` all_models_solved=True should_drop_hardness=False

Solve-all packs (reporting label only): realpr-qs-487, realpr-werkzeug-2979, realpr-werkzeug-3006, realpr-werkzeug-3101.

## Spend / ledger
- ledger path: `datasets/panel_prod_hard_m26_n5/ledger.jsonl`
- settled_call_count: 20
- settled_exact_usd: 45.7685217899999983
- open_reserved_usd: 0
- remaining_usd: 554.2314782100000017
- under_cap: True
- unknown_billing_count: 0
- has_unknown_billing: False

## Jobs retained
- jobs_dir: `/tmp/harbor-deepagent-jobs-prod-m26`
- trial dirs: 20
- reward.json count: 20

## Artifacts
- report.json
- scoreboard.json
- SUMMARY.md
- ledger.jsonl + ledger_summary.json
- easy_detect_labels.json
- SKIP_PREFLIGHT_NOTE.md
- _free_before.txt / _free_after.txt
- eval_run.log
