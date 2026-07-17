# DeepAgent prod_hard_keep re-eval (M23) summary

**Authority:** This SUMMARY.md, `scoreboard.json`, and mission
`library/m23-prod-reeval.md` are the **authoritative** Grok vs Kimi
scoreboard for VAL-DREEVAL. Handoff `salientSummary` may be wiped by
endfeature-proxy; do not re-run the multi-hour live wave solely to repair
handoff stubs. Use these durable files instead.

## Wave
- product_root: `/projects/Agent-SWE/deepagent/datasets/prod_hard_keep`
- out: `datasets/panel_prod_hard_bench10_n5`
- fidelity: `pier_miniswe_harbor`
- models: x-ai/grok-4.5, moonshotai/kimi-k2.6
- k: 1
- n_concurrent: 5 (true M20 pool; actual_max_inflight=5; concurrent_pool=True)
- hard_stop_usd: 600.0
- jobs_dir: `/tmp/harbor-deepagent-jobs-prod-reeval` (`--no-reclaim`)
- n_packs_requested: 10
- n_packs_scored: 10
- budget_stop: False
- stop_reason: None
- spend_usd: 51.75201631599999884
- remaining_usd: 548.24798368400000116
- wall_s: 3225.899
- invented_rewards: False
- started_at: 2026-07-17T01:24:21.032060+00:00
- finished_at: 2026-07-17T02:18:06.930871+00:00

## Dual-truth note
- Pier oracle preflight remains a known false-negative (solution apply_failed) on these Harbor packs.
- HarborDocker dual-truth (sol=1,null=0) certified for all 10 packs under `datasets/prod_hard_keep/evidence/docker/`.
- Live wave used `--skip-preflight` + `--no-reclaim` (same honesty path as M15/M16/M19).

## Overall pass@1 (k=1)
- `x-ai/grok-4.5`: **5/10 = 0.500**
- `moonshotai/kimi-k2.6`: **4/10 = 0.400**
- aggregate (both models): **9/20 = 0.450**

## Per-pack matrix (pass@1)
| pack | grok-4.5 | kimi-k2.6 | frontier | verdict |
|------|----------|-----------|----------|---------|
| realpr-itemadapter-101 | 1.0 | 0.0 | 0.5 | keep |
| realpr-attrs-1323 | 0.0 | 1.0 | 0.5 | keep |
| realpr-httpx-3672 | 0.0 | 0.0 | 0.0 | drop |
| realpr-packaging-1120 | 0.0 | 0.0 | 0.0 | drop |
| realpr-attrs-1457 | 0.0 | 0.0 | 0.0 | drop |
| realpr-qs-487 | 1.0 | 0.0 | 0.5 | keep |
| realpr-qs-488 | 0.0 | 0.0 | 0.0 | drop |
| realpr-werkzeug-2979 | 1.0 | 1.0 | 1.0 | drop |
| realpr-werkzeug-3006 | 1.0 | 1.0 | 1.0 | drop |
| realpr-werkzeug-3101 | 1.0 | 1.0 | 1.0 | drop |

## Spend / ledger
- ledger path: `datasets/panel_prod_hard_bench10_n5/ledger.jsonl`
- settled_call_count: 20
- settled_exact_usd: 51.75201631599999884
- open_reserved_usd: 0
- remaining_usd: 548.24798368400000116
- under_cap: True
- unknown_billing_count: 0
- has_unknown_billing: False

## Trial notes
- All trials produced rewards and/or cost settlements without invented_rewards.

## Jobs retained
- jobs_dir exists: True
- trial dirs: 20
- reward.json count: 20

## Artifacts
- report.json
- scoreboard.json
- ledger.jsonl + ledger_summary.json
- SKIP_PREFLIGHT_NOTE.md
- _free_before.txt / _free_after.txt
- eval_run.log
