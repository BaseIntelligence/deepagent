# DeepAgent live bench10 n5 summary

## Wave
- product_root: `/projects/Agent-SWE/deepagent/datasets/test_n10`
- out: `/projects/Agent-SWE/deepagent/datasets/panel_deepagent_bench10_n5`
- fidelity: `pier_miniswe_harbor`
- models: x-ai/grok-4.5, moonshotai/kimi-k2.6
- k: 1
- n_concurrent: 5 (CLI/report; M19 wave launched sequential; M20 true ThreadPoolExecutor pool for n>1 is now product path)
- hard_stop_usd: 600.0
- n_packs_requested: 10
- n_packs_scored: 10
- budget_stop: False
- stop_reason: None
- spend_usd: 33.72599655000000386
- remaining_usd: 566.27400344999999614
- wall_s: 13473.02 (~3.74 h)
- invented_rewards: False
- started_at: 2026-07-16T13:46:10.027672+00:00
- finished_at: 2026-07-16T17:30:43.047738+00:00

## Dual-truth note
- Pier oracle preflight scores solution as apply_failed/reward=0 on these packs.
- HarborDocker dual-truth (sol=1,null=0) is recorded for all 10 packs under `datasets/test_n10/evidence/docker/` and rechecked live on itemadapter.
- This live mini-swe wave used `--skip-preflight` after that docker dual-truth gate (same honesty path as prior M15/M16 waves).

## Overall pass@1 (k=1)
- `x-ai/grok-4.5`: **5/10 = 0.500**
- `moonshotai/kimi-k2.6`: **4/10 = 0.400**
- aggregate (both models): **9/20 = 0.450**

## Per-pack matrix (pass@1)
| pack | grok-4.5 | kimi-k2.6 | frontier | verdict |
|------|----------|-----------|----------|---------|
| realpr-itemadapter-101 | 1.0 | 0.0 | 0.5 | keep |
| realpr-attrs-1323 | 0.0 | 1.0 | 0.5 | keep |
| realpr-httpx-3672 | 1.0 | 0.0 | 0.5 | keep |
| realpr-packaging-1120 | 0.0 | 0.0 | 0.0 | drop |
| realpr-attrs-1457 | 0.0 | 0.0 | 0.0 | drop |
| realpr-charset-normalizer-715 | 1.0 | 1.0 | 1.0 | drop |
| realpr-more-itertools-1136 | 0.0 | 0.0 | 0.0 | drop |
| realpr-more-itertools-943 | 0.0 | 1.0 | 0.5 | keep |
| realpr-rich-3486 | 1.0 | 0.0 | 0.5 | keep |
| realpr-rich-4070 | 1.0 | 1.0 | 1.0 | drop |

## Spend / ledger
- ledger path: `datasets/panel_deepagent_bench10_n5/ledger.jsonl`
- settled_call_count: 20
- settled_exact_usd: 33.72599655000000386
- open_reserved_usd: 0
- under_cap: True
- has_unknown_billing: False

## Trial anomalies
- `realpr-packaging-1120` / `moonshotai/kimi-k2.6`: pier timeout after 3600s, reward=null, cost_usd=0, solved=false (honest fail; not invented).
- Remaining 19 trials produced verifier rewards and/or non-zero trajectory cost settlements as applicable.
- M19 bench committed CLI `n_concurrent=5` while the then-current core loop still launched trials sequentially (observed concurrent pier/docker ≤1 in that monitoring). **M20 true pier pool:** `eval_deepagent` now uses `ThreadPoolExecutor(max_workers=n_concurrent)` with per-trial job dirs, thread-safe ledger reserve/settle, hard-stop before schedule, and report fields `actual_max_inflight` / `concurrent_pool=true`. Default `n_concurrent=1` remains serial-equivalent via the same pool (`max_workers=1`).

## Artifacts
- `report.json`
- `ledger.jsonl` + `ledger_summary.json`
- `eval_run.log` (CLI --json stdout)
- `preflight_only_failed/` first attempt without skip-preflight (n_scored=0, pier apply_failed)

