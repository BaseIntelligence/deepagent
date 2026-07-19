# DeepAgent prod_hard_deepswe_med scoreboard (M27) SUMMARY

**Authority:** This SUMMARY.md, `scoreboard.json`, and mission
`library/m27-deepswe-median.md` are the **authoritative current product**
Grok 4.5 vs Kimi k2.7-code matrix for the DeepSWE-median hardness band
(VAL-DMED-007/008). M26 `datasets/panel_prod_hard_m26_n5` remains the
historical softer `prod_hard_keep` matrix only.

Handoff `salientSummary` may be wiped by endfeature-proxy; do not re-run the
live wave solely to repair handoff stubs. Use these durable files, the retained
jobs dir, and mission library notes instead.

## Wave
- product_root: `/projects/Agent-SWE/deepagent/datasets/prod_hard_deepswe_med` (DeepSWE-median N=5)
- out: `datasets/panel_prod_hard_deepswe_med_n5`
- fidelity: `pier_miniswe_harbor`
- models: x-ai/grok-4.5, moonshotai/kimi-k2.7-code (explicit `--model` override; defaults still k2.6 for historical panels)
- k: 1
- n_concurrent: 5 (true M20 pool; actual_max_inflight=5; concurrent_pool=True)
- hard_stop_usd: 600.0
- jobs_dir: `/tmp/harbor-deepagent-jobs-prod-m27` (`--no-reclaim`)
- jobs_dir_retry_grok: `/tmp/harbor-deepagent-jobs-prod-m27-retry-grok` (2 grok mintag-race retries)
- n_packs_requested: 5
- n_packs_scored: **5**
- budget_stop: False
- spend_usd (primary+retry_grok): 30.5769712300000015
- primary_wave_spend_usd: 24.3892764300000005
- retry_grok_spend_usd: 6.187694800000001
- remaining_usd: 569.4230287699999985
- wall_s primary: 1456.193
- wall_s retry_grok: 1345.217
- invented_rewards: False
- started_at: 2026-07-19T01:40:27.399470+00:00
- finished_at: 2026-07-19T02:29:20.096151+00:00

## Dual-truth / skip-preflight note
- Pier oracle preflight is a **known false-negative** (solution apply_failed) on these Harbor packs.
- HarborDocker dual-truth (sol=1, null=0) is already certified for all 5 packs under `datasets/prod_hard_deepswe_med/evidence/docker/`.
- Live wave used `--skip-preflight` + `--no-reclaim` (same honesty path as M15/M16/M19/M23/M26). See `SKIP_PREFLIGHT_NOTE.md`.

## Merge note (mintag race)
- Concurrent n=5 agent mintag docker export raced with `image already exists` on two packs for grok only at wave start.
- Failed: `realpr-itemadapter-101` + `realpr-packaging-1120` grok (reward=None, cost=$0 error settle).
- Retried those two packs (n_concurrent=2) after images existed; grok rewards merged into this scoreboard.
- Kimi primary trial rewards retained (retry kimi not used for matrix).

## M25 / M27 policy
- Scoreboard is **leaderboard / observational ranking only**.
- Dual-model success does **not** auto-drop hardness product packs (M25).
- Product hardness = dual-truth + alignment + **DeepSWE-median structural floors** + intrinsic non-easy.
- Soft-panel `decision=drop` is historical band stamp only.

## Quality gate: dual-solve pack rate
- dual_solve_packs: ['realpr-werkzeug-3116']
- dual_solve_pack_count: 1 / 5
- **dual_solve_rate: 0.2000**
- gate: must be ≤ 0.30 (fail if ≥ 0.40)
- gate_pass: **True**

## Overall pass@1 (k=1) — observational ranking
- `x-ai/grok-4.5`: **2/5 = 0.400**
- `moonshotai/kimi-k2.7-code`: **1/5 = 0.200**
- Ranking: Observational on N=5: **Grok 4.5 above Kimi k2.7-code** (0.400 > 0.200).
- **Not a hard gate** if small-N noise inverts; dual-solve rate is the hardness quality gate.

## DeepSWE public leaderboard comparison note
- Public DeepSWE sample / LB reference (full ~113 tasks, different harness scale):
  - grok-4.5[high] ≈ **53.8%**
  - kimi-k2-7-code ≈ **30.5%** (SKU family peer of `moonshotai/kimi-k2.7-code`)
- This median product N=5: Grok **40.0%**, Kimi2.7 **20.0%** (directionally consistent with Grok ≥ Kimi2.7; absolute rates not comparable 1:1 to full 113).
- Structural band intent: DeepSWE median floors (files≥4/hybrid3, hunks≥14, added≥400, F2P≥5) so dual-solve rate stays non-soft.

## Per-pack matrix (pass@1) — all median product IDs
| pack | grok-4.5 | kimi-k2.7-code | frontier | dual | panel_verdict |
|------|----------|----------------|----------|------|---------------|
| realpr-itemadapter-101 (grok merged-retry) | 1.0 | 0.0 | 1.0 | N | keep |
| realpr-packaging-1120 (grok merged-retry) | 0.0 | 0.0 | 0.0 | N | drop |
| realpr-werkzeug-2608 | 0.0 | 0.0 | 0.0 | N | drop |
| realpr-werkzeug-2637 | 0.0 | 0.0 | 0.0 | N | drop |
| realpr-werkzeug-3116 | 1.0 | 1.0 | 1.0 | Y | drop |

Note: `panel_verdict` is the historical soft-panel keep-band stamp in the eval engine.
It is **not** a product hardness drop. M25 forbids product drops from dual solve-all alone.

## Spend / ledger
- primary ledger: `datasets/panel_prod_hard_deepswe_med_n5/ledger.jsonl`
- retry ledger: `datasets/panel_prod_hard_deepswe_med_n5_retry_grok/ledger.jsonl`
- combined spend (primary + retry grok only): 30.5769712300000015
- under_cap vs 600: **True**
- invented_rewards: False

## Jobs retained
- jobs_dir: `/tmp/harbor-deepagent-jobs-prod-m27`
- primary trial dirs: 12
- retry jobs: `/tmp/harbor-deepagent-jobs-prod-m27-retry-grok`

## Artifacts
- report.json (primary wave raw)
- merged_report.json (post grok-retry merge)
- scoreboard.json
- SUMMARY.md
- ledger.jsonl + ledger_summary.json
- SKIP_PREFLIGHT_NOTE.md
- _free_before.txt / _free_after.txt
- secrets_scan.txt

