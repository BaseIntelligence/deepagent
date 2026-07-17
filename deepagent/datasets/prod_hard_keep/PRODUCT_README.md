# datasets/prod_hard_keep — real_pr live-mine generate wave

This directory ships **source_track=real_pr** Harbor packs only.
Hybrid motor corpus: `datasets/deepagent_v1_hybrid_archive/`.
Prior seed product: `datasets/deepagent_v1_seed5_archive/`.
Certified N this wave: **10** (target=10, min≥5).
Materials: datasets/live_materials_m22 (live_mine=True; never fixtures/real_pr_ship for product).
Docker oracle: HarborDockerVerifier sol=1 / null=0; pier mode honest;
gate_audit dual-truth pass required before overwrite (VAL-LSHIP-007 / VAL-DGEN).
Hardness floors (VAL-DHARD-002): F2P≥3 (default MIN_F2P_NODES=3), source hunks≥10, multi-file sources; thin F2P=1 refused; solve-all class dropped from hardness promote (VAL-DHARD-003). See docs/PRODUCT_HARDNESS.md.

## M23 re-eval scoreboard (Grok 4.5 vs Kimi 2.6)

Authoritative dual-model live scoreboard lives under
`datasets/panel_prod_hard_bench10_n5/` (see also mission
`library/m23-prod-reeval.md`).

| Machine | pass@1 (k=1, n_concurrent=5 true pool) |
|---|---|
| x-ai/grok-4.5 | **5/10 = 0.500** |
| moonshotai/kimi-k2.6 | **4/10 = 0.400** |
| aggregate | 9/20 = 0.450 |

| pack | grok | kimi | frontier |
|---|---:|---:|---:|
| itemadapter-101 | 1 | 0 | 0.5 |
| attrs-1323 | 0 | 1 | 0.5 |
| httpx-3672 | 0 | 0 | 0.0 |
| packaging-1120 | 0 | 0 | 0.0 |
| attrs-1457 | 0 | 0 | 0.0 |
| qs-487 | 1 | 0 | 0.5 |
| qs-488 | 0 | 0 | 0.0 |
| werkzeug-2979 | 1 | 1 | 1.0 |
| werkzeug-3006 | 1 | 1 | 1.0 |
| werkzeug-3101 | 1 | 1 | 1.0 |

Spend ≈ $51.75 / $600 hard-stop; invented_rewards=false; fidelity
`pier_miniswe_harbor`. Full matrix + ledger: panel `SUMMARY.md` /
`scoreboard.json`.
