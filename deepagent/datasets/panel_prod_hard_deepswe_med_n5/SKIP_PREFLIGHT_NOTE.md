# skip-preflight rationale (M27 DeepSWE-median)

Pier gold preflight is a **known false-negative** on these Harbor product packs
(solution apply_failed under pier oracle path even when gold is valid).

HarborDocker dual-truth is already certified for all 5 DeepSWE-median packs under
`datasets/prod_hard_deepswe_med/evidence/docker/` (sol=1, null=0).

This live M27 scoreboard therefore uses `--skip-preflight` with the same honesty
path as M15/M16/M19/M23/M26, alongside `--no-reclaim` for durable jobs under
`/tmp/harbor-deepagent-jobs-prod-m27`.

Models: `x-ai/grok-4.5` + `moonshotai/kimi-k2.7-code` (explicit `--model` override;
defaults still k2.6 for historical panels).

Policy: M27 scoreboard is the authoritative *current median product* Grok vs Kimi
matrix (supersedes M26 `panel_prod_hard_m26_n5` for the median product root).
M25 intrinsic hardness still applies: dual-model solve-alls do not auto-drop product.
Dual-solve pack rate quality gate: ≤0.30 (fail if ≥0.40).
