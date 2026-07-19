# skip-preflight rationale (M28 diversified DeepSWE-median)

Pier gold preflight is a **known false-negative** on these Harbor product packs
(solution apply_failed under pier oracle path even when gold is valid).

HarborDocker dual-truth is already certified for all N packs under
`datasets/prod_hard_deepswe_med/evidence/docker/` (sol=1, null=0).

This live M28 diversified scoreboard therefore uses `--skip-preflight` with the
same honesty path as M15/M16/M19/M23/M26/M27, alongside `--no-reclaim` for durable
jobs under `/tmp/harbor-deepagent-jobs-prod-m28`.

Models: `x-ai/grok-4.5` + `moonshotai/kimi-k2.7-code` (explicit `--model` override;
defaults still k2.6 for historical panels).

Product N=9 post-M28b diversity (max 2 packs/repo, unique_repos=7).
Quality gate: dual_solve_rate ≤ 0.30 (fail if ≥ 0.40). Ranking observational.
M25 intrinsic: dual-model solve-alls do not auto-drop product packs.
