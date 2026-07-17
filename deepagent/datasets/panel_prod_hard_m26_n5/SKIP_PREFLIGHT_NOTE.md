# skip-preflight rationale (M26)

Pier gold preflight is a **known false-negative** on these Harbor product packs
(solution apply_failed under pier oracle path even when gold is valid).

HarborDocker dual-truth is already certified for all 10 packs under
`datasets/prod_hard_keep/evidence/docker/` (sol=1, null=0).

This live M26 refresh therefore uses `--skip-preflight` with the same honesty
path as M15/M16/M19/M23, alongside `--no-reclaim` for durable jobs under
`/tmp/harbor-deepagent-jobs-prod-m26`.

Policy: M26 scoreboard is informational (M25 intrinsic hardness). Dual-model
solve-alls do not auto-drop product packs.
