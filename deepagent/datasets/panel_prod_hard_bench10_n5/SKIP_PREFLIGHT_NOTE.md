# Skip preflight justification (M23 prod re-eval)

Pier oracle/gold preflight is a known false-negative on these Harbor packs
(solution apply_failed / reward=0) while HarborDockerVerifier dual-truth is
already certified sol=1/null=0 for all 10 product packs under
`datasets/prod_hard_keep/evidence/docker/*.oracle_evidence.json`.

This matches M15/M16/M19 documented honesty path used for live mini-swe waves:
prefer `--skip-preflight` only when pier gold preflight false-neg is documented
AND docker dual-truth is certified on every pack under eval.

Do not invent rewards; live mini-swe trials harvest trajectory costs and
verifier reward.json from real pier/`mini-swe-agent` runs.
