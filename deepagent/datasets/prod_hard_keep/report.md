# DeepAgent production hardness ship report (M21c)

- Generated (UTC): `2026-07-16T21:29:36.498964+00:00`
- Source corpus: `datasets/test_n10`
- Curated product path: `datasets/prod_hard_keep`
- Source track (product): **`real_pr` only**
- Certified hardness packs N: **5** (floor ≥5)
- Docker oracle: dual-truth sol=1 / null=0 retained on every keep
- Prompt–verifier alignment: required on every keep
- Hardness floors: F2P≥3, multi-file, source hunks≥10
- Status: `OK` — curated production hardness panel (no fixture pad)

## Drop reasons

| pack_id | reason_code | panel | detail |
|---|---|---|---|
| `realpr-charset-normalizer-715` | `solve_all_easy_policy_drop` | solve-all | Solve-all / easy class: frontier panel pass@k=1.0 (both models resolved) and thin F2P=1 below MIN_F2P_NODES floor. |
| `realpr-more-itertools-1136` | `prompt_verifier_misalign` | solve-none | Prompt–verifier misalignment: instruction claims version/export-only and do-not-change-runtime while F2P/gold assert runtime behaviour (windowed/unique_everseen class). |
| `realpr-more-itertools-943` | `prompt_no_runtime_claim_vs_runtime_f2p` | in-band-high-discrimination | instruction claims do-not-change-runtime while test.patch/gold exercises runtime behavioural asserts (classes=['error_contract']) |
| `realpr-rich-3486` | `f2p_nodes_below_floor` | in-band-high-discrimination | f2p_count=1 < MIN_F2P_NODES=3 (thin F2P refused on product hardness path); thin F2P≈1 classified as easy; anti-easy policy drop |
| `realpr-rich-4070` | `solve_all_easy_policy_drop` | solve-all | Solve-all / easy class: frontier panel pass@k=1.0 (both models resolved) and thin F2P=1 below MIN_F2P_NODES floor. |

## Certified hardness keeps (real_pr)

- `realpr-attrs-1323` reason=`keep_hard_panel_band` f2p=4 hunks=14 sol=1 null=0 panel=in-band-high-discrimination/keep frontier=0.5
- `realpr-attrs-1457` reason=`keep_legit_hard_solve_none` f2p=14 hunks=21 sol=1 null=0 panel=solve-none/drop frontier=0.0
- `realpr-httpx-3672` reason=`keep_hard_panel_band` f2p=5 hunks=18 sol=1 null=0 panel=in-band-high-discrimination/keep frontier=0.5
- `realpr-itemadapter-101` reason=`keep_hard_panel_band` f2p=43 hunks=15 sol=1 null=0 panel=in-band-high-discrimination/keep frontier=0.5
- `realpr-packaging-1120` reason=`keep_legit_hard_solve_none` f2p=9 hunks=24 sol=1 null=0 panel=solve-none/drop frontier=0.0

## Gates (no relaxation)

- dual-truth HarborDocker sol=1 / null=0
- prompt–verifier alignment fail-closed
- hardness floors F2P≥3, multi-file≥2, hunks≥10
- anti-easy: solve-all + thin F2P dropped
- no fixture pad; hybrid never product N
- re-upload HF BaseIntelligence/deepagent revision `test`

## Assertions

- VAL-DHARD-004 curated production N to HF test
- VAL-DHARD-002 / VAL-DHARD-003 floors + anti-easy
