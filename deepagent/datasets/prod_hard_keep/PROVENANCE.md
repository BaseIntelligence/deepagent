# PROVENANCE — datasets/prod_hard_keep (M21c production hardness)

Curated **real_pr** Harbor hardness packs only. Droplist documents 
misalign + solve-all/easy removals from `datasets/test_n10`. Dual-truth 
retained for every keep (sol=1 / null=0). No fixture pad.

**Product hardness certified N: 5** (target ≥5)

| pack_id | language | license | upstream_url | base_sha | source_track | pr |
|---|---|---|---|---|---|---:|
| `realpr-attrs-1323` | python | MIT | https://github.com/python-attrs/attrs.git | `dbb25ce34787a30e2ffa65685f6c689a269c3521` | real_pr | pr:1323 |
| `realpr-attrs-1457` | python | MIT | https://github.com/python-attrs/attrs.git | `af349876f4fb5df2b71a3f4878239b775bc89230` | real_pr | pr:1457 |
| `realpr-httpx-3672` | python | BSD-3-Clause | https://github.com/encode/httpx.git | `4acf5c2c37714cc63b5cf71b3e284fca83c90311` | real_pr | pr:3672 |
| `realpr-itemadapter-101` | python | MIT | https://github.com/scrapy/itemadapter.git | `f7860b6ec7c5b49f623ecd1f67e73877f08039b6` | real_pr | pr:101 |
| `realpr-packaging-1120` | python | Apache-2.0 | https://github.com/pypa/packaging.git | `ff14df979f865165553999d9d1a111feec6f4843` | real_pr | pr:1120 |

## Drop reasons (not product hardness N)

| pack_id | reason_code | detail |
|---|---|---|
| `realpr-charset-normalizer-715` | `solve_all_easy_policy_drop` | Solve-all / easy class: frontier panel pass@k=1.0 (both models resolved) and thin F2P=1 below MIN_F2P_NODES floor. |
| `realpr-more-itertools-1136` | `prompt_verifier_misalign` | Prompt–verifier misalignment: instruction claims version/export-only and do-not-change-runtime while F2P/gold assert runtime behaviour (windowed/unique_everseen class). |
| `realpr-more-itertools-943` | `prompt_no_runtime_claim_vs_runtime_f2p` | instruction claims do-not-change-runtime while test.patch/gold exercises runtime behavioural asserts (classes=['error_contract']) |
| `realpr-rich-3486` | `f2p_nodes_below_floor` | f2p_count=1 < MIN_F2P_NODES=3 (thin F2P refused on product hardness path); thin F2P≈1 classified as easy; anti-easy policy drop |
| `realpr-rich-4070` | `solve_all_easy_policy_drop` | Solve-all / easy class: frontier panel pass@k=1.0 (both models resolved) and thin F2P=1 below MIN_F2P_NODES floor. |

## Notes

- Source wave: `datasets/test_n10` live-mine dual-truth packs.
- Gates: prompt–verifier alignment, MIN_F2P≥3, ≥10 hunks, multi-file, dual-truth.
- Solve-all class + misalign class dropped from hardness promote.
- Legit hard solve-none kept when dual-truth+floors+align hold (model scoreout ≠ drop).
- Agent trees: public git clone@SHA; Docker oracle never `oracle_mode=fake`.
- Fixtures / hybrid archives are never product N.
