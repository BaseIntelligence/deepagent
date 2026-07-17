# PROVENANCE — datasets/prod_hard_keep (M21c production hardness)

Curated **real_pr** Harbor hardness packs only. Droplist documents 
misalign + solve-all/easy removals from `datasets/test_n10`. Dual-truth 
retained for every keep (sol=1 / null=0). No fixture pad.

**Product hardness certified N: 7** (target ≥5)

| pack_id | language | license | upstream_url | base_sha | source_track | pr |
|---|---|---|---|---|---|---:|
| `realpr-attrs-1323` | python | MIT | https://github.com/python-attrs/attrs.git | `dbb25ce34787a30e2ffa65685f6c689a269c3521` | real_pr | pr:1323 |
| `realpr-attrs-1457` | python | MIT | https://github.com/python-attrs/attrs.git | `af349876f4fb5df2b71a3f4878239b775bc89230` | real_pr | pr:1457 |
| `realpr-httpx-3672` | python | BSD-3-Clause | https://github.com/encode/httpx.git | `4acf5c2c37714cc63b5cf71b3e284fca83c90311` | real_pr | pr:3672 |
| `realpr-itemadapter-101` | python | MIT | https://github.com/scrapy/itemadapter.git | `f7860b6ec7c5b49f623ecd1f67e73877f08039b6` | real_pr | pr:101 |
| `realpr-packaging-1120` | python | Apache-2.0 | https://github.com/pypa/packaging.git | `ff14df979f865165553999d9d1a111feec6f4843` | real_pr | pr:1120 |
| `realpr-qs-487` | javascript | BSD-3-Clause | https://github.com/ljharb/qs.git | `04f422fe91985103d2fdca0280ee362ecf5e43f2` | real_pr | pr:487 |
| `realpr-qs-488` | javascript | BSD-3-Clause | https://github.com/ljharb/qs.git | `5f0449fff1d9fb236d297cd0d3650b42d2d93b8a` | real_pr | pr:488 |

## Drop reasons (not product hardness N)

| pack_id | reason_code | detail |
|---|---|---|
| `realpr-werkzeug-2979` | `solve_all_easy_policy_drop` | EASY_SOLVE_ALL: all 2 panel model(s) pass@1=1.0 (moonshotai/kimi-k2.6=1.0, x-ai/grok-4.5=1.0); frontier=1.0; auto-drop from hardness without name hardcoding (VAL-DEASY-002) |
| `realpr-werkzeug-3006` | `solve_all_easy_policy_drop` | EASY_SOLVE_ALL: all 2 panel model(s) pass@1=1.0 (moonshotai/kimi-k2.6=1.0, x-ai/grok-4.5=1.0); frontier=1.0; auto-drop from hardness without name hardcoding (VAL-DEASY-002) |
| `realpr-werkzeug-3101` | `solve_all_easy_policy_drop` | EASY_SOLVE_ALL: all 2 panel model(s) pass@1=1.0 (moonshotai/kimi-k2.6=1.0, x-ai/grok-4.5=1.0); frontier=1.0; auto-drop from hardness without name hardcoding (VAL-DEASY-002) |

## Notes

- Source wave: `datasets/test_n10` live-mine dual-truth packs.
- Gates: prompt–verifier alignment, MIN_F2P≥3, ≥10 hunks, multi-file, dual-truth.
- Solve-all class + misalign class dropped from hardness promote.
- Legit hard solve-none kept when dual-truth+floors+align hold (model scoreout ≠ drop).
- Agent trees: public git clone@SHA; Docker oracle never `oracle_mode=fake`.
- Fixtures / hybrid archives are never product N.
