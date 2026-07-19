# PROVENANCE — datasets/prod_hard_deepswe_med (DeepSWE-median product)

Corpus of Docker-oracle-certified **real_pr** Harbor packs for the M27 DeepSWE-median
hardness band, densified under M28 diversity (max 2 packs/repo). Hybrid motors live under
`datasets/deepagent_v1_hybrid_archive/` (historical; never counted here as product N).
Soft historical band remains under `datasets/prod_hard_keep` (audit only).
Each row is one certified keep. Copyleft / unknown-license candidates are fail-closed
and never appear here.

| pack_id | language | license | upstream_url | base_sha | source_track | pr |
|---|---|---|---|---|---|---:|
| `realpr-itemadapter-101` | python | MIT | https://github.com/scrapy/itemadapter.git | `f7860b6ec7c5b49f623ecd1f67e73877f08039b6` | real_pr | pr:101 |
| `realpr-packaging-1120` | python | Apache-2.0 | https://github.com/pypa/packaging.git | `ff14df979f865165553999d9d1a111feec6f4843` | real_pr | pr:1120 |
| `realpr-rich-3930` | python | MIT | https://github.com/Textualize/rich.git | `53757bc234cf18977cade41a5b64f3abaccb0b85` | real_pr | pr:3930 |
| `realpr-wtforms-923` | python | MIT | https://github.com/pallets-eco/wtforms.git | `b68f4d2bbd04fb6b6b06451e22d0457305fadaf5` | real_pr | pr:923 |
| `realpr-werkzeug-2637` | python | MIT | https://github.com/pallets/werkzeug.git | `4f8fddd2ec527d70331e35440fa14edb377dcff0` | real_pr | pr:2637 |
| `realpr-werkzeug-3116` | python | MIT | https://github.com/pallets/werkzeug.git | `a792bc2d1ebd52abfd285db847ef0fd42a911df9` | real_pr | pr:3116 |
| `realpr-oauthlib-889` | python | MIT | https://github.com/oauthlib/oauthlib.git | `1fd5253630c03e3f12719dd8c13d43111f66a8d2` | real_pr | pr:889 |
| `realpr-click-3442` | python | MIT | https://github.com/pallets/click.git | `d5fbd32842da361cc9be8658d94a64e9cc417fb5` | real_pr | pr:3442 |

**Product certified N (real_pr only): 8**
**unique_repos: 7** | **max_packs_per_repo: 2**
**fixture_pad: false** | **floors_band: deepswe_median_m27**

## M28b densify notes

m28b retry densify: host-venv SUT-shadow uninstall + SOCKS-free host pip; pytest collection-error capture + worktree-prefix nodeid strip unlock dual F2P; GH Archive 24 hourly dumps under datasets/gh_archive_m28b; Docker dual-truth keeps N=8 repos=7 max/repo=2; click-3442 side-certified with trimmed F2P + empty P2P (docker ambient P2P flake); no fixture pad.

Dual-truth evidence: `evidence/docker/<task_id>.{sol,null}.reward.json` (reward 1/0).
Coverage: `coverage_stats.json`. Gate audit: `gate_audit.jsonl` + `gate_audit_summary.json`.

