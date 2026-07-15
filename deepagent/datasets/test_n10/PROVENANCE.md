# PROVENANCE — datasets/deepagent_v1 (Real-PR product)

Corpus of Docker-oracle-certified **real_pr** Harbor packs only.
Hybrid motors live under `datasets/deepagent_v1_hybrid_archive/` (historical; never counted here as product N).
Each row is one certified keep. Copyleft / unknown-license candidates
are fail-closed and never appear here.

| pack_id | language | license | upstream_url | base_sha | source_track | pr |
|---|---|---|---|---|---|---:|
| `realpr-packaging-1120` | python | Apache-2.0 | https://github.com/pypa/packaging.git | `ff14df979f865165553999d9d1a111feec6f4843` | real_pr | pr:1120 |
| `realpr-charset-normalizer-715` | python | MIT | https://github.com/jawah/charset_normalizer.git | `7411396ebd495e1abc28f5682975b5c662b2ff35` | real_pr | pr:715 |
| `realpr-itemadapter-101` | python | MIT | https://github.com/scrapy/itemadapter.git | `f7860b6ec7c5b49f623ecd1f67e73877f08039b6` | real_pr | pr:101 |
| `realpr-attrs-1323` | python | MIT | https://github.com/python-attrs/attrs.git | `dbb25ce34787a30e2ffa65685f6c689a269c3521` | real_pr | pr:1323 |
| `realpr-rich-3486` | python | MIT | https://github.com/Textualize/rich.git | `d0de442c08df8793c5ff36d9ad322ca5c47fc38c` | real_pr | pr:3486 |
| `realpr-attrs-1457` | python | MIT | https://github.com/python-attrs/attrs.git | `af349876f4fb5df2b71a3f4878239b775bc89230` | real_pr | pr:1457 |
| `realpr-httpx-3672` | python | BSD-3-Clause | https://github.com/encode/httpx.git | `4acf5c2c37714cc63b5cf71b3e284fca83c90311` | real_pr | pr:3672 |
| `realpr-more-itertools-1136` | python | MIT | https://github.com/more-itertools/more-itertools.git | `e4d2a4a2a97246a73856754b2c4866d7f41d4875` | real_pr | pr:1136 |
| `realpr-more-itertools-943` | python | MIT | https://github.com/more-itertools/more-itertools.git | `f36c88fe03688fa442154ef14f429bcfa4c38525` | real_pr | pr:943 |
| `realpr-rich-4070` | python | MIT | https://github.com/Textualize/rich.git | `fc41075a3206d2a5fd846c6f41c4d2becab814fa` | real_pr | pr:4070 |

**Product certified N (real_pr only): 10**

## Notes

- Product surface: `datasets/deepagent_v1` (source_track=real_pr only).
- Hybrid archive (historical only): `datasets/deepagent_v1_hybrid_archive/`.
- Fixtures (non-product): `datasets/harbor_v1`, `datasets/v1`.
- Agent trees clone public git @ base SHA (no motor COPY hybrid_bind).
- Docker oracle dual truth: sol=1 / null=0 (never `oracle_mode=fake`).
