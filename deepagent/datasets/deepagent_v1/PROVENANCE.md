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
| `realpr-click-3645` | python | BSD-3-Clause | https://github.com/pallets/click.git | `679a7a0eccbdded7a6e85680bdaaf08003765e01` | real_pr | pr:3645 |
| `realpr-httpcore-882` | python | MIT | https://github.com/encode/httpcore.git | `c46802478cdd8a82ee8cb333420080fab1aed00b` | real_pr | pr:882 |
| `realpr-werkzeug-2995` | python | MIT | https://github.com/pallets/werkzeug.git | `1a1728ed88939ca68928dade168e1989be062c6f` | real_pr | pr:2995 |
| `realpr-werkzeug-2979` | python | MIT | https://github.com/pallets/werkzeug.git | `862cb193c2b13db860d886725fa4235173d0dfcd` | real_pr | pr:2979 |
| `realpr-werkzeug-3006` | python | MIT | https://github.com/pallets/werkzeug.git | `cb307c144e7b9092bf72b1a1dba5281e7c6ff838` | real_pr | pr:3006 |
| `realpr-werkzeug-3101` | python | MIT | https://github.com/pallets/werkzeug.git | `70551309d170d43696fff527cd5b5893421996ba` | real_pr | pr:3101 |
| `realpr-werkzeug-3116` | python | MIT | https://github.com/pallets/werkzeug.git | `a792bc2d1ebd52abfd285db847ef0fd42a911df9` | real_pr | pr:3116 |
| `realpr-bitflags-483` | rust | MIT OR Apache-2.0 | https://github.com/bitflags/bitflags.git | `4ed9ffa949970239cd2d87c775e9fdcf9c438fb5` | real_pr | pr:483 |

**Product certified N (real_pr only): 20**

## Notes

- Product surface: `datasets/deepagent_v1` (source_track=real_pr only).
- Hybrid archive (historical only): `datasets/deepagent_v1_hybrid_archive/`.
- Fixtures (non-product): `datasets/harbor_v1`, `datasets/v1`.
- Agent trees clone public git @ base SHA (no motor COPY hybrid_bind).
- Docker oracle dual truth: sol=1 / null=0 (never `oracle_mode=fake`).


## Multilang additive cert (m14-fix-multilang-suite-reporters)

- Added: ['realpr-qs-487', 'realpr-qs-488']
- Languages: {'python': 17, 'javascript': 2}
- HarborDocker dual-truth additive keeps; python product packs preserved (no wipe)


## Multilang third language (m14-fix-third-language-keep)

- Added: [`realpr-bitflags-483`] (rust bitflags/bitflags PR#483)
- Languages after additive: {'javascript': 2, 'python': 17, 'rust': 1}
- HarborDocker dual-truth: sol=1 null=0; rust base image rust:1.88-bookworm + cargo fetch bake
- Existing python=17 javascript=2 preserved (no wipe)
- Dual-run real node IDs: pass, tests/compile-pass/bitflags_flag_name.rs
