# PROVENANCE — datasets/deepswe_v1 (Real-PR product)

Corpus of Docker-oracle-certified **real_pr** Harbor packs only.
Hybrid motors live under `datasets/deepswe_v1_hybrid_archive/` (historical; never counted here as product N).
Each row is one certified keep. Copyleft / unknown-license candidates
are fail-closed and never appear here.

| pack_id | language | license | upstream_url | base_sha | source_track | pr |
|---|---|---|---|---|---|---:|
| `realpr-attrs-1323` | python | MIT | https://github.com/python-attrs/attrs.git | `dbb25ce34787a30e2ffa65685f6c689a269c3521` | real_pr | pr:1323 |
| `realpr-rich-3486` | python | MIT | https://github.com/Textualize/rich.git | `d0de442c08df8793c5ff36d9ad322ca5c47fc38c` | real_pr | pr:3486 |
| `realpr-attrs-1457` | python | MIT | https://github.com/python-attrs/attrs.git | `af349876f4fb5df2b71a3f4878239b775bc89230` | real_pr | pr:1457 |
| `realpr-httpx-3672` | python | BSD-3-Clause | https://github.com/encode/httpx.git | `4acf5c2c37714cc63b5cf71b3e284fca83c90311` | real_pr | pr:3672 |
| `realpr-more-itertools-1136` | python | MIT | https://github.com/more-itertools/more-itertools.git | `e4d2a4a2a97246a73856754b2c4866d7f41d4875` | real_pr | pr:1136 |
| `realpr-more-itertools-943` | python | MIT | https://github.com/more-itertools/more-itertools.git | `f36c88fe03688fa442154ef14f429bcfa4c38525` | real_pr | pr:943 |
| `realpr-rich-4070` | python | MIT | https://github.com/Textualize/rich.git | `fc41075a3206d2a5fd846c6f41c4d2becab814fa` | real_pr | pr:4070 |
| `realpr-click-3645` | python | BSD-3-Clause | https://github.com/pallets/click.git | `679a7a0eccbdded7a6e85680bdaaf08003765e01` | real_pr | pr:3645 |
| `realpr-httpcore-882` | python | MIT | https://github.com/encode/httpcore.git | `c46802478cdd8a82ee8cb333420080fab1aed00b` | real_pr | pr:882 |
| `realpr-werkzeug-2995` | python | BSD-3-Clause | https://github.com/pallets/werkzeug.git | `1a1728ed88939ca68928dade168e1989be062c6f` | real_pr | pr:2995 |
| `realpr-werkzeug-2979` | python | BSD-3-Clause | https://github.com/pallets/werkzeug.git | `862cb193c2b13db860d886725fa4235173d0dfcd` | real_pr | pr:2979 |
| `realpr-werkzeug-3006` | python | BSD-3-Clause | https://github.com/pallets/werkzeug.git | `cb307c144e7b9092bf72b1a1dba5281e7c6ff838` | real_pr | pr:3006 |
| `realpr-werkzeug-3101` | python | BSD-3-Clause | https://github.com/pallets/werkzeug.git | `70551309d170d43696fff527cd5b5893421996ba` | real_pr | pr:3101 |
| `realpr-werkzeug-3116` | python | BSD-3-Clause | https://github.com/pallets/werkzeug.git | `a792bc2d1ebd52abfd285db847ef0fd42a911df9` | real_pr | pr:3116 |

**Product certified N (real_pr only): 14**

## Notes

- Product surface: `datasets/deepswe_v1` (source_track=real_pr only).
- Hybrid archive (historical only): `datasets/deepswe_v1_hybrid_archive/`.
- Fixtures (non-product): `datasets/harbor_v1`, `datasets/v1`.
- Agent trees clone public git @ base SHA (no motor COPY hybrid_bind).
- Docker oracle dual truth: sol=1 / null=0 (never `oracle_mode=fake`).
