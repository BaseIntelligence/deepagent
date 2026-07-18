# PROVENANCE — datasets/deepagent_v1 (Real-PR product)

Corpus of Docker-oracle-certified **real_pr** Harbor packs only.
Hybrid motors live under `datasets/deepagent_v1_hybrid_archive/` (historical; never counted here as product N).
Each row is one certified keep. Copyleft / unknown-license candidates
are fail-closed and never appear here.

| pack_id | language | license | upstream_url | base_sha | source_track | pr |
|---|---|---|---|---|---|---:|
| `realpr-itemadapter-101` | python | MIT | https://github.com/scrapy/itemadapter.git | `f7860b6ec7c5b49f623ecd1f67e73877f08039b6` | real_pr | pr:101 |
| `realpr-werkzeug-3116` | python | MIT | https://github.com/pallets/werkzeug.git | `a792bc2d1ebd52abfd285db847ef0fd42a911df9` | real_pr | pr:3116 |

**Product certified N (real_pr only): 2**

## Notes

- Product surface: `datasets/deepagent_v1` (source_track=real_pr only).
- Hybrid archive (historical only): `datasets/deepagent_v1_hybrid_archive/`.
- Fixtures (non-product): `datasets/harbor_v1`, `datasets/v1`.
- Agent trees clone public git @ base SHA (no motor COPY hybrid_bind).
- Docker oracle dual truth: sol=1 / null=0 (never `oracle_mode=fake`).
