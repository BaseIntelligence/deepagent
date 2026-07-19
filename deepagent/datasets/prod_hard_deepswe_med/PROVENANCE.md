# PROVENANCE — datasets/prod_hard_deepswe_med (DeepSWE-median product)

Corpus of Docker-oracle-certified **real_pr** Harbor packs for the M27 DeepSWE-median
hardness band. Hybrid motors live under `datasets/deepagent_v1_hybrid_archive/`
(historical; never counted here as product N). Soft historical band remains under
`datasets/prod_hard_keep` (audit only). Each row is one certified keep. Copyleft /
unknown-license candidates are fail-closed and never appear here.

| pack_id | language | license | upstream_url | base_sha | source_track | pr |
|---|---|---|---|---|---|---:|
| `realpr-itemadapter-101` | python | MIT | https://github.com/scrapy/itemadapter.git | `f7860b6ec7c5b49f623ecd1f67e73877f08039b6` | real_pr | pr:101 |
| `realpr-packaging-1120` | python | Apache-2.0 | https://github.com/pypa/packaging.git | `ff14df979f865165553999d9d1a111feec6f4843` | real_pr | pr:1120 |
| `realpr-werkzeug-2608` | python | MIT | https://github.com/pallets/werkzeug.git | `b2cdf1743790e7fe2799d95de966103607bb9b82` | real_pr | pr:2608 |
| `realpr-werkzeug-2637` | python | MIT | https://github.com/pallets/werkzeug.git | `4f8fddd2ec527d70331e35440fa14edb377dcff0` | real_pr | pr:2637 |
| `realpr-werkzeug-3116` | python | MIT | https://github.com/pallets/werkzeug.git | `a792bc2d1ebd52abfd285db847ef0fd42a911df9` | real_pr | pr:3116 |

**Product certified N (real_pr only): 5**

## Dual-truth audit

- Root ledger: `gate_audit.jsonl` / `gate_audit_summary.json` → `accepted_count=5` / `intended_count=5`.
- Docker evidence: `evidence/docker/<task_id>.{json,oracle_evidence.json,sol.reward.json,null.reward.json}` for all five keeps.
- `realpr-werkzeug-2608` late-path cert under `evidence/oracle_trim_2608c/` (sol=1/null=0 after ambient P2P fileurl trim) rolled into root gate_audit + `evidence/docker/`.
- Backend: `HarborDockerVerifier` only (never `oracle_mode=fake`).

## Structural floors (M27 median band)

- source files ≥ 4 **OR** hybrid (files ≥ 3 AND gold added ≥ 500 AND hunks ≥ 14)
- source hunks ≥ 14
- gold added lines ≥ 400
- F2P nodes ≥ 5
- live dual-run labels + alignment + intrinsic non-easy (M25: no drop solely from dual-model solve)

## Notes

- Product surface: `datasets/prod_hard_deepswe_med` (source_track=real_pr only).
- Soft historical band: `datasets/prod_hard_keep` (not current product N).
- Hybrid archive (historical only): `datasets/deepagent_v1_hybrid_archive/`.
- Fixtures (non-product): `fixtures/real_pr_ship`, `datasets/harbor_v1`, `datasets/v1`.
- Agent trees clone public git @ base SHA (no motor COPY hybrid_bind).
- Docker oracle dual truth: sol=1 / null=0 for every certified keep.
- HF mirror: `BaseIntelligence/deepagent` revision `test` (pack_manifest certified_n=5).
