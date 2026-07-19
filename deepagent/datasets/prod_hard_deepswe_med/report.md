# DeepAgent ship report — prod_hard_deepswe_med (DeepSWE-median)

- Generated (UTC): `2026-07-19T03:05:34.293291+00:00`
- Product path: `datasets/prod_hard_deepswe_med`
- Source track (product): **`real_pr` only** (hybrid not certified product)
- Certified packs N: **5** (wave min ≥5, target=10, cap 15)
- Docker oracle mode: `docker` (fake refused)
- Pier mode: `scripted`
- Panel mode: live scoreboard separate (`datasets/panel_prod_hard_deepswe_med_n5`, M27e)
- Provider calls this densify wave: `0` (oracle docker only)
- Status: `PASS` — certified_n=5 with root gate_audit accepted=5/5 (m27g rolled werkzeug-2608 from evidence/oracle_trim_2608c)

## Hybrid archive vs product honesty

Hybrid archive action=`already_archived` → `datasets/deepagent_v1_hybrid_archive` (archive_pack_count=113); ok=True; historical only, never product N.

- Hybrid archive is **historical only** and **not** folded into product N.
- No `hybrid_curated` rows appear in product PROVENANCE.
- hybrid_ids_in_certified_scan: `[]` (must be empty)

## Historical fixtures (non-product)

Historical / engineering only (not Real-PR product N): datasets/deepagent_v1_hybrid_archive (hybrid_curated motors), datasets/harbor_v1 (synth motors), datasets/v1 (boltons), fixtures/real_pr_ship (unit shortlist), datasets/prod_hard_keep (soft M25/M26 band). Product N uses live-mined real_pr materials under datasets/prod_hard_deepswe_med only.

## Language mix (honest real_pr)

| language | certified |
|---|---:|
| python | 5 |

### Under-supply notes

- go/typescript/javascript/rust=0 best-effort under-supply on this Real-PR densify wave (honest shortfall; not hybrid fill-in)
- GH secondary rate-limit blocked further densify toward target 10; no fixture pad

## Funnel (ordered Real-PR stages)

- materials / tree / real-pack ok: 5
- docker oracle cert: 5 (4 from densify5 gate + 1 later-path oracle_trim_2608c rolled into root)
- pier cert: 4 scripted densify5 + 2608 dual-truth docker (sol=1/null=0)
- certified keeps (real_pr): **5**

## Certified packs (real_pr)

- `realpr-itemadapter-101` track=real_pr lang=python files=['itemadapter/_imports.py', 'itemadapter/_json_schema.py', 'itemadapter/adapter.py', 'itemadapter/utils.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/scrapy/itemadapter.git sha=`f7860b6ec7c5b49f623ecd1f67e73877f08039b6` seed=pr:101
- `realpr-packaging-1120` track=real_pr lang=python files=['benchmarks/specifiers.py', 'src/packaging/_ranges.py', 'src/packaging/specifiers.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pypa/packaging.git sha=`ff14df979f865165553999d9d1a111feec6f4843` seed=pr:1120
- `realpr-werkzeug-2608` track=real_pr lang=python files=['examples/couchy/utils.py', 'examples/shortly/shortly.py', 'examples/shorty/utils.py', 'examples/simplewiki/utils.py', 'src/werkzeug/formparser.py', 'src/werkzeug/middleware/http_proxy.py', 'src/werkzeug/routing/converters.py', 'src/werkzeug/routing/map.py', 'src/werkzeug/routing/rules.py', 'src/werkzeug/sansio/request.py', 'src/werkzeug/sansio/utils.py', 'src/werkzeug/serving.py', 'src/werkzeug/urls.py', 'src/werkzeug/utils.py', 'src/werkzeug/wrappers/response.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`b2cdf1743790e7fe2799d95de966103607bb9b82` seed=pr:2608
- `realpr-werkzeug-2637` track=real_pr lang=python files=['src/werkzeug/_internal.py', 'src/werkzeug/http.py', 'src/werkzeug/sansio/http.py', 'src/werkzeug/sansio/response.py', 'src/werkzeug/test.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`4f8fddd2ec527d70331e35440fa14edb377dcff0` seed=pr:2637
- `realpr-werkzeug-3116` track=real_pr lang=python files=['src/werkzeug/datastructures/accept.py', 'src/werkzeug/datastructures/auth.py', 'src/werkzeug/datastructures/cache_control.py', 'src/werkzeug/datastructures/csp.py', 'src/werkzeug/datastructures/etag.py', 'src/werkzeug/datastructures/file_storage.py', 'src/werkzeug/datastructures/headers.py', 'src/werkzeug/datastructures/range.py', 'src/werkzeug/datastructures/structures.py', 'src/werkzeug/http.py', 'src/werkzeug/sansio/http.py', 'src/werkzeug/sansio/request.py', 'src/werkzeug/sansio/response.py', 'src/werkzeug/sansio/utils.py', 'src/werkzeug/serving.py', 'src/werkzeug/wrappers/response.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`a792bc2d1ebd52abfd285db847ef0fd42a911df9` seed=pr:3116

## Dual-truth / finalize invariant (m27g)

- PROVENANCE.md branded `prod_hard_deepswe_med`, certified N=5, includes werkzeug-2608 base `b2cdf174…`
- root `gate_audit_summary.json`: accepted_count=5, intended_count=5, accepted_ids all five
- `evidence/docker/` holds reward/oracle JSON for all five keeps
- pack_manifest / median_stats / tasks/ remain N=5 consistent
- secrets_scan: 0 hits

## Pier/Harbor structural load smoke

- ok: `True`
- tool: `harbor`
- sampled task_id: `realpr-packaging-1120`
- errors: []

## Gates (no relaxation)

- source_track=real_pr (hybrid refuse)
- Real HTTPS repository_url + full 40-char base_commit
- Agent Dockerfile: git clone@SHA (no motor COPY hybrid_bind)
- M27 floors: files≥4 OR hybrid(files≥3+added≥500+hunks≥14); hunks≥14; added≥400; F2P≥5
- Dual-run labels → tests/config.json F2P/P2P
- Docker oracle only: sol=1, null=0 (never fake)
- Product N counts real_pr only; hybrid archive excluded

## Cross-flow (mine → … → ship)

- Stages: archive_hybrid → mine/materials → export_real_harbor → dual_run_labels → docker_oracle → pier_cert → promote → (m27g gate_audit/PROVENANCE rollup)
- E2E drip: `datasets/prod_hard_deepswe_med/e2e_drip.jsonl`

## Ledger summary (project)


```json
{
  "cap_usd": "600.0",
  "remaining_usd": "600.0",
  "settled_call_count": 34856,
  "settled_exact_usd": "0",
  "total_commit_usd": "0",
  "under_cap": true
}
```
