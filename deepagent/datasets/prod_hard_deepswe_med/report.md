# DeepAgent v1 ship report (Real-PR product surface)

- Generated (UTC): `2026-07-19T01:08:52.881840+00:00`
- Product path: `datasets/prod_hard_deepswe_med`
- Source track (product): **`real_pr` only** (hybrid not certified product)
- Certified packs N: **4** (wave target ≥5, target=10, cap 15)
- Docker oracle mode: `docker` (fake refused)
- Pier mode: `scripted`
- Panel mode: `offline` (budget_stop=False)
- Provider calls this wave: `0`
- Project spend commit: `$0` (remaining `$600.0`, under_cap=True)
- Status: `FAIL` — under-yield real_pr certified=4 < min=5; live-mine materials=datasets/live_materials_m27c_densify5 (no fixture pad)

## Hybrid archive vs product honesty

Hybrid archive action=`already_archived` → `datasets/deepagent_v1_hybrid_archive` (archive_pack_count=113); ok=True; historical only, never product N. reason=Idempotent no-op: archive already contains hybrid pack evidence (N=113). Product not claimed as current certified real_pr corpus by this archive step.

- Hybrid archive is **historical only** and **not** folded into product N.
- No `hybrid_curated` rows appear in product PROVENANCE.
- hybrid_ids_in_certified_scan: `[]` (must be empty)

## Historical fixtures (non-product)

Historical / engineering only (not Real-PR product N): datasets/deepagent_v1_hybrid_archive (hybrid_curated motors), datasets/harbor_v1 (synth motors), datasets/v1 (boltons), fixtures/real_pr_ship (unit shortlist). Product N uses live-mined real_pr materials under datasets/deepagent_v1 only.

## Language mix (honest real_pr)

| language | certified |
|---|---:|
| python | 4 |

### Under-supply notes

- certified=4 < min=5: real_pr funnel did not promote enough merged-PR packs (honest under-yield; never pad with hybrid motors)
- go=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- typescript=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- javascript=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- rust=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)

## Funnel (ordered Real-PR stages)

- materials loaded: 4
- tree complete: 4
- real-pack ok: 4
- docker oracle cert: 4
- pier cert: 4
- certified keeps (real_pr): 4

## Certified packs (real_pr)

- `realpr-packaging-1120` track=real_pr lang=python files=['benchmarks/specifiers.py', 'src/packaging/_ranges.py', 'src/packaging/specifiers.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pypa/packaging.git sha=`ff14df979f865165553999d9d1a111feec6f4843` seed=pr:1120
- `realpr-itemadapter-101` track=real_pr lang=python files=['itemadapter/_imports.py', 'itemadapter/_json_schema.py', 'itemadapter/adapter.py', 'itemadapter/utils.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/scrapy/itemadapter.git sha=`f7860b6ec7c5b49f623ecd1f67e73877f08039b6` seed=pr:101
- `realpr-werkzeug-2637` track=real_pr lang=python files=['src/werkzeug/_internal.py', 'src/werkzeug/http.py', 'src/werkzeug/sansio/http.py', 'src/werkzeug/sansio/response.py', 'src/werkzeug/test.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`4f8fddd2ec527d70331e35440fa14edb377dcff0` seed=pr:2637
- `realpr-werkzeug-3116` track=real_pr lang=python files=['src/werkzeug/datastructures/accept.py', 'src/werkzeug/datastructures/auth.py', 'src/werkzeug/datastructures/cache_control.py', 'src/werkzeug/datastructures/csp.py', 'src/werkzeug/datastructures/etag.py', 'src/werkzeug/datastructures/file_storage.py', 'src/werkzeug/datastructures/headers.py', 'src/werkzeug/datastructures/range.py', 'src/werkzeug/datastructures/structures.py', 'src/werkzeug/http.py', 'src/werkzeug/sansio/http.py', 'src/werkzeug/sansio/request.py', 'src/werkzeug/sansio/response.py', 'src/werkzeug/sansio/utils.py', 'src/werkzeug/serving.py', 'src/werkzeug/wrappers/response.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`a792bc2d1ebd52abfd285db847ef0fd42a911df9` seed=pr:3116

## Pier/Harbor structural load smoke

- ok: `True`
- tool: `harbor`
- sampled task_id: `realpr-packaging-1120`
- errors: []

## Gates (no relaxation)

- source_track=real_pr (hybrid refuse)
- Real HTTPS repository_url + full 40-char base_commit
- Agent Dockerfile: git clone@SHA (no motor COPY hybrid_bind)
- Multi-file solution floor ≥2 product sources + held-out test.patch
- Dual-run labels → tests/config.json F2P/P2P
- Docker oracle only: sol=1, null=0 (never fake)
- Pier oracle evidence (prefer live; scripted only with explicit ship flag)
- Product N counts real_pr only; hybrid archive excluded
- Project OpenRouter spend ≤ $600 (exact ledger)

## Cross-flow (mine → … → ship)

- Stages: archive_hybrid → mine/materials → export_real_harbor → dual_run_labels → docker_oracle → pier_cert → panel → promote
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
