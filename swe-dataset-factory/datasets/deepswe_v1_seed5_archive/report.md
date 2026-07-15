# DeepSWE v1 ship report (Real-PR product surface)

- Generated (UTC): `2026-07-15T09:47:32.667007+00:00`
- Product path: `/projects/swe-dataset-factory/datasets/deepswe_v1`
- Source track (product): **`real_pr` only** (hybrid not certified product)
- Certified packs N: **14** (wave target ≥15, target=20, cap 70)
- Docker oracle mode: `docker` (fake refused)
- Pier mode: `scripted`
- Panel mode: `offline` (budget_stop=False)
- Provider calls this wave: `0`
- Project spend commit: `$3.4603450` (remaining `$596.5396550`, under_cap=True)
- Status: `FAIL` — under-yield real_pr certified=14 < min=15; live-mine materials=/projects/swe-dataset-factory/datasets/live_materials (no fixture pad)

## Hybrid archive vs product honesty

Hybrid archive action=`already_archived` → `datasets/deepswe_v1_hybrid_archive` (archive_pack_count=113); ok=True; historical only, never product N. reason=Idempotent no-op: archive already contains hybrid pack evidence (N=113). Product not claimed as current certified real_pr corpus by this archive step. || Seed5 archive action=`already_archived` → `datasets/deepswe_v1_seed5_archive` (archive_pack_count=12); ok=True; historical real_pr seed only, never live N. reason=Idempotent no-op: seed5 archive already contains prior product evidence (N=12). Hybrid archive separate.

- Hybrid archive is **historical only** and **not** folded into product N.
- No `hybrid_curated` rows appear in product PROVENANCE.
- hybrid_ids_in_certified_scan: `[]` (must be empty)

## Historical fixtures (non-product)

Historical / engineering only (not Real-PR product N): datasets/deepswe_v1_hybrid_archive (hybrid_curated motors), datasets/harbor_v1 (synth motors), datasets/v1 (boltons), fixtures/real_pr_ship (unit shortlist). Product N uses live-mined real_pr materials under datasets/deepswe_v1 only.

## Language mix (honest real_pr)

| language | certified |
|---|---:|
| python | 14 |

### Under-supply notes

- go=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- typescript=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- javascript=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- rust=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)

## Funnel (ordered Real-PR stages)

- materials loaded: 105
- tree complete: 84
- real-pack ok: 21
- docker oracle cert: 14
- pier cert: 21
- certified keeps (real_pr): 14

## Certified packs (real_pr)

- `realpr-attrs-1323` track=real_pr lang=python files=['src/attr/_make.py', 'src/attr/_next_gen.py', 'src/attr/validators.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/python-attrs/attrs.git sha=`dbb25ce34787a30e2ffa65685f6c689a269c3521` seed=pr:1323
- `realpr-rich-3486` track=real_pr lang=python files=['rich/default_styles.py', 'rich/syntax.py', 'rich/traceback.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/Textualize/rich.git sha=`d0de442c08df8793c5ff36d9ad322ca5c47fc38c` seed=pr:3486
- `realpr-attrs-1457` track=real_pr lang=python files=['src/attr/_make.py', 'src/attr/_next_gen.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/python-attrs/attrs.git sha=`af349876f4fb5df2b71a3f4878239b775bc89230` seed=pr:1457
- `realpr-httpx-3672` track=real_pr lang=python files=['src/ahttpx/_parsers.py', 'src/ahttpx/_pool.py', 'src/ahttpx/_server.py', 'src/httpx/_network.py', 'src/httpx/_parsers.py', 'src/httpx/_pool.py', 'src/httpx/_server.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/encode/httpx.git sha=`4acf5c2c37714cc63b5cf71b3e284fca83c90311` seed=pr:3672
- `realpr-more-itertools-1136` track=real_pr lang=python files=['more_itertools/__init__.py', 'more_itertools/more.py', 'more_itertools/recipes.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/more-itertools/more-itertools.git sha=`e4d2a4a2a97246a73856754b2c4866d7f41d4875` seed=pr:1136
- `realpr-more-itertools-943` track=real_pr lang=python files=['docs/conf.py', 'more_itertools/more.py', 'more_itertools/recipes.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/more-itertools/more-itertools.git sha=`f36c88fe03688fa442154ef14f429bcfa4c38525` seed=pr:943
- `realpr-rich-4070` track=real_pr lang=python files=['rich/_emoji_replace.py', 'rich/console.py', 'rich/emoji.py', 'rich/logging.py', 'rich/protocol.py', 'rich/repr.py', 'rich/segment.py', 'rich/syntax.py', 'rich/theme.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/Textualize/rich.git sha=`fc41075a3206d2a5fd846c6f41c4d2becab814fa` seed=pr:4070
- `realpr-click-3645` track=real_pr lang=python files=['src/click/core.py', 'src/click/decorators.py', 'src/click/shell_completion.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/click.git sha=`679a7a0eccbdded7a6e85680bdaaf08003765e01` seed=pr:3645
- `realpr-httpcore-882` track=real_pr lang=python files=['httpcore/_async/http11.py', 'httpcore/_sync/http11.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/encode/httpcore.git sha=`c46802478cdd8a82ee8cb333420080fab1aed00b` seed=pr:882
- `realpr-werkzeug-2995` track=real_pr lang=python files=['src/werkzeug/datastructures/headers.py', 'src/werkzeug/datastructures/structures.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`1a1728ed88939ca68928dade168e1989be062c6f` seed=pr:2995
- `realpr-werkzeug-2979` track=real_pr lang=python files=['src/werkzeug/datastructures/headers.py', 'src/werkzeug/datastructures/mixins.py', 'src/werkzeug/datastructures/structures.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`862cb193c2b13db860d886725fa4235173d0dfcd` seed=pr:2979
- `realpr-werkzeug-3006` track=real_pr lang=python files=['src/werkzeug/routing/map.py', 'src/werkzeug/routing/rules.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`cb307c144e7b9092bf72b1a1dba5281e7c6ff838` seed=pr:3006
- `realpr-werkzeug-3101` track=real_pr lang=python files=['src/werkzeug/datastructures/file_storage.py', 'src/werkzeug/formparser.py', 'src/werkzeug/test.py', 'src/werkzeug/wrappers/request.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`70551309d170d43696fff527cd5b5893421996ba` seed=pr:3101
- `realpr-werkzeug-3116` track=real_pr lang=python files=['src/werkzeug/datastructures/accept.py', 'src/werkzeug/datastructures/auth.py', 'src/werkzeug/datastructures/cache_control.py', 'src/werkzeug/datastructures/csp.py', 'src/werkzeug/datastructures/etag.py', 'src/werkzeug/datastructures/file_storage.py', 'src/werkzeug/datastructures/headers.py', 'src/werkzeug/datastructures/range.py', 'src/werkzeug/datastructures/structures.py', 'src/werkzeug/http.py', 'src/werkzeug/sansio/http.py', 'src/werkzeug/sansio/request.py', 'src/werkzeug/sansio/response.py', 'src/werkzeug/sansio/utils.py', 'src/werkzeug/serving.py', 'src/werkzeug/wrappers/response.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pallets/werkzeug.git sha=`a792bc2d1ebd52abfd285db847ef0fd42a911df9` seed=pr:3116

## Pier/Harbor structural load smoke

- ok: `True`
- tool: `harbor`
- sampled task_id: `realpr-attrs-1323`
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
- E2E drip: `/projects/swe-dataset-factory/datasets/deepswe_v1/e2e_drip.jsonl`

## Ledger summary (project)

```json
{
  "cap_usd": "600.0",
  "remaining_usd": "596.5396550",
  "settled_call_count": 32141,
  "settled_exact_usd": "3.4603450",
  "total_commit_usd": "3.4603450",
  "under_cap": true
}
```
