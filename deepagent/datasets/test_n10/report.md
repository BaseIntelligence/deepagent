# DeepAgent v1 ship report (Real-PR product surface)

- Generated (UTC): `2026-07-15T22:05:31.615772+00:00`
- Product path: `datasets/test_n10`
- Source track (product): **`real_pr` only** (hybrid not certified product)
- Certified packs N: **10** (wave target ≥5, target=10, cap 10)
- Docker oracle mode: `docker` (fake refused)
- Pier mode: `scripted`
- Panel mode: `offline` (budget_stop=False)
- Provider calls this wave: `0`
- Project spend commit: `$0` (remaining `$600.0`, under_cap=True)
- Status: `OK` — shipped 10 real_pr Docker-oracle + Pier-certified packs (wave ≥5; hybrid archived, not product); materials=datasets/live_materials

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
| python | 10 |

### Under-supply notes

- go=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- typescript=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- javascript=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)
- rust=0 best-effort under-supply on Real-PR wave: no certified merged-PR keep announced for this language (honest shortfall; not hybrid fill-in)

## Funnel (ordered Real-PR stages)

- materials loaded: 17
- tree complete: 17
- real-pack ok: 13
- docker oracle cert: 10
- pier cert: 13
- certified keeps (real_pr): 10

## Certified packs (real_pr)

- `realpr-packaging-1120` track=real_pr lang=python files=['benchmarks/specifiers.py', 'src/packaging/_ranges.py', 'src/packaging/specifiers.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/pypa/packaging.git sha=`ff14df979f865165553999d9d1a111feec6f4843` seed=pr:1120
- `realpr-charset-normalizer-715` track=real_pr lang=python files=['_mypyc_hook/backend.py', 'noxfile.py', 'src/charset_normalizer/api.py', 'src/charset_normalizer/cd.py', 'src/charset_normalizer/cli/__main__.py', 'src/charset_normalizer/constant.py', 'src/charset_normalizer/legacy.py', 'src/charset_normalizer/md.py', 'src/charset_normalizer/models.py', 'src/charset_normalizer/utils.py', 'src/charset_normalizer/version.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/jawah/charset_normalizer.git sha=`7411396ebd495e1abc28f5682975b5c662b2ff35` seed=pr:715
- `realpr-itemadapter-101` track=real_pr lang=python files=['itemadapter/_imports.py', 'itemadapter/_json_schema.py', 'itemadapter/adapter.py', 'itemadapter/utils.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/scrapy/itemadapter.git sha=`f7860b6ec7c5b49f623ecd1f67e73877f08039b6` seed=pr:101
- `realpr-attrs-1323` track=real_pr lang=python files=['src/attr/_make.py', 'src/attr/_next_gen.py', 'src/attr/validators.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/python-attrs/attrs.git sha=`dbb25ce34787a30e2ffa65685f6c689a269c3521` seed=pr:1323
- `realpr-rich-3486` track=real_pr lang=python files=['rich/default_styles.py', 'rich/syntax.py', 'rich/traceback.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/Textualize/rich.git sha=`d0de442c08df8793c5ff36d9ad322ca5c47fc38c` seed=pr:3486
- `realpr-attrs-1457` track=real_pr lang=python files=['src/attr/_make.py', 'src/attr/_next_gen.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/python-attrs/attrs.git sha=`af349876f4fb5df2b71a3f4878239b775bc89230` seed=pr:1457
- `realpr-httpx-3672` track=real_pr lang=python files=['src/ahttpx/_parsers.py', 'src/ahttpx/_pool.py', 'src/ahttpx/_server.py', 'src/httpx/_network.py', 'src/httpx/_parsers.py', 'src/httpx/_pool.py', 'src/httpx/_server.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/encode/httpx.git sha=`4acf5c2c37714cc63b5cf71b3e284fca83c90311` seed=pr:3672
- `realpr-more-itertools-1136` track=real_pr lang=python files=['more_itertools/__init__.py', 'more_itertools/more.py', 'more_itertools/recipes.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/more-itertools/more-itertools.git sha=`e4d2a4a2a97246a73856754b2c4866d7f41d4875` seed=pr:1136
- `realpr-more-itertools-943` track=real_pr lang=python files=['docs/conf.py', 'more_itertools/more.py', 'more_itertools/recipes.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/more-itertools/more-itertools.git sha=`f36c88fe03688fa442154ef14f429bcfa4c38525` seed=pr:943
- `realpr-rich-4070` track=real_pr lang=python files=['rich/_emoji_replace.py', 'rich/console.py', 'rich/emoji.py', 'rich/logging.py', 'rich/protocol.py', 'rich/repr.py', 'rich/segment.py', 'rich/syntax.py', 'rich/theme.py'] sol=1 null=0 pier_oracle=1 upstream=https://github.com/Textualize/rich.git sha=`fc41075a3206d2a5fd846c6f41c4d2becab814fa` seed=pr:4070

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
- E2E drip: `datasets/test_n10/e2e_drip.jsonl`

## Ledger summary (project)

```json
{
  "cap_usd": "600.0",
  "remaining_usd": "600.0",
  "settled_call_count": 18160,
  "settled_exact_usd": "0",
  "total_commit_usd": "0",
  "under_cap": true
}
```
