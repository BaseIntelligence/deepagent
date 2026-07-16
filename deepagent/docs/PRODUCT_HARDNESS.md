# Product hardness floors (M21)

DeepAgent production / live-generate hardness packs are **fail-closed** on
structural floors and an anti-easy policy. This document is the operator-facing
summary of VAL-DHARD-002, VAL-DHARD-003, and VAL-DHARD-005 (mission validation
contract).

## Floors (product + live generate)

| Floor | Default | Config |
|---|---|---|
| Source hunks | **≥ 10** | `PRODUCT_SOURCE_HUNK_FLOOR` (fixed product definition) |
| Multi-file product sources | **≥ 2** | `PRODUCT_MULTI_FILE_FLOOR` |
| Fail-to-pass nodes | **≥ 3** | `MIN_F2P_NODES` / `DEEPAGENT_MIN_F2P_NODES` env, or const `DEFAULT_MIN_F2P_NODES` |

Enforced destinations:

- `datasets/deepagent_v1` (product archive)
- `datasets/test_n10` (live generate wave)
- `datasets/prod_hard_keep` (curated hardness panel)
- any path when `--live-mine` honesty is on

**Thin F2P=1 packs are refused** on these dests by default. Example easy-class
fingerprints include packs whose held-out suite only contributes a single F2P
node.

## Anti-easy promote policy

Hardness promote **drops** the following classes (complimentary to dual-truth):

| Class | Reason code(s) | Notes |
|---|---|---|
| Thin F2P (count < MIN_F2P, incl. F2P≈1) | `f2p_nodes_below_floor`, `thin_f2p_easy_class` | Structural floor |
| Solve-all (frontier pass@k = 1.0) | `solve_all_easy_policy_drop` | Panel band + curate; optional panel aggregate on cert |
| Prompt–verifier misalign | `prompt_version_only_vs_behavioral_f2p`, … | M21a alignment gate |
| Multi-file / hunk shortfalls | `multi_file_floor_rejected`, `source_hunks_below_floor` | Existing hard floors |

Solve-all class is **dropped from hardness promote**. Curated production waves
(m21c) also document `drop_reasons` for solve-all / misalign packs (for example
charset-normalizer-715 and rich-4070).

## Curated production hardness panel (m21c / VAL-DHARD-004)

Curate from a dual-truth `datasets/test_n10` wave into `datasets/prod_hard_keep`:

```bash
.venv/bin/python - <<'PY'
from swe_factory.pipeline.curate_prod_hard import materialize_prod_hard_keep
r = materialize_prod_hard_keep(
    "datasets/test_n10",
    "datasets/prod_hard_keep",
    panel_report="datasets/panel_deepagent_bench10_n5/report.json",
)
print(r.pack_count, r.keep_ids, list(r.drop_reasons))
PY

# Re-upload HF revision test only (never main without approval)
.venv/bin/deepagent upload --src datasets/prod_hard_keep \
  --repo-id BaseIntelligence/deepagent --revision test --json
```

Policy residual:

| Class | Action | Example |
|---|---|---|
| Prompt–verifier misalign | DROP | more-itertools-1136 |
| Solve-all / thin F2P easy | DROP | charset-normalizer-715, rich-4070; also thin panel-band packs (F2P=1) |
| Hard panel keep-band + floors + align | KEEP | itemadapter-101, attrs-1323, httpx-3672 |
| Legit hard solve-none (dual-truth ok) | KEEP | attrs-1457, packaging-1120 |

If residual keep N < 5 after gates: fail-closed (`ProdHardCurationError`) and
**re-mine with new floors** — never pad fixtures. Corpus files:
`pack_manifest.json` (`drop_reasons`), `drop_reasons.json`, `PROVENANCE.md`,
`report.md`, dual-truth pack trees under `tasks/`.

Agent **timeout-class** model failures remain harness OK (not automatic dataset
drop) when dual-truth, alignment, and floors hold.

## Engineering opt-out (fixtures only)

Offline unit fixtures may pass with:

```python
from swe_factory.pipeline.hardness_floors import refuse_product_hardness_floors

refuse_product_hardness_floors(
    f2p_node_ids=["only_one"],
    source_files=["a.py", "b.py"],
    source_hunk_count=12,
    dest="tmp/offline_sandbox",
    offline_only=True,          # or engineering_opt_out=True
)
```

Rules:

- `engineering_opt_out=True` / `offline_only=True` **never** default for product
  or live_generate dests.
- Product cert path (`ship_real_pr` / `deepagent generate`) enforces floors by
  default (VAL-DHARD-005).
- Fixture shortlists (`fixtures/real_pr_ship`) remain engineering-only and do
  not pad product N.

## Code entrypoints

| Module | Role |
|---|---|
| `swe_factory.pipeline.hardness_floors` | Floors, refuse, anti-easy summary |
| `swe_factory.pipeline.curate_prod_hard` | M21c curate test_n10 → prod_hard_keep + drop_reasons |
| `swe_factory.pipeline.ship_real_pr` | Cert/export wiring; `ProductHardnessFloorsRejected` |
| `swe_factory.pipeline.gate_audit_product` | Dual-truth audit rows include F2P floor |
| `swe_factory.pipeline.prompt_alignment` | Alignment gate (companion) |
| `swe_factory.producers.hard_filter` | Mine-time ≥10 hunk / multi-file floors |
| `swe_factory.panel.band` | Solve-all / solve-none band drops |

## Related docs

- [architecture.md](architecture.md) — certified keep gates
- Root `README.md` — CLI generate / honesty floors
- Mission `AGENTS.md` — M21 production hardness policy
