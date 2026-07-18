# Product hardness floors (M21 / M27 DeepSWE-median)

DeepAgent production / live-generate hardness packs are **fail-closed** on
structural floors and an anti-easy policy. This document is the operator-facing
summary of VAL-DMED-001/002 (DeepSWE-median band), VAL-DHARD-002/003/005, and
VAL-DINTR (intrinsic, M25).

## Floors (product + live generate) — M27 DeepSWE-median

Public DeepSWE sample (N≈48 packs): files p50≈6 / p25≈4, hunks p50≈14,
added p50≈640 / min≈438. Product defaults match that structural band:

| Floor | Default | Config |
|---|---|---|
| Multi-file product sources | **≥ 4** | `PRODUCT_MULTI_FILE_FLOOR` |
| Source hunks | **≥ 14** | `PRODUCT_SOURCE_HUNK_FLOOR` |
| Gold added lines (solution.patch plus-lines) | **≥ 400** | `PRODUCT_MIN_ADDED_LINES` / env `PRODUCT_MIN_ADDED_LINES` |
| Fail-to-pass nodes | **≥ 5** | `MIN_F2P_NODES` / `DEEPAGENT_MIN_F2P_NODES` env, or const `DEFAULT_MIN_F2P_NODES` |

Enforced destinations:

- `datasets/deepagent_v1` (product archive)
- `datasets/test_n10` (live generate wave)
- `datasets/prod_hard_keep` (historical M25 softer band; floors still enforced)
- `datasets/prod_hard_deepswe_med` (M27 current product root)
- any path when `--live-mine` honesty is on

**Thin packs** (e.g. qs-487: 2 files, ~21 added, F2P=3, ~11 hunks) are refused
by at least one floor. Soft offline fixtures may skip with
`engineering_opt_out=True` / `offline_only=True` only.

## Anti-easy promote policy

Hardness promote **drops** the following classes (complimentary to dual-truth):

| Class | Reason code(s) | Notes |
|---|---|---|
| Thin F2P (count < MIN_F2P) | `f2p_nodes_below_floor`, `thin_f2p_easy_class` | Structural floor |
| Multi-file / hunk / added shortfalls | `multi_file_floor_rejected`, `source_hunks_below_floor`, `gold_added_lines_below_floor` | M27 DeepSWE-median |
| High-confidence intrinsic EASY_REQUEST | `intrinsic_easy_request` | Prompt+gold only (M25/M27); never model dual-success alone |
| Prompt–verifier misalign | `prompt_version_only_vs_behavioral_f2p`, … | M21a alignment gate |
| Solve-all (frontier=1.0) | `solve_all_easy_policy_drop` | Scoreboard **label** only under M25; not sole product drop |

**M25 authority:** model dual-success (solve-all) does **not** auto-drop product
hardness. Use `score_request_patch_difficulty` / intrinsic + structural floors.

Agent **timeout-class** model failures remain harness OK (not automatic dataset
drop) when dual-truth, alignment, and floors hold.

## Intrinsic request+patch difficulty (M25 / M27)

Module: `swe_factory.pipeline.intrinsic_difficulty`.

| Class | Behavior |
|---|---|
| `EASY_REQUEST` (high conf) | `should_drop_hardness=True` when drop gate used (qs-class thin gold) |
| `HARD_REQUEST` | DeepSWE-median multi-file large gold; keep |
| `UNCERTAIN` | Default keep |

Model pass@k is never an input to the scorer.

## Curated production hardness

```bash
# M25/M26 historical soft band (audit only after M27)
.venv/bin/deepagent curate-hardness \
  --src datasets/prod_hard_keep \
  --scoreboard datasets/panel_prod_hard_m26_n5/scoreboard.json \
  --out datasets/prod_hard_keep --json

# M27 regenerate into new product root (live, separate feature)
.venv/bin/deepagent generate --target 10 --min-packs 5 --max-packs 15 \
  --out datasets/prod_hard_deepswe_med --live-mine \
  --materials datasets/live_materials_m22 --oracle docker --panel offline --pier scripted --json
```

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
  default (VAL-DHARD-005 / VAL-DMED-001).
- Fixture shortlists (`fixtures/real_pr_ship`) remain engineering-only and do
  not pad product N.

## Code entrypoints

| Module | Role |
|---|---|
| `swe_factory.pipeline.hardness_floors` | Floors (files/hunks/f2p/added), refuse, anti-easy summary |
| `swe_factory.pipeline.intrinsic_difficulty` | Prompt+gold EASY/HARD/UNCERTAIN (M25/M27) |
| `swe_factory.pipeline.curate_prod_hard` | Curate gates + drop_reasons |
| `swe_factory.pipeline.ship_real_pr` | Cert/export wiring; `ProductHardnessFloorsRejected` |
| `swe_factory.pipeline.gate_audit_product` | Dual-truth audit rows include M27 floors |
| `swe_factory.pipeline.prompt_alignment` | Alignment gate (companion) |
| `swe_factory.producers.hard_filter` | Mine-time files≥4 / hunks≥14 floors |
| `swe_factory.panel.band` | Solve-all / solve-none band labels |

## Related docs

- [architecture.md](architecture.md) — certified keep gates
- Root `README.md` — CLI generate / honesty floors
- Mission `AGENTS.md` — M25 intrinsic + M27 DeepSWE-median policy
