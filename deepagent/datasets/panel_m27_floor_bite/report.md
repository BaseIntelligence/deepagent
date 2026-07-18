# M27b offline curate bite — thin prod_hard_keep under DeepSWE-median floors

**Feature:** `m27b-offline-curate-bite-thin-prod`  
**Assertion:** `VAL-DMED-003`  
**Generated:** `2026-07-18T17:42:30.297444+00:00`  
**ok:** `True`

## Goal

Prove new M27 product floors drop soft historical packs before live regen.
Historical `datasets/prod_hard_keep` pack trees stay on disk (audit-only).
No Hugging Face change in this step.

## Floors applied

| Floor | Default |
|---|---|
| source files | ≥ **4** |
| source hunks | ≥ **14** |
| gold added lines | ≥ **400** |
| F2P nodes | ≥ **5** |
| dual-truth + alignment | required |
| intrinsic high-conf EASY_REQUEST | drop |
| model dual-solve alone | **never** sole drop (M25) |

**Source:** `datasets/prod_hard_keep`  
**Scoreboard (annotate only):** `datasets/panel_prod_hard_m26_n5/scoreboard.json`  
**n_src / keep / drop:** **10** / **1** / **9**

- qs-487 dropped: **True**
- qs-487-class drops: `realpr-qs-487, realpr-qs-488`
- thin dual-solve drops: `realpr-qs-487, realpr-werkzeug-2979, realpr-werkzeug-3006, realpr-werkzeug-3101`
- historical trees preserved: **True**
- HF change: **False**
- materialize ran: **False**

## Drop reasons (each pack below floors / easy intrinsic)

### `realpr-attrs-1323`

- **reason_code:** `f2p_nodes_below_floor`
- **floor_primary:** `f2p_nodes_below_floor`
- **floor_reasons:** `f2p_nodes_below_floor, multi_file_floor_rejected, gold_added_lines_below_floor`
- **detail:** f2p_count=4 < MIN_F2P_NODES=5 (thin F2P refused on product hardness path); product source files 3 < multi-file floor 4; gold_added_lines=37 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=3 hunks=14 added=37 f2p=4
- **intrinsic:** class=`HARD_REQUEST` should_drop=`False` code=`intrinsic_hard_request`
- **dual_model_solve (M26 scoreboard):** `False` frontier=`0.5` per_model=`{'grok-4.5': 0.0, 'kimi-k2.6': 1.0}`
- **tree_preserved:** `True`

### `realpr-attrs-1457`

- **reason_code:** `gold_added_lines_below_floor`
- **floor_primary:** `gold_added_lines_below_floor`
- **floor_reasons:** `multi_file_floor_rejected, gold_added_lines_below_floor`
- **detail:** product source files 2 < multi-file floor 4; gold_added_lines=58 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=2 hunks=21 added=58 f2p=14
- **intrinsic:** class=`HARD_REQUEST` should_drop=`False` code=`intrinsic_hard_request`
- **dual_model_solve (M26 scoreboard):** `False` frontier=`0.0` per_model=`{'grok-4.5': 0.0, 'kimi-k2.6': 0.0}`
- **tree_preserved:** `True`

### `realpr-httpx-3672`

- **reason_code:** `gold_added_lines_below_floor`
- **floor_primary:** `gold_added_lines_below_floor`
- **floor_reasons:** `gold_added_lines_below_floor`
- **detail:** gold_added_lines=57 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=7 hunks=18 added=57 f2p=5
- **intrinsic:** class=`HARD_REQUEST` should_drop=`False` code=`intrinsic_hard_request`
- **dual_model_solve (M26 scoreboard):** `False` frontier=`0.0` per_model=`{'grok-4.5': 0.0, 'kimi-k2.6': 0.0}`
- **tree_preserved:** `True`

### `realpr-packaging-1120`

- **reason_code:** `multi_file_floor_rejected`
- **floor_primary:** `multi_file_floor_rejected`
- **floor_reasons:** `multi_file_floor_rejected`
- **detail:** product source files 3 < multi-file floor 4
- **stats:** files=3 hunks=24 added=882 f2p=9
- **intrinsic:** class=`HARD_REQUEST` should_drop=`False` code=`intrinsic_hard_request`
- **dual_model_solve (M26 scoreboard):** `False` frontier=`0.0` per_model=`{'grok-4.5': 0.0, 'kimi-k2.6': 0.0}`
- **tree_preserved:** `True`

### `realpr-qs-487`

- **reason_code:** `f2p_nodes_below_floor`
- **floor_primary:** `f2p_nodes_below_floor`
- **floor_reasons:** `f2p_nodes_below_floor, multi_file_floor_rejected, source_hunks_below_floor, gold_added_lines_below_floor`
- **detail:** f2p_count=3 < MIN_F2P_NODES=5 (thin F2P refused on product hardness path); product source files 2 < multi-file floor 4; source_hunk_count=11 < product floor 14; gold_added_lines=21 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=2 hunks=11 added=21 f2p=3
- **intrinsic:** class=`EASY_REQUEST` should_drop=`True` code=`intrinsic_easy_request`
- **dual_model_solve (M26 scoreboard):** `True` frontier=`1.0` per_model=`{'grok-4.5': 1.0, 'kimi-k2.6': 1.0}`
- **tree_preserved:** `True`

### `realpr-qs-488`

- **reason_code:** `gold_added_lines_below_floor`
- **floor_primary:** `gold_added_lines_below_floor`
- **floor_reasons:** `multi_file_floor_rejected, source_hunks_below_floor, gold_added_lines_below_floor`
- **detail:** product source files 2 < multi-file floor 4; source_hunk_count=12 < product floor 14; gold_added_lines=34 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=2 hunks=12 added=34 f2p=7
- **intrinsic:** class=`UNCERTAIN` should_drop=`False` code=`intrinsic_uncertain`
- **dual_model_solve (M26 scoreboard):** `False` frontier=`0.0` per_model=`{'grok-4.5': 0.0, 'kimi-k2.6': 0.0}`
- **tree_preserved:** `True`

### `realpr-werkzeug-2979`

- **reason_code:** `gold_added_lines_below_floor`
- **floor_primary:** `gold_added_lines_below_floor`
- **floor_reasons:** `multi_file_floor_rejected, source_hunks_below_floor, gold_added_lines_below_floor`
- **detail:** product source files 3 < multi-file floor 4; source_hunk_count=11 < product floor 14; gold_added_lines=77 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=3 hunks=11 added=77 f2p=9
- **intrinsic:** class=`HARD_REQUEST` should_drop=`False` code=`intrinsic_hard_request`
- **dual_model_solve (M26 scoreboard):** `True` frontier=`1.0` per_model=`{'grok-4.5': 1.0, 'kimi-k2.6': 1.0}`
- **tree_preserved:** `True`

### `realpr-werkzeug-3006`

- **reason_code:** `gold_added_lines_below_floor`
- **floor_primary:** `gold_added_lines_below_floor`
- **floor_reasons:** `multi_file_floor_rejected, gold_added_lines_below_floor`
- **detail:** product source files 2 < multi-file floor 4; gold_added_lines=142 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=2 hunks=14 added=142 f2p=6
- **intrinsic:** class=`HARD_REQUEST` should_drop=`False` code=`intrinsic_hard_request`
- **dual_model_solve (M26 scoreboard):** `True` frontier=`1.0` per_model=`{'grok-4.5': 1.0, 'kimi-k2.6': 1.0}`
- **tree_preserved:** `True`

### `realpr-werkzeug-3101`

- **reason_code:** `f2p_nodes_below_floor`
- **floor_primary:** `f2p_nodes_below_floor`
- **floor_reasons:** `f2p_nodes_below_floor, gold_added_lines_below_floor`
- **detail:** f2p_count=4 < MIN_F2P_NODES=5 (thin F2P refused on product hardness path); gold_added_lines=188 < PRODUCT_MIN_ADDED_LINES=400 (DeepSWE-median gold size floor)
- **stats:** files=4 hunks=29 added=188 f2p=4
- **intrinsic:** class=`HARD_REQUEST` should_drop=`False` code=`intrinsic_hard_request`
- **dual_model_solve (M26 scoreboard):** `True` frontier=`1.0` per_model=`{'grok-4.5': 1.0, 'kimi-k2.6': 1.0}`
- **tree_preserved:** `True`

## Keeps (still pass M27 under offline gates)

Note: residual keep under historical N=10 is expected to be thin; live regen into `prod_hard_deepswe_med` is m27c.

- `realpr-itemadapter-101`

## Policy notes

- M27 floors: files≥4, hunks≥14, added≥400, F2P≥5.
- M25: dual-model solve-all is never the sole drop reason.
- High-confidence intrinsic EASY_REQUEST may drop (qs-487 class).
- Offline bite only: no materialize clean/rmtree of prod_hard_keep.
- No HF upload/pull in this feature.

