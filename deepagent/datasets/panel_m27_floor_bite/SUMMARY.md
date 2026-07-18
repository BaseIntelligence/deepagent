# M27b offline floor bite — SUMMARY

- **ok:** `True`
- **assertion:** `VAL-DMED-003`
- **src:** `datasets/prod_hard_keep` (read-only; trees preserved)
- **n_src / keep / drop:** 10 / 1 / 9
- **qs-487 dropped:** `True`
- **thin dual-solve drops:** `realpr-qs-487, realpr-werkzeug-2979, realpr-werkzeug-3006, realpr-werkzeug-3101`
- **HF change:** `False`
- **materialize ran:** `False`

## Floors (DeepSWE-median M27)

```json
{
  "min_source_files": 4,
  "min_source_hunks": 14,
  "min_added_lines": 400,
  "min_f2p_nodes": 5,
  "band": "deepswe_median_m27"
}
```

## Drop ids

- `realpr-attrs-1323` — `f2p_nodes_below_floor` (floor_primary=`f2p_nodes_below_floor`, added=37, files=3, hunks=14, f2p=4, dual_solve=False)
- `realpr-attrs-1457` — `gold_added_lines_below_floor` (floor_primary=`gold_added_lines_below_floor`, added=58, files=2, hunks=21, f2p=14, dual_solve=False)
- `realpr-httpx-3672` — `gold_added_lines_below_floor` (floor_primary=`gold_added_lines_below_floor`, added=57, files=7, hunks=18, f2p=5, dual_solve=False)
- `realpr-packaging-1120` — `multi_file_floor_rejected` (floor_primary=`multi_file_floor_rejected`, added=882, files=3, hunks=24, f2p=9, dual_solve=False)
- `realpr-qs-487` — `f2p_nodes_below_floor` (floor_primary=`f2p_nodes_below_floor`, added=21, files=2, hunks=11, f2p=3, dual_solve=True)
- `realpr-qs-488` — `gold_added_lines_below_floor` (floor_primary=`gold_added_lines_below_floor`, added=34, files=2, hunks=12, f2p=7, dual_solve=False)
- `realpr-werkzeug-2979` — `gold_added_lines_below_floor` (floor_primary=`gold_added_lines_below_floor`, added=77, files=3, hunks=11, f2p=9, dual_solve=True)
- `realpr-werkzeug-3006` — `gold_added_lines_below_floor` (floor_primary=`gold_added_lines_below_floor`, added=142, files=2, hunks=14, f2p=6, dual_solve=True)
- `realpr-werkzeug-3101` — `f2p_nodes_below_floor` (floor_primary=`f2p_nodes_below_floor`, added=188, files=4, hunks=29, f2p=4, dual_solve=True)

## Keep ids (pass M27 floors + non-easy intrinsic)

- `realpr-itemadapter-101`

