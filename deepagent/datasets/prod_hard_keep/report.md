# DeepAgent production hardness ship report (M21c)

- Generated (UTC): `2026-07-17T13:04:24.853766+00:00`
- Source corpus: `datasets/prod_hard_keep`
- Curated product path: `datasets/prod_hard_keep`
- Source track (product): **`real_pr` only**
- Certified hardness packs N: **10** (floor ≥5)
- Docker oracle: dual-truth sol=1 / null=0 retained on every keep
- Prompt–verifier alignment: required on every keep
- Hardness floors: F2P≥3, multi-file, source hunks≥10
- Status: `OK` — curated production hardness panel (no fixture pad)

## Drop reasons

| pack_id | reason_code | panel | detail |
|---|---|---|---|

## Certified hardness keeps (real_pr)

- `realpr-attrs-1323` reason=`keep_hard_panel_band` f2p=4 hunks=14 sol=1 null=0 panel=split/keep frontier=0.5
- `realpr-attrs-1457` reason=`keep_legit_hard_solve_none` f2p=14 hunks=21 sol=1 null=0 panel=solve-none/keep frontier=0.0
- `realpr-httpx-3672` reason=`keep_legit_hard_solve_none` f2p=5 hunks=18 sol=1 null=0 panel=solve-none/keep frontier=0.0
- `realpr-itemadapter-101` reason=`keep_hard_panel_band` f2p=43 hunks=15 sol=1 null=0 panel=split/keep frontier=0.5
- `realpr-packaging-1120` reason=`keep_legit_hard_solve_none` f2p=9 hunks=24 sol=1 null=0 panel=solve-none/keep frontier=0.0
- `realpr-qs-487` reason=`keep_hard_panel_band` f2p=3 hunks=11 sol=1 null=0 panel=split/keep frontier=0.5
- `realpr-qs-488` reason=`keep_legit_hard_solve_none` f2p=7 hunks=12 sol=1 null=0 panel=solve-none/keep frontier=0.0
- `realpr-werkzeug-2979` reason=`keep_despite_model_solve_all` f2p=9 hunks=11 sol=1 null=0 panel=solve-all/drop frontier=1.0
- `realpr-werkzeug-3006` reason=`keep_despite_model_solve_all` f2p=6 hunks=14 sol=1 null=0 panel=solve-all/drop frontier=1.0
- `realpr-werkzeug-3101` reason=`keep_despite_model_solve_all` f2p=4 hunks=29 sol=1 null=0 panel=solve-all/drop frontier=1.0

## Gates (no relaxation)

- dual-truth HarborDocker sol=1 / null=0
- prompt–verifier alignment fail-closed
- hardness floors F2P≥3, multi-file≥2, hunks≥10
- anti-easy: thin F2P floors + high-confidence intrinsic EASY_REQUEST
- model dual success is not a hardness gate (M25)
- no fixture pad; hybrid never product N
- re-upload HF BaseIntelligence/deepagent revision `test`

## Assertions

- VAL-DHARD-004 curated production N to HF test
- VAL-DHARD-002 / VAL-DHARD-003 floors + anti-easy
- VAL-DINTR-001 / VAL-DINTR-002 / VAL-DINTR-003 / VAL-DINTR-005 intrinsic policy (restore solve-all-only drops)
