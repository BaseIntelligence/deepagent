# DeepAgent production hardness ship report (M21c)

- Generated (UTC): `2026-07-17T10:48:54.406609+00:00`
- Source corpus: `datasets/prod_hard_keep`
- Curated product path: `datasets/prod_hard_keep`
- Source track (product): **`real_pr` only**
- Certified hardness packs N: **7** (floor ≥5)
- Docker oracle: dual-truth sol=1 / null=0 retained on every keep
- Prompt–verifier alignment: required on every keep
- Hardness floors: F2P≥3, multi-file, source hunks≥10
- Status: `OK` — curated production hardness panel (no fixture pad)

## Drop reasons

| pack_id | reason_code | panel | detail |
|---|---|---|---|
| `realpr-werkzeug-2979` | `solve_all_easy_policy_drop` | - | EASY_SOLVE_ALL: all 2 panel model(s) pass@1=1.0 (moonshotai/kimi-k2.6=1.0, x-ai/grok-4.5=1.0); frontier=1.0; auto-drop from hardness without name hardcoding (VAL-DEASY-002) |
| `realpr-werkzeug-3006` | `solve_all_easy_policy_drop` | - | EASY_SOLVE_ALL: all 2 panel model(s) pass@1=1.0 (moonshotai/kimi-k2.6=1.0, x-ai/grok-4.5=1.0); frontier=1.0; auto-drop from hardness without name hardcoding (VAL-DEASY-002) |
| `realpr-werkzeug-3101` | `solve_all_easy_policy_drop` | - | EASY_SOLVE_ALL: all 2 panel model(s) pass@1=1.0 (moonshotai/kimi-k2.6=1.0, x-ai/grok-4.5=1.0); frontier=1.0; auto-drop from hardness without name hardcoding (VAL-DEASY-002) |

## Certified hardness keeps (real_pr)

- `realpr-attrs-1323` reason=`keep_hard_dual_truth` f2p=4 hunks=14 sol=1 null=0 panel=None/None frontier=None
- `realpr-attrs-1457` reason=`keep_hard_dual_truth` f2p=14 hunks=21 sol=1 null=0 panel=None/None frontier=None
- `realpr-httpx-3672` reason=`keep_hard_dual_truth` f2p=5 hunks=18 sol=1 null=0 panel=None/None frontier=None
- `realpr-itemadapter-101` reason=`keep_hard_dual_truth` f2p=43 hunks=15 sol=1 null=0 panel=None/None frontier=None
- `realpr-packaging-1120` reason=`keep_hard_dual_truth` f2p=9 hunks=24 sol=1 null=0 panel=None/None frontier=None
- `realpr-qs-487` reason=`keep_hard_dual_truth` f2p=3 hunks=11 sol=1 null=0 panel=None/None frontier=None
- `realpr-qs-488` reason=`keep_hard_dual_truth` f2p=7 hunks=12 sol=1 null=0 panel=None/None frontier=None

## Gates (no relaxation)

- dual-truth HarborDocker sol=1 / null=0
- prompt–verifier alignment fail-closed
- hardness floors F2P≥3, multi-file≥2, hunks≥10
- anti-easy: solve-all + thin F2P dropped
- no fixture pad; hybrid never product N
- re-upload HF BaseIntelligence/deepagent revision `test`

## Assertions

- VAL-DHARD-004 curated production N to HF test
- VAL-DHARD-002 / VAL-DHARD-003 floors + anti-easy
