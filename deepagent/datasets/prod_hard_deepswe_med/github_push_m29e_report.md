# m29e — GitHub origin/main push evidence

**Feature:** `m29e-git-push-origin-main`  
**Fulfills:** VAL-PUB-006  
**UTC:** 2026-07-20T12:13:00Z

## Auth

- `gh` active account: **echobt**
- Token scopes confirmed: `repo`, `workflow` (and gist/read:org)
- Remote: `https://github.com/BaseIntelligence/deepagent.git`
- Force push: **no**

## Pre-push (SHAs only)

| Ref | SHA |
|---|---|
| `origin/main` before | `4f983cf41a32043326dc440f477f14454942dcb7` |
| local `HEAD` (intended tip) | `4a433b44d47785a06b809201eedde145d24389ce` |
| `rev-list --left-right --count origin/main...HEAD` before | `0 54` |

## Public docs tip chain (m29a–d on local main before push)

| Feature | Short | Full SHA | Subject |
|---|---|---|---|
| m29a | `fe77342` | `fe77342b0afd672f0f36f48a6aef3b526f7a2a9b` | docs: rewrite root README as DeepAgent product-first public surface |
| m29b | `aaf51e1` | `aaf51e1cf23ffb5a9022ff32164e47c5c23404db` | docs(deepagent): m29b README N=9 product + env proxy placeholders |
| m29c | `3b260af` | `3b260af9dd29d4bc71b4ba66b7d31eb5c1f81264` | docs(deepagent): m29c refresh M28 panel secrets_scan for DCOV-007 |
| m29d | `4a433b4` | `4a433b44d47785a06b809201eedde145d24389ce` | docs(deepagent): m29d HF full re-upload N=9 report + card |

## Push

```text
git push origin main
# 4f983cf..4a433b4  main -> main
```

## Post-push (SHAs only)

| Ref | SHA |
|---|---|
| `origin/main` after | `4a433b44d47785a06b809201eedde145d24389ce` |
| local `HEAD` after | `4a433b44d47785a06b809201eedde145d24389ce` |
| `rev-list --left-right --count origin/main...HEAD` after | `0 0` |

- m29a–d tips confirmed ancestors of `origin/main`: **yes**
- Intended published tip matches local public tip: **yes**
- Unpushed intentional WIP after main push: **none** (`0 0`)

## Working tree policy (left unstaged)

Did **not** stage or push:

- `.env` (gitignored)
- `.firecrawl/`
- `datasets/_m27*`, `datasets/_m28*_work*`, `datasets/_m28c_hf_stage`, `datasets/_m29d_hf_stage` and other work/stage junk
- dirty `tasks/*/tests/Dockerfile` noise
- fixtures `meta.json` noise
- unrelated modified archive/report paths

## Secrets

- `git log --all --full-history -- .env '**/.env'`: **empty** (no `.env` commits)
- No `.env` path added in `origin/main..HEAD` range before push
- Tokens never printed

## Note on this report commit

A follow-up commit may land this file on `origin/main` after the primary tip push above; primary published tip for m29a–d remains `4a433b44d47785a06b809201eedde145d24389ce` (or a later report-only tip that remains `0 0` with origin).
