"""Emitter for DeepSWE pre_artifacts.sh (committed work → model.patch).

The workspace base commit inside Harbor motor images is created at Docker
build time via ``git commit -m base``. That SHA is not the synthetic
placeholder stored in seed metadata (e.g. ``a1000…``). Diffing against a
non-ancestor placeholder yields an empty model.patch and reward 0.

``pre_artifacts.sh`` therefore resolves the base dynamically:

1. Prefer an explicit BASE_COMMIT env / optional baked marker file.
2. Prefer the configured base only when it is a valid ancestor object.
3. Otherwise fall back to the root commit of the current branch history
   (``git rev-list --max-parents=0 HEAD``), which matches motor Dockerfiles
   that create a single base commit.
"""

from __future__ import annotations

PRE_ARTIFACTS_TEMPLATE = """\
#!/bin/bash
# Capture the agent's committed work as the submission artifact: the diff
# between the starting commit and the agent's final HEAD.
#
# Prefer the configured base SHA when it is a real git object; otherwise use
# the repository root commit created by the environment Dockerfile.
set -uo pipefail
cd /app || exit 0
mkdir -p /logs/artifacts
git config --global --add safe.directory /app 2>/dev/null || true

resolve_base_ref() {{
  # 1) Guard file written by Dockerfile after committing base (optional).
  if [ -f /app/.harbor_base_commit ]; then
    local file_ref
    file_ref=$(tr -d '[:space:]' < /app/.harbor_base_commit 2>/dev/null || true)
    if [ -n "$file_ref" ] && git cat-file -e "${{file_ref}}^{{commit}}" 2>/dev/null; then
      printf '%s\\n' "$file_ref"
      return 0
    fi
  fi
  # 2) Explicit env override (CLI / Pier can inject).
  if [ -n "${{BASE_COMMIT:-}}" ] && git cat-file -e "${{BASE_COMMIT}}^{{commit}}" 2>/dev/null; then
    printf '%s\\n' "$BASE_COMMIT"
    return 0
  fi
  # 3) Configured SHA from pack ship (only if real object present).
  local configured="{base_commit}"
  if [ -n "$configured" ] && git cat-file -e "${{configured}}^{{commit}}" 2>/dev/null; then
    printf '%s\\n' "$configured"
    return 0
  fi
  # 4) Root commit of the workspace history (motor Dockerfile single commit).
  local root
  root=$(git rev-list --max-parents=0 HEAD 2>/dev/null | head -1 || true)
  if [ -n "$root" ]; then
    printf '%s\\n' "$root"
    return 0
  fi
  # Last resort: empty tree-ish symbolic — still produce a patch file.
  printf 'HEAD\\n'
}}

BASE_REF=$(resolve_base_ref)
# Empty file on total git failure so verifier still sees a submission path.
: > /logs/artifacts/model.patch
if [ -n "$BASE_REF" ] && [ "$BASE_REF" != "HEAD" ]; then
  git diff --binary "$BASE_REF" HEAD > /logs/artifacts/model.patch 2>/dev/null || true
fi
echo "[pre_artifacts] base=$BASE_REF captured $(wc -c < /logs/artifacts/model.patch) bytes"
"""


def render_pre_artifacts_sh(base_commit: str | None = None) -> str:
    """Render pre_artifacts.sh with optional preferred base commit metadata.

    ``base_commit`` is recorded as a preference when the object exists in the
    agent workspace; inventory hashes that never enter the image are ignored
    in favor of the dynamic root commit.
    """
    commit = (base_commit or "").strip()
    # Empty is allowed: the template falls back to root / env / marker.
    return PRE_ARTIFACTS_TEMPLATE.format(base_commit=commit)


__all__ = ["PRE_ARTIFACTS_TEMPLATE", "render_pre_artifacts_sh"]
