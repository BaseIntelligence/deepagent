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

resolve_base_ref() {
  # 1) Guard file written by Dockerfile after committing base (optional).
  if [ -f /app/.harbor_base_commit ]; then
    local file_ref
    file_ref=$(tr -d '[:space:]' < /app/.harbor_base_commit 2>/dev/null || true)
    if [ -n "$file_ref" ] && git cat-file -e "${file_ref}^{commit}" 2>/dev/null; then
      printf '%s\n' "$file_ref"
      return 0
    fi
  fi
  # 2) Explicit env override (CLI / Pier can inject).
  if [ -n "${BASE_COMMIT:-}" ] && git cat-file -e "${BASE_COMMIT}^{commit}" 2>/dev/null; then
    printf '%s\n' "$BASE_COMMIT"
    return 0
  fi
  # 3) Configured SHA from pack ship (only if real object present).
  local configured="5f0449fff1d9fb236d297cd0d3650b42d2d93b8a"
  if [ -n "$configured" ] && git cat-file -e "${configured}^{commit}" 2>/dev/null; then
    printf '%s\n' "$configured"
    return 0
  fi
  # 4) Root commit of the workspace history (motor Dockerfile single commit).
  local root
  root=$(git rev-list --max-parents=0 HEAD 2>/dev/null | head -1 || true)
  if [ -n "$root" ]; then
    printf '%s\n' "$root"
    return 0
  fi
  # Last resort: empty tree-ish symbolic — still produce a patch file.
  printf 'HEAD\n'
}

BASE_REF=$(resolve_base_ref)
# Empty file on total git failure so verifier still sees a submission path.
: > /logs/artifacts/model.patch
if [ -n "$BASE_REF" ] && [ "$BASE_REF" != "HEAD" ]; then
  git diff --binary "$BASE_REF" HEAD > /logs/artifacts/model.patch 2>/dev/null || true
fi
echo "[pre_artifacts] base=$BASE_REF captured $(wc -c < /logs/artifacts/model.patch) bytes"
