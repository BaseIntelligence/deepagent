# DeepSWE / Harbor agent environment image.
# Runtime network policy: allow_internet=false
# Deps are baked at build; agent/verifier runs must not require internet.
# Image/container prefixes used by factory: sdf-/deepswe-/harbor-sdf-
# Concurrency ceiling (envbuild/Pier jobs): ≤16 (hard band 16–24).
# Language recipe: javascript (base node:20-bookworm); held-out tests only via tests/test.patch (verifier); agent image must not bake durable test.patch.
FROM node:20-bookworm

# --- runtime offline contract (VAL-ENVR-002 / VAL-RCLN-004) ---
LABEL harbor.allow_internet="false"
LABEL swe_factory.base_commit="04f422fe91985103d2fdca0280ee362ecf5e43f2"
LABEL swe_factory.language="javascript"
LABEL swe_factory.source_track="real_pr"
ENV HARBOR_ALLOW_INTERNET=false
ENV SWE_FACTORY_ALLOW_INTERNET=false
ENV BASE_COMMIT=04f422fe91985103d2fdca0280ee362ecf5e43f2

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Live clone @ pinned base SHA (VAL-RCLN-001) then history-scrub.
# Product real_pr path never COPYs synthetic motor fixtures as the agent tree
# (VAL-RCLN-002). HEAD must equal BASE_SHA (VAL-RCLN-003).
ARG BASE_SHA=04f422fe91985103d2fdca0280ee362ecf5e43f2
RUN set -eux; \
    git clone --filter=blob:none "https://github.com/ljharb/qs.git" /app; \
    cd /app; \
    git fetch --depth 1 origin "$BASE_SHA" || git fetch origin "$BASE_SHA"; \
    git checkout --force "$BASE_SHA"; \
    test "$(git rev-parse HEAD)" = "$BASE_SHA" \
      || test "$(git rev-parse HEAD)" = "$(git rev-parse "$BASE_SHA")"; \
    # Prove base_commit is a real commit object in this image.
    git rev-parse --verify "$BASE_SHA^{commit}"; \
    git remote remove origin || true; git reflog expire --expire=now --all || true; git gc --prune=now || true; \
    git rev-parse HEAD > /app/.harbor_base_commit

# Build-time dependency install only — runtime must run with allow_internet=false (no npm network install at agent start).
RUN if [ -f package.json ]; then npm install --no-audit --no-fund; fi

# Drop install dirt so porcelain check reflects source pin, not build leftovers.
RUN cd /app \
 && find . -name '*.egg-info' -type d -prune -exec rm -rf {} + 2>/dev/null || true \
 && find . -name '*.egg-info' -type f -delete 2>/dev/null || true \
 && rm -rf .eggs dist build .pytest_cache .mypy_cache .ruff_cache __pycache__ \
      target/debug target/release node_modules/.cache 2>/dev/null || true \
 && git checkout --force -- . 2>/dev/null || true \
 && git clean -fd -e .harbor_base_commit 2>/dev/null || true

# --- porcelain + hooks (VAL-ENVR-004) ---
# Ignore factory pin marker file (.harbor_base_commit) which is intentional
# and untracked after clone; all other dirty paths fail the gate.
RUN cd /app \
 && git config --global --add safe.directory /app || true \
 && git config core.hooksPath /dev/null || true \
 && git status --porcelain \
      | grep -v '.harbor_base_commit' \
      | tee /tmp/porcelain.txt \
 && test ! -s /tmp/porcelain.txt \
 && test ! -d solution \
 && test ! -f solution.patch \
 && test ! -f test.patch \
 && test ! -f tests/test.patch

# Agent must not see held-out gold or held-out tests as durable files
# (VAL-ENVR-005 / VAL-LX-008). Fail closed if present.
# Held-out suite deltas apply only via verifier tests/test.patch, never agent bake.
RUN cd /app \
 && if [ -d solution ] || [ -f solution/solve.sh ] || [ -f gold.patch ] \
      || [ -f test.patch ] || [ -f tests/test.patch ]; then \
      echo "FATAL: solution/gold/held-out test.patch leaked into agent context" >&2; exit 42; \
    fi

CMD ["/bin/bash"]
