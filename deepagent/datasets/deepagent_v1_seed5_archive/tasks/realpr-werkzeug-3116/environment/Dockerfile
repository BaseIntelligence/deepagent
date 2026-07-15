# DeepAgent / Harbor agent environment image.
# Runtime network policy: allow_internet=false
# Deps are baked at build; agent/verifier runs must not require internet.
# Image/container prefixes used by factory: sdf-/deepagent-/harbor-sdf-
# Concurrency ceiling (envbuild/Pier jobs): ≤16 (hard band 16–24).
# Language recipe: python (base python:3.12-slim); held-out tests only via tests/test.patch (verifier); agent image must not bake durable test.patch.
FROM python:3.12-slim

# --- runtime offline contract (VAL-ENVR-002 / VAL-RCLN-004) ---
LABEL harbor.allow_internet="false"
LABEL swe_factory.base_commit="a792bc2d1ebd52abfd285db847ef0fd42a911df9"
LABEL swe_factory.language="python"
LABEL swe_factory.source_track="real_pr"
ENV HARBOR_ALLOW_INTERNET=false
ENV SWE_FACTORY_ALLOW_INTERNET=false
ENV BASE_COMMIT=a792bc2d1ebd52abfd285db847ef0fd42a911df9

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Live clone @ pinned base SHA (VAL-RCLN-001) then history-scrub.
# Product real_pr path never COPYs synthetic motor fixtures as the agent tree
# (VAL-RCLN-002). HEAD must equal BASE_SHA (VAL-RCLN-003).
ARG BASE_SHA=a792bc2d1ebd52abfd285db847ef0fd42a911df9
RUN set -eux; \
    git clone --filter=blob:none "https://github.com/pallets/werkzeug.git" /app; \
    cd /app; \
    git fetch --depth 1 origin "$BASE_SHA" || git fetch origin "$BASE_SHA"; \
    git checkout --force "$BASE_SHA"; \
    test "$(git rev-parse HEAD)" = "$BASE_SHA" \
      || test "$(git rev-parse HEAD)" = "$(git rev-parse "$BASE_SHA")"; \
    # Prove base_commit is a real commit object in this image.
    git rev-parse --verify "$BASE_SHA^{commit}"; \
    git remote remove origin || true; git reflog expire --expire=now --all || true; git gc --prune=now || true; \
    git rev-parse HEAD > /app/.harbor_base_commit

# Build-time dependency install only — runtime must run with allow_internet=false (no pip network install at agent start).
RUN pip install --no-cache-dir pytest freezegun hypothesis pretend anyio trio sniffio h11 pytest-asyncio pytest-mock ephemeral-port-reserve tomli_w tomli appdirs cloudpickle && \
    pip install --no-cache-dir jaraco.itertools jaraco.functools jaraco.context jaraco.collections jaraco.test more_itertools setuptools wheel inflect && \
    if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then pip install --no-cache-dir -e . || pip install --no-cache-dir . ; fi && \
    printf '%s\n' /app /app/src > /usr/local/lib/python3.12/site-packages/sdf_app.pth

RUN python - <<'EOS'
from pathlib import Path
import sys
cands = [Path(p) for p in sys.path if p and Path(p).is_dir() and "site-packages" not in p]
root = None
for c in cands:
    if (c / "os.py").exists() or (c / "encodings").exists():
        root = c
        break
if root is None:
    root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
root.mkdir(parents=True, exist_ok=True)
test_pkg = root / "test"
supp = test_pkg / "support"
supp.mkdir(parents=True, exist_ok=True)
(test_pkg / "__init__.py").write_text("# vendored minimal test package\n", encoding="utf-8")
(test_pkg / "__main__.py").write_text("", encoding="utf-8")
(supp / "__init__.py").write_text(
    "from __future__ import annotations\n"
    "import contextlib, os, shutil, tempfile\n"
    "from pathlib import Path\n\n"
    "class FakePath:\n"
    "    def __init__(self, path):\n"
    "        self._path = os.fspath(path)\n"
    "    def __fspath__(self):\n"
    "        return self._path\n"
    "    def __str__(self):\n"
    "        return self._path\n"
    "    def __repr__(self):\n"
    "        return f'FakePath({self._path!r})'\n\n"
    "@contextlib.contextmanager\n"
    "def temp_dir(quiet=True, path=None, disable_gc=False):\n"
    "    d = tempfile.mkdtemp() if path is None else path\n"
    "    try:\n"
    "        yield d\n"
    "    finally:\n"
    "        if path is None:\n"
    "            shutil.rmtree(d, ignore_errors=True)\n\n"
    "def create_empty_file(filename):\n"
    "    Path(filename).write_bytes(b'')\n\n"
    "def captured_stdout():\n"
    "    return contextlib.nullcontext()\n\n"
    "def captured_stderr():\n"
    "    return contextlib.nullcontext()\n\n"
    "def captured_stdin():\n"
    "    return contextlib.nullcontext()\n\n"
    "def requires_resource(resource):\n"
    "    return (lambda fn: fn)\n\n"
    "def cpython_only(fn):\n"
    "    return fn\n\n"
    "class EnvironmentVarGuard:\n"
    "    def __init__(self):\n"
    "        self._environ = os.environ\n"
    "        self._changed = {}\n"
    "    def set(self, envvar, value):\n"
    "        self._changed.setdefault(envvar, self._environ.get(envvar))\n"
    "        self._environ[envvar] = value\n"
    "    def unset(self, envvar):\n"
    "        self._changed.setdefault(envvar, self._environ.get(envvar))\n"
    "        self._environ.pop(envvar, None)\n"
    "    def __enter__(self):\n"
    "        return self\n"
    "    def __exit__(self, *exc):\n"
    "        for k, v in self._changed.items():\n"
    "            if v is None:\n"
    "                self._environ.pop(k, None)\n"
    "            else:\n"
    "                self._environ[k] = v\n\n"
    "__all__ = [\n"
    "    'FakePath', 'temp_dir', 'create_empty_file', 'captured_stdout',\n"
    "    'captured_stderr', 'captured_stdin', 'requires_resource',\n"
    "    'cpython_only', 'EnvironmentVarGuard',\n"
    "]\n",
    encoding="utf-8",
)
(supp / "os_helper.py").write_text(
    "from __future__ import annotations\n"
    "from test.support import FakePath, temp_dir, create_empty_file\n"
    "__all__ = ['FakePath', 'temp_dir', 'create_empty_file']\n",
    encoding="utf-8",
)
print("vendored test.support at", supp)
EOS

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
