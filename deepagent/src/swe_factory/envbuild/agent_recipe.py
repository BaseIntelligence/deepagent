"""Agent image Dockerfile / bake recipes for DeepAgent real-SHA envbuild.

Runtime contract (VAL-ENVR-002 / VAL-RCLN-004): Harbor ``allow_internet=false``.
- Dependencies are baked at *build* time (``RUN pip/npm/go`` or COPY of locked cache).
- The runtime image documents ``allow_internet=false`` via LABEL + ENV markers.
- No runtime network install is required to start the workspace / green baseline.

Real-PR product path (VAL-RCLN-001..003):
- Agent Dockerfile materializes the public repo at the pinned 40-char base SHA
  via ``git clone`` + ``checkout`` (or equivalent bake of that worktree).
- Product real_pr recipes **must not** ``COPY`` synthetic motor trees
  (orderlib / kvstore / ts_registry fixtures) as the primary agent workspace
  while claiming a real ``repository_url``.
- Base commit must be a real git object after clone (HEAD match + marker).

Also documents porcelain + hooks-off + history scrub (VAL-ENVR-004/005):
- ``git config core.hooksPath /dev/null``
- clean porcelain at start (checkout only base tree)
- drop remotes / prune reflogs so future commits are not agent-visible leeway
- never COPY ``solution/`` into the agent context
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

LangCode = Literal["python", "go", "typescript", "javascript", "rust"]

# SCALE languages claimable on product / dual-run (VAL-MLANG-001).
SUPPORTED_RECIPE_LANGUAGES: tuple[str, ...] = (
    "python",
    "go",
    "javascript",
    "typescript",
    "rust",
)

# Marker strings that packages, tests, and docs search for.
ALLOW_INTERNET_FALSE = "allow_internet=false"
RUNTIME_OFFLINE_LABEL = 'LABEL harbor.allow_internet="false"'
RUNTIME_OFFLINE_ENV = "ENV HARBOR_ALLOW_INTERNET=false"
HOOKS_OFF_LINE = "git config core.hooksPath /dev/null"
BASE_COMMIT_MARKER = ".harbor_base_commit"
HISTORY_SCRUB_HINT = (
    "git remote remove origin || true; "
    "git reflog expire --expire=now --all || true; "
    "git gc --prune=now || true"
)

# Held-out isolation (VAL-LX-008): agent image must not bake durable tests
# via COPY of test.patch; agent sees them only after verifier applies test.patch.
HELD_OUT_TEST_PATCH_NAME = "test.patch"
HELD_OUT_ISOLATION_HINT = (
    "held-out tests only via tests/test.patch (verifier); "
    "agent image must not bake durable test.patch"
)

# ----- VAL-RCLN motor-fixture markers (product real_pr refuse) -----
MOTOR_COPY_MARKERS: tuple[str, ...] = (
    "orderlib/",
    "orderlib\\",
    "kvstore/",
    "ts_registry",
    "python_orders",
    "go_kvstore",
    "harbor_motors",
    "fixtures/harbor_motors",
    "fixtures/tiny_green",
    "fixtures/tiny_offline",
)

_COPY_REPO_RE = re.compile(r"^\s*COPY\s+repo/\s+", re.MULTILINE | re.IGNORECASE)
_GIT_CLONE_RE = re.compile(r"\bgit\s+clone\b", re.IGNORECASE)
_CHECKOUT_SHA_RE = re.compile(
    r"(git\s+checkout\b.*\$BASE_SHA|git\s+checkout\b.*BASE_SHA|"
    r"git\s+checkout\s+--force|rev-parse\s+HEAD)",
    re.IGNORECASE,
)
_HTTP_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)

# Label / marker proving base pin in product images
BASE_COMMIT_LABEL_PREFIX = "LABEL swe_factory.base_commit="
SOURCE_TRACK_LABEL_PREFIX = "LABEL swe_factory.source_track="


class RealPrDockerfileError(ValueError):
    """Product real_pr agent Dockerfile violates clone@SHA / no-motor gates."""


@dataclass(frozen=True, slots=True)
class AgentRecipeContract:
    """Documented contract surface for agent images @ base SHA."""

    allow_internet: bool = False
    hooks_path: str = "/dev/null"
    workspace_dir: str = "/app"
    base_commit_marker: str = BASE_COMMIT_MARKER
    scrub_future_history: bool = True
    exclude_solution_tree: bool = True
    concurrency_ceiling: int = 16

    def as_dict(self) -> dict[str, object]:
        return {
            "allow_internet": self.allow_internet,
            "hooks_path": self.hooks_path,
            "workspace_dir": self.workspace_dir,
            "base_commit_marker": self.base_commit_marker,
            "scrub_future_history": self.scrub_future_history,
            "exclude_solution_tree": self.exclude_solution_tree,
            "concurrency_ceiling": self.concurrency_ceiling,
            # Stable document string for scanners / VAL-ENVR-002.
            "runtime_network_policy": ALLOW_INTERNET_FALSE,
        }


def default_agent_contract() -> AgentRecipeContract:
    return AgentRecipeContract()


def is_public_git_https(url: str) -> bool:
    """True when *url* is a public http(s) git remote (not file:// / fixtures)."""
    cleaned = (url or "").strip()
    if not cleaned:
        return False
    lower = cleaned.lower()
    if lower.startswith("file://"):
        return False
    if "fixture" in lower or "harbor_motor" in lower or "localhost" in lower:
        return False
    if "example.com" in lower:
        return False
    return bool(_HTTP_URL_RE.match(cleaned))


def is_full_base_sha(value: str) -> bool:
    return bool(_FULL_SHA_RE.match((value or "").strip()))


def looks_motor_fixture_copy(dockerfile_text: str) -> bool:
    """Heuristic: Dockerfile materializes agent tree via local motor ``COPY repo/``."""
    text = dockerfile_text or ""
    if not _COPY_REPO_RE.search(text):
        return False
    # COPY alone is fixture-path when there is no git clone authority.
    if _GIT_CLONE_RE.search(text):
        # Hybrid hybrid-bind smell: both clone URL metadata and motor COPY.
        motorish = any(m in text for m in MOTOR_COPY_MARKERS)
        return motorish or ("fixture" in text.lower() and "offline" in text.lower())
    return True


def dockerfile_has_clone_at_sha(dockerfile_text: str) -> bool:
    """True when recipe clones then checks out a pinned BASE_SHA / base commit."""
    text = dockerfile_text or ""
    if not _GIT_CLONE_RE.search(text):
        return False
    return bool(_CHECKOUT_SHA_RE.search(text) or "BASE_SHA" in text)


def assert_real_pr_agent_dockerfile(
    dockerfile_text: str,
    *,
    repository_url: str = "",
    base_commit: str = "",
    source_track: str = "real_pr",
    allow_fixture_only: bool = False,
) -> None:
    """Fail closed when a product real_pr agent Dockerfile is motor-only hybrid.

    VAL-RCLN-001: clone/pin authority at base SHA.
    VAL-RCLN-002: no synthetic motor COPY as primary tree for product.
    VAL-RCLN-003: base_commit is a full 40-char SHA (object proof is separate).
    VAL-RCLN-004: runtime offline documented.

    Offline unit fixtures may set ``allow_fixture_only=True`` (archived motors,
    tiny_green, etc.) so they are free of product rules.
    """
    track = (source_track or "").strip().lower()
    if allow_fixture_only or track not in {"", "real_pr"}:
        return

    text = dockerfile_text or ""
    if not text.strip():
        raise RealPrDockerfileError("empty agent Dockerfile for real_pr product path")

    url = (repository_url or "").strip()
    # If URL is claimed public/real, Dockerfile must not be motor COPY-only.
    if is_public_git_https(url) or "github.com" in url.lower() or track == "real_pr":
        if looks_motor_fixture_copy(text) and not dockerfile_has_clone_at_sha(text):
            raise RealPrDockerfileError(
                "real_pr product agent Dockerfile must not COPY synthetic motor "
                "fixtures (orderlib/kvstore/ts_registry or COPY repo/ without git "
                f"clone@SHA) while claiming repository_url={url!r}"
            )
        # Even with concurrent clone text, product refuses motor fixture COPY.
        if (
            _COPY_REPO_RE.search(text)
            and is_public_git_https(url)
            and (
                any(m in text for m in MOTOR_COPY_MARKERS)
                or "Offline fixture" in text
                or "offline fixture" in text.lower()
            )
        ):
            raise RealPrDockerfileError(
                "real_pr product path refuses motor/hybrid COPY repo layout "
                f"for repository_url={url!r}"
            )
        if not dockerfile_has_clone_at_sha(text):
            raise RealPrDockerfileError(
                "real_pr product agent Dockerfile must materialize git clone @ "
                f"base_commit_hash (clone+checkout); missing clone@SHA authority "
                f"for repository_url={url!r}"
            )

    pin = (base_commit or "").strip()
    if pin and not is_full_base_sha(pin):
        raise RealPrDockerfileError(
            f"real_pr base_commit must be a full 40-char git SHA; got {base_commit!r}"
        )

    if ALLOW_INTERNET_FALSE not in text and "HARBOR_ALLOW_INTERNET=false" not in text:
        raise RealPrDockerfileError(
            "real_pr agent Dockerfile must document runtime allow_internet=false"
        )


def render_real_pr_agent_dockerfile(
    *,
    repository_url: str,
    base_commit: str,
    language: str = "python",
    base_image: str | None = None,
    install_commands: list[str] | None = None,
    workspace_dir: str = "/app",
    image_prefix_note: str = "sdf-/deepagent-/harbor-sdf-",
) -> str:
    """Render a **product** real_pr agent Dockerfile: clone@SHA only (no motor COPY).

    Guarantees VAL-RCLN-001..004 markers. Always refuses local ``COPY repo/`` because
    the claimed authority is the public git remote + base_commit_hash.
    """
    url = (repository_url or "").strip()
    pin = (base_commit or "").strip()
    if not is_public_git_https(url):
        raise RealPrDockerfileError(
            f"real_pr clone Dockerfile requires public HTTPS repository_url; got {repository_url!r}"
        )
    if not is_full_base_sha(pin):
        raise RealPrDockerfileError(
            f"real_pr clone Dockerfile requires full 40-char base_commit; got {base_commit!r}"
        )
    df = render_agent_dockerfile(
        base_commit=pin,
        language=language,
        base_image=base_image,
        install_commands=install_commands,
        workspace_dir=workspace_dir,
        repo_url=url,
        copy_context=False,  # never motor/fixture COPY for product real_pr
        image_prefix_note=image_prefix_note,
        source_track="real_pr",
        force_clone=True,
    )
    # Stamp explicit real_pr product markers for scanners.
    if SOURCE_TRACK_LABEL_PREFIX not in df:
        df = df.replace(
            RUNTIME_OFFLINE_LABEL,
            f'{RUNTIME_OFFLINE_LABEL}\n{SOURCE_TRACK_LABEL_PREFIX}"real_pr"',
            1,
        )
    assert_real_pr_agent_dockerfile(
        df,
        repository_url=url,
        base_commit=pin,
        source_track="real_pr",
    )
    return df


def normalize_recipe_language(language: str) -> str:
    """Canonical recipe language token (python|go|javascript|typescript|rust)."""
    lang = (language or "python").strip().lower()
    aliases = {
        "py": "python",
        "golang": "go",
        "ts": "typescript",
        "js": "javascript",
        "node": "javascript",
        "rs": "rust",
    }
    return aliases.get(lang, lang)


def base_image_for_language(language: str) -> str:
    """Map SCALE language → non-Python-fallthrough base image (VAL-MLANG-001).

    Rust must use a rustc/cargo bookworm-class image, never ``python:3.12-slim``.
    """
    lang = normalize_recipe_language(language)
    if lang == "go":
        return "golang:1.22-bookworm"
    if lang in {"typescript", "javascript"}:
        return "node:20-bookworm"
    if lang == "rust":
        # 1.78 cannot parse modern crates.io editions (edition2024 manifests),
        # so product dual-run/HarborDocker for real crates needs a current stable.
        return "rust:1.88-bookworm"
    # Slim keeps layers small. CPython meat package ``test.support`` is vendored
    # via TEST_SUPPORT_VENDOR_RUN (python:*-slim does not ship it).
    return "python:3.12-slim"


# Back-compat private alias used by render_agent_dockerfile / fixtures.
def _base_image_for(language: str) -> str:
    return base_image_for_language(language)


def default_install_commands(language: str) -> list[str]:
    """Build-time install commands only (runtime stays offline)."""
    lang = normalize_recipe_language(language)
    if lang == "python":
        return ["pip install --no-cache-dir pytest"]
    if lang == "go":
        return ["true"]  # module graph is local for fixtures / clone@SHA
    if lang in {"typescript", "javascript"}:
        return ["if [ -f package.json ]; then npm install --no-audit --no-fund; fi"]
    if lang == "rust":
        # Cargo will resolve workspace crates offline only for already-vendored trees;
        # default real-PR path expects deps baked or cargo fetch at build.
        return ["if [ -f Cargo.toml ]; then cargo fetch || true; fi"]
    return ["true"]


def default_baseline_test_command(language: str) -> str:
    """Language suite baseline for envbuild green gate / dual-run smoke."""
    lang = normalize_recipe_language(language)
    return {
        "python": "python -m pytest -q",
        "go": "go test ./...",
        "typescript": "npm test",
        "javascript": "npm test",
        "rust": "cargo test -- --nocapture",
    }.get(lang, "python -m pytest -q")


def agent_dockerfile_bakes_held_out_tests(dockerfile_text: str) -> bool:
    """True when agent recipe incorrectly embeds durable held-out ``test.patch``.

    VAL-LX-008: held-out tests must apply only via verifier ``tests/test.patch``,
    never via ``COPY … test.patch`` into the agent environment image body.
    """
    text = dockerfile_text or ""
    # Explicit bad patterns: COPY test.patch or COPY …/tests/test.patch into agent.
    if re.search(
        r"^\s*COPY\s+(?:.+\s+)?(?:.?/?tests?/)?test\.patch\b",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"^\s*COPY\s+.+test\.patch\s+",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    )


def agent_recipe_isolates_held_out_tests(dockerfile_text: str) -> bool:
    """True when recipe documents held-out isolation and does not bake test.patch."""
    text = dockerfile_text or ""
    if agent_dockerfile_bakes_held_out_tests(text):
        return False
    # Product recipes fail closed when test.patch/solution present in agent workdir.
    return "test.patch" in text and ("solution" in text or "held-out" in text.lower())


# Minimal CPython ``test`` package for slim agent images (zipp/jaraco neo-tests).
# Official python:*-slim strips stdlib ``test.support``; jaraco.test imports it.
TEST_SUPPORT_VENDOR_RUN = r"""
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
"""


def _install_block(language: str, install_commands: list[str] | None) -> str:
    """Render install RUN lines. Build-time only (offline at runtime)."""
    cmds = list(install_commands or [])
    if not cmds:
        cmds = default_install_commands(language)
    body = " && \\\n    ".join(cmds)
    lang = normalize_recipe_language(language)
    net_hint = {
        "python": "pip",
        "go": "go",
        "typescript": "npm",
        "javascript": "npm",
        "rust": "cargo",
    }.get(lang, "network")
    return (
        "# Build-time dependency install only — runtime must run with "
        f"{ALLOW_INTERNET_FALSE} (no {net_hint} network install at agent start).\n"
        f"RUN {body}"
    )


def render_agent_dockerfile(
    *,
    base_commit: str,
    language: str = "python",
    base_image: str | None = None,
    install_commands: list[str] | None = None,
    workspace_dir: str = "/app",
    repo_url: str = "",
    copy_context: bool = True,
    image_prefix_note: str = "sdf-/deepagent-/harbor-sdf-",
    source_track: str = "",
    force_clone: bool = False,
) -> str:
    """Render a certified-style agent Dockerfile @ *base_commit*.

    Documents:
    - ``allow_internet=false`` runtime contract (LABEL + ENV)
    - hooks off after checkout
    - history scrub (no remote / prune)
    - base SHA marker at ``.harbor_base_commit``
    - no ``solution/`` in agent context (comment + check)

    **Mutual exclusion (VAL-RCLN-001/002):** when *repo_url* is a public HTTPS
    remote (or *force_clone* is True), the recipe uses ``git clone`` @ SHA and
    **never** emits ``COPY repo/`` motor fixture layout — even if *copy_context*
    was left True by a caller. Local fixture motors may still use COPY when no
    public remote is claimed.
    """
    pin = (base_commit or "").strip()
    url = (repo_url or "").strip()
    lang = normalize_recipe_language(language)
    image = base_image or base_image_for_language(lang)
    install = _install_block(lang, install_commands)
    contract = default_agent_contract()
    # Python slim path only: vendor minimal test.support for jaraco-style suites.
    # Non-Python images do not ship CPython and must not emit the vendor block.
    vendor_block = TEST_SUPPORT_VENDOR_RUN if lang == "python" else ""

    # Product real_pr / public URL path: clone@SHA is the sole tree authority.
    clone_authority = force_clone or is_public_git_https(url)
    if clone_authority and not url:
        raise RealPrDockerfileError(
            "clone@SHA Dockerfile requires repository_url when force_clone=True"
        )

    # Refuse hybrid: never COPY motor fixture while cloning a claimed real URL.
    use_copy = bool(copy_context) and not clone_authority
    copy_block = ""
    if use_copy:
        copy_block = (
            "# Local/fixture path: COPY pre-pinned tree only (never solution/ or gold).\n"
            "# Build context must exclude solution/, held-out test.patch, and future tips.\n"
            "# Not valid for real_pr product packs that claim a public repository_url.\n"
            f"COPY repo/ {workspace_dir.rstrip('/')}/\n"
        )

    clone_block = ""
    if clone_authority:
        clone_block = f"""\
# Live clone @ pinned base SHA (VAL-RCLN-001) then history-scrub.
# Product real_pr path never COPYs synthetic motor fixtures as the agent tree
# (VAL-RCLN-002). HEAD must equal BASE_SHA (VAL-RCLN-003).
ARG BASE_SHA={pin}
RUN set -eux; \\
    git clone --filter=blob:none "{url}" {workspace_dir}; \\
    cd {workspace_dir}; \\
    git fetch --depth 1 origin "$BASE_SHA" || git fetch origin "$BASE_SHA"; \\
    git checkout --force "$BASE_SHA"; \\
    test "$(git rev-parse HEAD)" = "$BASE_SHA" \\
      || test "$(git rev-parse HEAD)" = "$(git rev-parse "$BASE_SHA")"; \\
    # Prove base_commit is a real commit object in this image.
    git rev-parse --verify "$BASE_SHA^{{commit}}"; \\
    {HISTORY_SCRUB_HINT}; \\
    git rev-parse HEAD > {workspace_dir.rstrip("/")}/{BASE_COMMIT_MARKER}
"""
    else:
        clone_block = f"""\
# Offline/local bake: tree already at base (COPY above). Init/pin if needed.
# Fixture/unit path only — not product real_pr authority.
RUN set -eux; \\
    cd {workspace_dir}; \\
    if [ ! -d .git ]; then \\
      git init; \\
      git config user.email "envbuild@local"; \\
      git config user.name "envbuild"; \\
      git add -A; \\
      git commit -q -m "base {pin}" || true; \\
    fi; \\
    # Soft pin marker — real SHA object is written by host builder when available.
    printf '%s\\n' "{pin}" > {workspace_dir.rstrip("/")}/{BASE_COMMIT_MARKER}; \\
    {HISTORY_SCRUB_HINT}
"""

    track_label = ""
    track = (source_track or "").strip().lower()
    if track:
        track_label = f'{SOURCE_TRACK_LABEL_PREFIX}"{track}"\n'

    # Editable installs / cargo generate create dirty untracked files (*.egg-info,
    # target/, node_modules residue). Reset worktree so porcelain check is about
    # source pin honesty, not build leftovers. Keep .harbor_base_commit marker.
    clean_install_dirt = f"""\
# Drop install dirt so porcelain check reflects source pin, not build leftovers.
RUN cd {workspace_dir} \\
 && find . -name '*.egg-info' -type d -prune -exec rm -rf {{}} + 2>/dev/null || true \\
 && find . -name '*.egg-info' -type f -delete 2>/dev/null || true \\
 && rm -rf .eggs dist build .pytest_cache .mypy_cache .ruff_cache __pycache__ \\
      target/debug target/release node_modules/.cache 2>/dev/null || true \\
 && git checkout --force -- . 2>/dev/null || true \\
 && git clean -fd -e {BASE_COMMIT_MARKER} 2>/dev/null || true
"""

    return f"""\
# DeepAgent / Harbor agent environment image.
# Runtime network policy: {ALLOW_INTERNET_FALSE}
# Deps are baked at build; agent/verifier runs must not require internet.
# Image/container prefixes used by factory: {image_prefix_note}
# Concurrency ceiling (envbuild/Pier jobs): ≤{contract.concurrency_ceiling} (hard band 16–24).
# Language recipe: {lang} (base {image}); {HELD_OUT_ISOLATION_HINT}.
FROM {image}

# --- runtime offline contract (VAL-ENVR-002 / VAL-RCLN-004) ---
{RUNTIME_OFFLINE_LABEL}
LABEL swe_factory.base_commit="{pin}"
LABEL swe_factory.language="{lang}"
{track_label}{RUNTIME_OFFLINE_ENV}
ENV SWE_FACTORY_ALLOW_INTERNET=false
ENV BASE_COMMIT={pin}

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR {workspace_dir}

{copy_block}{clone_block}
{install}
{vendor_block}
{clean_install_dirt}
# --- porcelain + hooks (VAL-ENVR-004) ---
# Ignore factory pin marker file (.harbor_base_commit) which is intentional
# and untracked after clone; all other dirty paths fail the gate.
RUN cd {workspace_dir} \\
 && git config --global --add safe.directory {workspace_dir} || true \\
 && {HOOKS_OFF_LINE} || true \\
 && git status --porcelain \\
      | grep -v '{BASE_COMMIT_MARKER}' \\
      | tee /tmp/porcelain.txt \\
 && test ! -s /tmp/porcelain.txt \\
 && test ! -d solution \\
 && test ! -f solution.patch \\
 && test ! -f test.patch \\
 && test ! -f tests/test.patch

# Agent must not see held-out gold or held-out tests as durable files
# (VAL-ENVR-005 / VAL-LX-008). Fail closed if present.
# Held-out suite deltas apply only via verifier tests/test.patch, never agent bake.
RUN cd {workspace_dir} \\
 && if [ -d solution ] || [ -f solution/solve.sh ] || [ -f gold.patch ] \\
      || [ -f test.patch ] || [ -f tests/test.patch ]; then \\
      echo "FATAL: solution/gold/held-out test.patch leaked into agent context" >&2; exit 42; \\
    fi

CMD ["/bin/bash"]
"""


def render_task_toml_env_snippet(*, allow_internet: bool = False) -> str:
    """Fragment suitable for Harbor ``task.toml`` environment section."""
    flag = "true" if allow_internet else "false"
    return f"allow_internet = {flag}"


__all__ = [
    "ALLOW_INTERNET_FALSE",
    "BASE_COMMIT_MARKER",
    "HELD_OUT_ISOLATION_HINT",
    "HELD_OUT_TEST_PATCH_NAME",
    "HISTORY_SCRUB_HINT",
    "HOOKS_OFF_LINE",
    "MOTOR_COPY_MARKERS",
    "RUNTIME_OFFLINE_ENV",
    "RUNTIME_OFFLINE_LABEL",
    "SUPPORTED_RECIPE_LANGUAGES",
    "AgentRecipeContract",
    "LangCode",
    "RealPrDockerfileError",
    "agent_dockerfile_bakes_held_out_tests",
    "agent_recipe_isolates_held_out_tests",
    "assert_real_pr_agent_dockerfile",
    "base_image_for_language",
    "default_agent_contract",
    "default_baseline_test_command",
    "default_install_commands",
    "dockerfile_has_clone_at_sha",
    "is_full_base_sha",
    "is_public_git_https",
    "looks_motor_fixture_copy",
    "normalize_recipe_language",
    "render_agent_dockerfile",
    "render_real_pr_agent_dockerfile",
    "render_task_toml_env_snippet",
    "TEST_SUPPORT_VENDOR_RUN",
]
