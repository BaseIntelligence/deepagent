"""Allowlisted modular seed repos for DeepSWE multi-lang mining + hybrid ship.

Seeds prefer multi-module Python / TypeScript / JavaScript / Go / Rust layouts
with permissive licenses only. Local fixtures (tiny_green + harbor motors)
remain offline; remote SHAs are immutable pins for discovery, hybrid
provenance, and later envbuild cloning.

M8 scale (2026-07-13): expanded public permissive inventory so git-clone-only
merge mining can feed ≥30 docker-oracle keeps without Oxylabs HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LanguageCode = Literal["python", "javascript", "typescript", "go", "rust"]

# Canonical language keys used in histogram / under-supply reporting.
SCALE_LANGUAGES: tuple[str, ...] = (
    "python",
    "typescript",
    "go",
    "javascript",
    "rust",
)


@dataclass(frozen=True, slots=True)
class SeedRepo:
    """One allowlisted base repo the synth producer / miner may use."""

    seed_id: str
    language: LanguageCode
    repo: str
    base_commit: str
    license: str = "MIT"
    description: str = ""
    local_fixture: str | None = None  # relative path under package root
    source_globs: tuple[str, ...] = ()
    f2p_commands: tuple[str, ...] = ()
    p2p_commands: tuple[str, ...] = ()
    install_commands: tuple[str, ...] = ()
    baseline_test_command: str = ""
    base_image: str = "python:3.12-slim"
    modular: bool = True
    notes: str = ""
    mine_priority: int = 100  # lower = try earlier for git-clone yield

    def resolve_local_path(self, package_root: Path | None = None) -> Path | None:
        """Return absolute path to local fixture repo root if present."""
        if not self.local_fixture:
            return None
        root = package_root or _package_root()
        path = (root / self.local_fixture).resolve()
        if path.is_dir():
            return path
        # Allow local_fixture to point at the "repo" leaf or its parent tree
        alt = (root / self.local_fixture / "repo").resolve()
        if alt.is_dir():
            return alt
        return path if path.exists() else None

    @property
    def repository_url(self) -> str:
        cleaned = (self.repo or "").strip()
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            return cleaned
        if cleaned.startswith("fixtures/") or cleaned.startswith("file:"):
            return f"file://{cleaned}"
        return f"https://github.com/{cleaned}"


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Offline modular seeds (always usable without network)
# ---------------------------------------------------------------------------

TINY_GREEN = SeedRepo(
    seed_id="fixture_tiny_green",
    language="python",
    repo="fixtures/tiny_green",
    # Valid hex SHA form (fixture synthetic pin, not a remote git object)
    base_commit="0000000000000000000000000000000000000001",
    license="MIT",
    description="Multi-module offline green Python fixture (math_ops + text_ops).",
    local_fixture="fixtures/tiny_green/repo",
    source_globs=("demo_pkg/*.py",),
    f2p_commands=("python -m pytest tests/test_math.py tests/test_text.py -q",),
    p2p_commands=("python -m pytest tests/test_ok.py -q",),
    install_commands=("pip install -q pytest",),
    baseline_test_command="python -m pytest -q",
    base_image="python:3.12-slim",
    modular=True,
    notes="Offline mandatory seed for synth producer unit + docker smoke.",
    mine_priority=1000,
)

# DeepSWE/Harbor multi-module motors (prefer over trivial single-function stubs)
HARBOR_PYTHON_ORDERS = SeedRepo(
    seed_id="harbor_python_orders",
    language="python",
    repo="fixtures/harbor_motors/python_orders",
    base_commit="a100000000000000000000000000000000000001",
    license="MIT",
    description="Multi-module Python order pipeline (pricing, inventory, checkout).",
    local_fixture="fixtures/harbor_motors/python_orders/repo",
    source_globs=("orderlib/*.py",),
    f2p_commands=(
        "python -m pytest tests/test_pricing.py tests/test_inventory.py tests/test_checkout.py -q",
    ),
    p2p_commands=("python -m pytest tests/ -q",),
    install_commands=("pip install -q pytest",),
    baseline_test_command="python -m pytest -q",
    base_image="python:3.12-slim",
    modular=True,
    notes="Harbor motor seed — multi-file hard floor offline.",
    mine_priority=1000,
)

HARBOR_GO_KVSTORE = SeedRepo(
    seed_id="harbor_go_kvstore",
    language="go",
    repo="fixtures/harbor_motors/go_kvstore",
    base_commit="b100000000000000000000000000000000000001",
    license="MIT",
    description="Go multi-file kv store + router motor for Harbor packs.",
    local_fixture="fixtures/harbor_motors/go_kvstore/repo",
    source_globs=("*.go",),
    f2p_commands=("go test ./...",),
    p2p_commands=("go test ./...",),
    install_commands=("go mod download",),
    baseline_test_command="go test ./...",
    base_image="golang:1.22",
    modular=True,
    notes="Harbor motor seed — structural multi-file Go faults.",
    mine_priority=1000,
)

HARBOR_TS_REGISTRY = SeedRepo(
    seed_id="harbor_ts_registry",
    language="typescript",
    repo="fixtures/harbor_motors/ts_registry",
    base_commit="c100000000000000000000000000000000000001",
    license="MIT",
    description="TypeScript/JS multi-module registry + catalog motor for Harbor packs.",
    local_fixture="fixtures/harbor_motors/ts_registry/repo",
    source_globs=("src/*.js",),
    f2p_commands=("npm test",),
    p2p_commands=("npm test",),
    install_commands=("npm install --no-audit --no-fund",),
    baseline_test_command="npm test",
    base_image="node:20-bookworm",
    modular=True,
    notes="Harbor motor seed — multi-module TS/JS hard track.",
    mine_priority=1000,
)


# ---------------------------------------------------------------------------
# Remote modular allowlist (permissive only) — git-clone history authority
# Pins are immutable full SHAs observed 2026-07-13 via git ls-remote HEAD.
# ---------------------------------------------------------------------------

REMOTE_SEEDS: tuple[SeedRepo, ...] = (
    # ---- Python (primary) -------------------------------------------------
    SeedRepo(
        seed_id="python_boltons",
        language="python",
        repo="mahmoud/boltons",
        base_commit="979fa9b613fa8c0a455ae16ea6f2ec91c11ecafe",
        license="BSD-3-Clause",
        description="Modular python utils; prefer small *utils.py targets.",
        source_globs=("boltons/*utils.py", "boltons/*.py"),
        f2p_commands=("pytest tests/test_mathutils.py tests/test_strutils.py -q",),
        p2p_commands=("pytest tests/test_dictutils.py -q",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=10,
    ),
    SeedRepo(
        seed_id="python_cachetools",
        language="python",
        repo="tkem/cachetools",
        base_commit="e164b7020e4211b57d20fe2b252d931af6244ad4",
        license="MIT",
        description="Modular caching library good for function-removal inversions.",
        source_globs=("src/cachetools/*.py", "cachetools/*.py"),
        f2p_commands=("pytest tests/test_lru.py tests/test_ttl.py -q",),
        p2p_commands=("pytest tests/test_lfu.py -q",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=10,
    ),
    SeedRepo(
        seed_id="python_click",
        language="python",
        repo="pallets/click",
        base_commit="b67832c2167e5b0ff6764a8c04a0a9087e697b5a",
        license="BSD-3-Clause",
        description="Composable CLI toolkit: multi-module command plumbing.",
        source_globs=("src/click/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="python_httpx",
        language="python",
        repo="encode/httpx",
        base_commit="b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
        license="BSD-3-Clause",
        description="HTTP client with modular transport / auth / models packages.",
        source_globs=("httpx/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="python_packaging",
        language="python",
        repo="pypa/packaging",
        base_commit="9d0ec47b36a1bcdf705b397b84cd89272410a00d",
        license="Apache-2.0",
        description="PyPA packaging version/specifiers; modular pure-Python.",
        source_globs=("src/packaging/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="python_httpcore",
        language="python",
        repo="encode/httpcore",
        base_commit="10a658221deb38a4c5b16db55ab554b0bf731707",
        license="BSD-3-Clause",
        description="Low-level HTTP transport used by httpx.",
        source_globs=("httpcore/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="python_jinja",
        language="python",
        repo="pallets/jinja",
        base_commit="5ef70112a1ff19c05324ff889dd30405b1002044",
        license="BSD-3-Clause",
        description="Jinja2 template engine modular runtime.",
        source_globs=("src/jinja2/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="python_markupsafe",
        language="python",
        repo="pallets/markupsafe",
        base_commit="b2e4d9c7687be25695fffbe93a37622302b24fb1",
        license="BSD-3-Clause",
        description="Safe string / markup helpers, small modular surface.",
        source_globs=("src/markupsafe/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=30,
    ),
    SeedRepo(
        seed_id="python_itsdangerous",
        language="python",
        repo="pallets/itsdangerous",
        base_commit="672971d66a2ef9f85151e53283113f33d642dabd",
        license="BSD-3-Clause",
        description="Cryptographic signing helpers for web apps.",
        source_globs=("src/itsdangerous/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=30,
    ),
    SeedRepo(
        seed_id="python_zipp",
        language="python",
        repo="jaraco/zipp",
        base_commit="29a7a55c6bac1a6f705b54135dbea82d03e997c3",
        license="MIT",
        description="Pathlib-compatible Zipfile wrapper.",
        source_globs=("zipp/*.py", "zipp.py"),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=30,
    ),
    # ---- Python (expanded dual-run-survivable surface, M14 yield) ---------
    SeedRepo(
        seed_id="python_flask",
        language="python",
        repo="pallets/flask",
        base_commit="36e4a824f340fdee7ed50937ba8e7f6bc7d17f81",
        license="BSD-3-Clause",
        description="Pallets microframework; multi-module request/app surface.",
        source_globs=("src/flask/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=18,
    ),
    SeedRepo(
        seed_id="python_werkzeug",
        language="python",
        repo="pallets/werkzeug",
        base_commit="1b00618e787f40dfb21eba29caf8f8be7c8e1d93",
        license="BSD-3-Clause",
        description="WSGI utility library; routing / serving modular sources.",
        source_globs=("src/werkzeug/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=18,
    ),
    SeedRepo(
        seed_id="python_attrs",
        language="python",
        repo="python-attrs/attrs",
        base_commit="d544dd2dacb4cbd660891e323ad92751ba65b4e0",
        license="MIT",
        description="Classes without boilerplate — pure-Python dual-run friend.",
        source_globs=("src/attr/*.py", "src/attrs/*.py"),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=15,
    ),
    SeedRepo(
        seed_id="python_urllib3",
        language="python",
        repo="urllib3/urllib3",
        base_commit="470374abd485b6705dfea82689b1e29c345785f1",
        license="MIT",
        description="HTTP client with multi-file connection/pool modules.",
        source_globs=("src/urllib3/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=18,
    ),
    SeedRepo(
        seed_id="python_idna",
        language="python",
        repo="kjd/idna",
        base_commit="5a954ffed62b15e2a313e5de0f949f60ed15416e",
        license="BSD-3-Clause",
        description="Internationalized Domain Names in Applications (pure-Python).",
        source_globs=("idna/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="python_more_itertools",
        language="python",
        repo="more-itertools/more-itertools",
        base_commit="da37f9de442b69fbcaa9f54fb042c2a6999473a6",
        license="MIT",
        description="More routines for operating on iterables (multi-module).",
        source_globs=("more_itertools/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=15,
    ),
    SeedRepo(
        seed_id="python_platformdirs",
        language="python",
        repo="tox-dev/platformdirs",
        base_commit="f68e56f00b522d620b81a8b186ff681282df8526",
        license="MIT",
        description="Determine platform-specific dirs (small pure-Python surface).",
        source_globs=("src/platformdirs/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="python_blinker",
        language="python",
        repo="pallets-eco/blinker",
        base_commit="c3364059663df1ddce32799d6b1922af89a345f6",
        license="MIT",
        description="Fast, simple object-to-object signaling (flask-adjacent).",
        source_globs=("src/blinker/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="python_requests",
        language="python",
        repo="psf/requests",
        base_commit="f361ead047be5cb873174218582f7d8b9fcd9f49",
        license="Apache-2.0",
        description="HTTP for Humans — multi-file sessions/adapters pure-Python.",
        source_globs=("src/requests/*.py", "requests/*.py"),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=22,
    ),
    SeedRepo(
        seed_id="python_rich",
        language="python",
        repo="Textualize/rich",
        base_commit="9d8f9a372cc5916fd4781fec207ced7ddac2f08f",
        license="MIT",
        description="Rich terminal formatting library (multi-module pure-Python).",
        source_globs=("rich/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=16,
    ),
    SeedRepo(
        seed_id="python_jsonschema",
        language="python",
        repo="python-jsonschema/jsonschema",
        base_commit="027429257d13ece4942c0ae501d3096a7a51f24a",
        license="MIT",
        description="JSON Schema validation implementation (modular).",
        source_globs=("jsonschema/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=18,
    ),
    SeedRepo(
        seed_id="python_tldextract",
        language="python",
        repo="john-kurkowski/tldextract",
        base_commit="f06eec7f1c86322554bcb4031b72a74699f04cf6",
        license="BSD-3-Clause",
        description="Accurately separate the TLD from domain strings.",
        source_globs=("tldextract/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="python_itemadapter",
        language="python",
        repo="scrapy/itemadapter",
        base_commit="4604b7a3174d2536768f87946b93961e09bfe941",
        license="BSD-3-Clause",
        description="Common interface for Scrapy item objects.",
        source_globs=("itemadapter/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=18,
    ),
    SeedRepo(
        seed_id="python_charset_normalizer",
        language="python",
        repo="jawah/charset_normalizer",
        base_commit="cc6840753f17f00dea4e339ce37507747217e916",
        license="MIT",
        description="Charset detection library (multi-file pure-Python).",
        source_globs=("charset_normalizer/*.py",),
        install_commands=("pip install -q pytest", "pip install -e ."),
        baseline_test_command="pytest -q",
        base_image="python:3.12-slim",
        modular=True,
        mine_priority=18,
    ),
    # python_chardet (chardet/chardet, LGPL-2.1-or-later) intentionally omitted:
    # REMOTE_SEEDS must stay fail-closed permissive only (is_permissive / no GPL family).
    # ---- Go ---------------------------------------------------------------
    SeedRepo(
        seed_id="go_cast",
        language="go",
        repo="spf13/cast",
        base_commit="6fd6afdc9df068c662e923a7c3ad04fda4647121",
        license="MIT",
        description="Go conversion helpers; mine multi-hunk structural mutations.",
        source_globs=("*.go",),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=10,
    ),
    SeedRepo(
        seed_id="go_uuid",
        language="go",
        repo="google/uuid",
        base_commit="2d3c2a9cc518326daf99a383f07c4d3c44317e4d",
        license="BSD-3-Clause",
        description="Go UUID library with modular version files for removal faults.",
        source_globs=("*.go",),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=10,
    ),
    SeedRepo(
        seed_id="go_chi",
        language="go",
        repo="go-chi/chi",
        base_commit="8b258c7bb28f97a5f2a856ff7ef962578fec9215",
        license="MIT",
        description="Lightweight router with modular middleware packages.",
        source_globs=("*.go", "middleware/*.go"),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="go_xid",
        language="go",
        repo="rs/xid",
        base_commit="643d1614ad4439dfd583d696953a5a57dc5c17e9",
        license="MIT",
        description="Globally unique id generator (compact multi-file Go).",
        source_globs=("*.go",),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="go_mapstructure",
        language="go",
        repo="mitchellh/mapstructure",
        base_commit="8508981c8b6c964e6986dd8aa85490e70ce3c2e2",
        license="MIT",
        description="Decode maps into native Go structs; multi-file decoder.",
        source_globs=("*.go",),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="go_semver",
        language="go",
        repo="masterminds/semver",
        base_commit="8b89c86cb53c57cfd5d07c13de12bc4d78954e99",
        license="MIT",
        description="Semantic version parsing and constraints (Go).",
        source_globs=("*.go",),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="go_multierror",
        language="go",
        repo="hashicorp/go-multierror",
        base_commit="6d4d48630db25c3c83fa83ecd41dd8438b82963c",
        license="MPL-2.0",
        description="Accumulate multiple errors in Go.",
        source_globs=("*.go",),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=30,
    ),
    SeedRepo(
        seed_id="go_cleanhttp",
        language="go",
        repo="hashicorp/go-cleanhttp",
        base_commit="9deb11f6076fad42bb47f2aa0595374d662331f2",
        license="MPL-2.0",
        description="Clean http.Client helpers without shared state.",
        source_globs=("*.go",),
        p2p_commands=("go test ./...",),
        install_commands=("go mod download",),
        baseline_test_command="go test ./...",
        base_image="golang:1.22",
        modular=True,
        mine_priority=30,
    ),
    # ---- JavaScript -------------------------------------------------------
    SeedRepo(
        seed_id="js_validator",
        language="javascript",
        repo="validatorjs/validator.js",
        base_commit="a38f15be6931722d295d2c7ad1d13d35036078a5",
        license="MIT",
        description="Multi-file src/lib/* validators; good multi-fault targets.",
        source_globs=("src/lib/*.js",),
        p2p_commands=("npm test",),
        install_commands=("npm install --no-audit --no-fund --legacy-peer-deps",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        notes="Exclude version self-test on non-release commits.",
        mine_priority=10,
    ),
    SeedRepo(
        seed_id="js_qs",
        language="javascript",
        repo="ljharb/qs",
        base_commit="3a890d4ecd3deb72a45d90be36f4f8c5970467c7",
        license="BSD-3-Clause",
        description="Query-string modular lib/ sources for multi-file inversions.",
        source_globs=("lib/*.js",),
        p2p_commands=("npm run tests-only",),
        install_commands=("npm install --no-audit --no-fund --legacy-peer-deps",),
        baseline_test_command="npm run tests-only",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=10,
    ),
    SeedRepo(
        seed_id="js_debug",
        language="javascript",
        repo="debug-js/debug",
        base_commit="f405ade8a4b7a0dc353e0f7390c1be90060f3621",
        license="MIT",
        description="Tiny multi-file debug logger used across the JS ecosystem.",
        source_globs=("src/*.js",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="js_chalk",
        language="javascript",
        repo="chalk/chalk",
        base_commit="aa06bb5ac3f14df9fda8cfb54274dfc165ddfdef",
        license="MIT",
        description="Terminal string styling with modular color tables.",
        source_globs=("source/*.js",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="js_ansi_styles",
        language="javascript",
        repo="chalk/ansi-styles",
        base_commit="faf414e7b479435b5a86b15ecb13fe89ecf5bd0e",
        license="MIT",
        description="ANSI escape code tables (JS).",
        source_globs=("index.js",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="js_uuid",
        language="javascript",
        repo="uuidjs/uuid",
        base_commit="ea83515d6a4de13a8f9d253fe772752c9dd7bbbe",
        license="MIT",
        description="UUID generation across modular ESM sources.",
        source_globs=("src/*.js",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="js_slash",
        language="javascript",
        repo="sindresorhus/slash",
        base_commit="98b618f5a3bfcb5dd374b204868818845b87bb2f",
        license="MIT",
        description="Convert Windows backslash paths to slash paths.",
        source_globs=("index.js",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=30,
    ),
    SeedRepo(
        seed_id="js_is_plain_obj",
        language="javascript",
        repo="sindresorhus/is-plain-obj",
        base_commit="97f38e8836f86a642cce98fc6ab3058bc36df181",
        license="MIT",
        description="Check if a value is a plain object.",
        source_globs=("index.js",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=30,
    ),
    # ---- TypeScript -------------------------------------------------------
    SeedRepo(
        seed_id="ts_zod",
        language="typescript",
        repo="colinhacks/zod",
        base_commit="912f0f51b0ced654d0069741e7160834dca742ee",
        license="MIT",
        description="TypeScript-first schema validation with modular codecs.",
        source_globs=("src/**/*.ts",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=15,
    ),
    SeedRepo(
        seed_id="ts_tslib",
        language="typescript",
        repo="microsoft/tslib",
        base_commit="12bd8a74b320e3acfaba36b0ecb0e14964a9165b",
        license="0BSD",
        description="TypeScript runtime helpers (modular emit support).",
        source_globs=("tslib.es6.js", "tslib.js"),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="ts_type_fest",
        language="typescript",
        repo="sindresorhus/type-fest",
        base_commit="f5cad3940db4e6b72ebbaad5c050361efef9dcc7",
        license="MIT",
        description="Collection of essential TypeScript types (modular d.ts).",
        source_globs=("source/**/*.ts",),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="ts_emittery",
        language="typescript",
        repo="sindresorhus/emittery",
        base_commit="147a8591045e00d0fe8088e2393e3eefea3aa4a5",
        license="MIT",
        description="Simple and modern async event emitter (TS-ready).",
        source_globs=("index.js", "maps.js"),
        install_commands=("npm install --no-audit --no-fund",),
        baseline_test_command="npm test",
        base_image="node:20-bookworm",
        modular=True,
        mine_priority=25,
    ),
    # ---- Rust (best-effort) ----------------------------------------------
    SeedRepo(
        seed_id="rust_log",
        language="rust",
        repo="rust-lang/log",
        base_commit="037d7a58f6ad184abb3afc4db81d37c43a5696ec",
        license="MIT",
        description="Logging facade for Rust crates (modular macros/impl).",
        source_globs=("src/**/*.rs",),
        p2p_commands=("cargo test",),
        install_commands=("cargo fetch",),
        baseline_test_command="cargo test",
        base_image="rust:1.78-bookworm",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="rust_thiserror",
        language="rust",
        repo="dtolnay/thiserror",
        base_commit="ec42ea70852f8db43e971700d1ccc184957d1b32",
        license="MIT",
        description="Derive macro for error types with modular impls.",
        source_globs=("src/**/*.rs",),
        p2p_commands=("cargo test",),
        install_commands=("cargo fetch",),
        baseline_test_command="cargo test",
        base_image="rust:1.78-bookworm",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="rust_anyhow",
        language="rust",
        repo="dtolnay/anyhow",
        base_commit="5bdb0e24db3994be119d42f18fe2d655e1f68f4a",
        license="MIT",
        description="Flexible Result error handling library for Rust.",
        source_globs=("src/**/*.rs",),
        p2p_commands=("cargo test",),
        install_commands=("cargo fetch",),
        baseline_test_command="cargo test",
        base_image="rust:1.78-bookworm",
        modular=True,
        mine_priority=20,
    ),
    SeedRepo(
        seed_id="rust_bitflags",
        language="rust",
        repo="bitflags/bitflags",
        base_commit="e077c4a679706661af508a91cfd96c6a7c4ac2d2",
        license="MIT",
        description="Macro to generate structures for bitflags.",
        source_globs=("src/**/*.rs",),
        p2p_commands=("cargo test",),
        install_commands=("cargo fetch",),
        baseline_test_command="cargo test",
        base_image="rust:1.78-bookworm",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="rust_byteorder",
        language="rust",
        repo="BurntSushi/byteorder",
        base_commit="5a82625fae462e8ba64cec8146b24a372b4d75c6",
        license="MIT",
        description="Reading/writing numbers in big-endian and little-endian.",
        source_globs=("src/**/*.rs",),
        p2p_commands=("cargo test",),
        install_commands=("cargo fetch",),
        baseline_test_command="cargo test",
        base_image="rust:1.78-bookworm",
        modular=True,
        mine_priority=25,
    ),
    SeedRepo(
        seed_id="rust_serde_json",
        language="rust",
        repo="serde-rs/json",
        base_commit="827a315bf2198558f0325b07bcc1e2cd973aba2f",
        license="MIT",
        description="JSON serialization for Serde (modular reader/writer).",
        source_globs=("src/**/*.rs",),
        p2p_commands=("cargo test",),
        install_commands=("cargo fetch",),
        baseline_test_command="cargo test",
        base_image="rust:1.78-bookworm",
        modular=True,
        mine_priority=25,
    ),
)


HARBOR_MOTOR_SEEDS: tuple[SeedRepo, ...] = (
    HARBOR_PYTHON_ORDERS,
    HARBOR_GO_KVSTORE,
    HARBOR_TS_REGISTRY,
)

ALLOWLIST: tuple[SeedRepo, ...] = (TINY_GREEN, *HARBOR_MOTOR_SEEDS, *REMOTE_SEEDS)


def get_seed(seed_id: str) -> SeedRepo:
    for seed in ALLOWLIST:
        if seed.seed_id == seed_id:
            return seed
    raise KeyError(f"unknown seed_id: {seed_id!r}; known={sorted(s.seed_id for s in ALLOWLIST)}")


def local_offline_seeds() -> list[SeedRepo]:
    """Seeds that can produce offline without cloning remotes."""
    out: list[SeedRepo] = []
    for seed in ALLOWLIST:
        path = seed.resolve_local_path()
        if path is not None and path.is_dir():
            out.append(seed)
    return out


def remote_mine_seeds(*, language: str | None = None) -> list[SeedRepo]:
    """Permissive remote seeds eligible for git-clone history mining."""
    seeds = list(REMOTE_SEEDS)
    if language is None:
        return sorted(seeds, key=lambda s: (s.mine_priority, s.seed_id))
    lang = normalize_language(language)
    return sorted(
        [s for s in seeds if s.language == lang],
        key=lambda s: (s.mine_priority, s.seed_id),
    )


def seeds_for_language(language: str) -> list[SeedRepo]:
    lang = normalize_language(language)
    if lang == "javascript":
        # Include typescript seeds when caller asks for broader JS family.
        return [s for s in ALLOWLIST if s.language in {"javascript", "typescript"}]
    return [s for s in ALLOWLIST if s.language == lang]


def normalize_language(language: str) -> str:
    """Map common aliases onto LanguageCode values."""
    raw = (language or "").strip().lower()
    aliases = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "node": "javascript",
        "nodejs": "javascript",
        "ts": "typescript",
        "golang": "go",
        "rs": "rust",
    }
    return aliases.get(raw, raw)


def harbor_motor_seeds() -> list[SeedRepo]:
    """Offline multi-module Harbor motors (Python/Go/TS)."""
    return list(HARBOR_MOTOR_SEEDS)


def language_histogram(
    seeds: list[SeedRepo] | tuple[SeedRepo, ...] | None = None,
) -> dict[str, int]:
    """Count seeds (or full allowlist remotes) per scale language, zeros included."""
    rows = list(seeds) if seeds is not None else list(REMOTE_SEEDS)
    counts = {lang: 0 for lang in SCALE_LANGUAGES}
    for seed in rows:
        lang = str(seed.language)
        if lang in counts:
            counts[lang] += 1
        else:
            counts[lang] = counts.get(lang, 0) + 1
    return counts


def under_supply_reasons(
    histogram: dict[str, int] | None = None,
    *,
    min_per_lang: int = 1,
) -> list[str]:
    """Honest under-supply narratives for scale languages with zero/low yield."""
    hist = histogram if histogram is not None else language_histogram()
    reasons: list[str] = []
    for lang in SCALE_LANGUAGES:
        count = int(hist.get(lang, 0))
        if count < min_per_lang:
            if lang in {"javascript", "rust"}:
                reasons.append(
                    f"{lang}: under-supply — inventory count={count} below floor "
                    f"{min_per_lang}; best-effort seeding, funnel may still be zero "
                    "after multi-file+tests filter."
                )
            else:
                reasons.append(
                    f"{lang}: under-supply — inventory count={count} below floor "
                    f"{min_per_lang}; expand allowlist or deepen merge-history mine."
                )
    return reasons


def allowlist_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in ALLOWLIST:
        local = seed.resolve_local_path()
        rows.append(
            {
                "seed_id": seed.seed_id,
                "language": seed.language,
                "repo": seed.repo,
                "base_commit": seed.base_commit,
                "license": seed.license,
                "modular": seed.modular,
                "mine_priority": seed.mine_priority,
                "repository_url": seed.repository_url,
                "local_available": bool(local and local.is_dir()),
                "local_path": str(local) if local and local.is_dir() else None,
                "remote": seed.seed_id in {s.seed_id for s in REMOTE_SEEDS},
            }
        )
    return rows


def scale_inventory_report() -> dict[str, object]:
    """Language classification + inventory stats for M8 funnel honesty."""
    remote_hist = language_histogram(REMOTE_SEEDS)
    full_hist = language_histogram([s for s in ALLOWLIST if s.language in SCALE_LANGUAGES])
    return {
        "scale_languages": list(SCALE_LANGUAGES),
        "remote_seed_count": len(REMOTE_SEEDS),
        "allowlist_count": len(ALLOWLIST),
        "remote_language_histogram": remote_hist,
        "allowlist_language_histogram": full_hist,
        "under_supply_reasons": under_supply_reasons(remote_hist),
        "licenses": sorted({s.license for s in REMOTE_SEEDS}),
        "history_authority": "git",
        "http_metadata_source_default": "none",
        "oxylabs_required_for_discover": False,
        "notes": (
            "DEFAULT LIVE PATH until Oxylabs present: git-clone-only merge-commit mine. "
            "HTTP GitHub page/raw via Oxylabs remains optional metadata only."
        ),
    }


__all__ = [
    "ALLOWLIST",
    "HARBOR_GO_KVSTORE",
    "HARBOR_MOTOR_SEEDS",
    "HARBOR_PYTHON_ORDERS",
    "HARBOR_TS_REGISTRY",
    "REMOTE_SEEDS",
    "SCALE_LANGUAGES",
    "TINY_GREEN",
    "LanguageCode",
    "SeedRepo",
    "allowlist_summary",
    "get_seed",
    "harbor_motor_seeds",
    "language_histogram",
    "local_offline_seeds",
    "normalize_language",
    "remote_mine_seeds",
    "scale_inventory_report",
    "seeds_for_language",
    "under_supply_reasons",
]
