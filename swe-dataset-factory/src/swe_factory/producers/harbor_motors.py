"""DeepSWE/Harbor multi-lang motors: multi-file solution + held-out tests.

Offline motors for Python, Go, and TypeScript emit Harbor-ready materials:
- multi-file ``solution.patch`` (hard track floor ≥2 source files)
- held-out ``test.patch`` not present in the agent repo tree
- ``f2p_node_ids`` / ``p2p_node_ids`` for grader config
- long-horizon behavioral ``instruction.md`` (not a stub checklist)

Prefer real multi-module seeds over single-function ``NotImplemented`` stubs
as the sole hard set (VAL-HARBOR-007).
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from swe_factory.harbor.export_pack import (
    HarborExportError,
    HarborPackResult,
    export_harbor_pack,
    verify_pack_tree,
)
from swe_factory.harbor.grader_frame import (
    default_test_sh,
    default_tests_dockerfile,
    offline_environment_dockerfile,
)
from swe_factory.harbor.schema import (
    GradeConfig,
    HarborEnvironment,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
    validate_pack_spec,
)
from swe_factory.oracle.gates import MULTI_FILE_FLOOR, count_files_in_patch
from swe_factory.producers.harbor_labeling import (
    DualRunLabels,
    HarborLabelError,
    assert_held_out_verifier_only,
    compute_dual_run_labels,
    write_tests_config_json,
)
from swe_factory.producers.suite_reporters import grade_tool_label_for, reporter_info
from swe_factory.producers.synth import SynthError, unified_diff_trees
from swe_factory.sources.allowlist import SeedRepo, get_seed

HarborLang = Literal["python", "go", "typescript"]

_DEFAULT_IMAGE = "sha256:harbor_motor_pending"
HARD_MULTI_FILE_FLOOR = max(2, MULTI_FILE_FLOOR)


class HarborMotorError(RuntimeError):
    """Raised when a Harbor motor cannot emit a multi-file ready material set."""


@dataclass(frozen=True, slots=True)
class HeldOutTest:
    """One held-out test module (relative path under repo) + node ids it contributes."""

    relative_path: str
    content: str
    f2p_node_ids: tuple[str, ...] = ()
    p2p_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FaultPlan:
    """Multi-file structural fault plan applied to the green tree."""

    description: str
    # relative_path -> rewrite callable semantics encoded as replacements
    replacements: tuple[tuple[str, str, str], ...]  # path, old, new
    # Prefer structural multi-module faults; avoid sole NotImplemented stubs.
    uses_not_implemented: bool = False


@dataclass(frozen=True, slots=True)
class HarborMotorSeed:
    """Local multi-module motor seed (offline path)."""

    seed_id: str
    language: HarborLang
    display_name: str
    repo_label: str
    base_commit: str
    local_fixture: str  # path under package root to motor fixture directory
    green_modules: tuple[str, ...]
    fault: FaultPlan
    held_out: HeldOutTest
    base_f2p_node_ids: tuple[str, ...]
    base_p2p_node_ids: tuple[str, ...]
    license: str = "MIT"
    grade_tool_label: str = "pytest"
    grade_format: Literal["junit", "ctrf"] = "junit"
    hard_track: bool = True

    def fixture_root(self, package_root: Path | None = None) -> Path:
        root = package_root or _package_root()
        return (root / self.local_fixture).resolve()

    def green_repo(self, package_root: Path | None = None) -> Path:
        root = self.fixture_root(package_root)
        repo = root / "repo"
        if not repo.is_dir():
            raise HarborMotorError(f"green repo missing: {repo}")
        return repo


@dataclass(frozen=True, slots=True)
class HarborMaterials:
    """DeepSWE-ready emission from a motor (pre-pack or with pack)."""

    task_id: str
    language: HarborLang
    seed_id: str
    solution_patch: str
    test_patch: str
    instruction_md: str
    f2p_node_ids: tuple[str, ...]
    p2p_node_ids: tuple[str, ...]
    solution_files: tuple[str, ...]
    broken_workspace: Path
    green_workspace: Path | None
    multi_file_ok: bool
    hard_track: bool
    source_track: str = "synthetic_grounded"
    base_commit: str = ""
    repository_url: str = ""
    license: str = "MIT"
    provider_calls: int = 0
    dual_run: DualRunLabels | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def assert_hard_floor(self, floor: int = HARD_MULTI_FILE_FLOOR) -> None:
        if self.hard_track and len(self.solution_files) < floor:
            raise HarborMotorError(
                f"VAL-HARBOR-007 multi-file floor failed: "
                f"touched={list(self.solution_files)} floor={floor}"
            )


@dataclass(frozen=True, slots=True)
class HarborMotorResult:
    """Materials + optional Harbor pack export."""

    materials: HarborMaterials
    pack: HarborPackResult | None = None
    pack_dir: Path | None = None
    missing: tuple[str, ...] = ()


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Built-in multi-module motor seeds (Python / Go / TypeScript)
# ---------------------------------------------------------------------------

PYTHON_ORDERS_FAULT = FaultPlan(
    description=(
        "Pricing discount band inverted and inventory reservation non-atomic; "
        "checkout totals and multi-sku reserve diverge from the green contract."
    ),
    replacements=(
        (
            "orderlib/pricing.py",
            "if sub >= self.discount_threshold:\n            return round(sub * 0.9, 2)\n"
            "        return round(sub, 2)",
            "if sub >= self.discount_threshold:\n            return round(sub, 2)\n"
            "        return round(sub * 0.9, 2)",
        ),
        (
            "orderlib/inventory.py",
            "if not self.can_reserve(requests):\n            return False\n"
            "        for sku, qty in requests.items():\n"
            "            self._stock[sku] = self.available(sku) - int(qty)\n"
            "        return True",
            "for sku, qty in requests.items():\n"
            "            if self.available(sku) < qty:\n"
            "                continue\n"
            "            self._stock[sku] = self.available(sku) - int(qty)\n"
            "        return True",
        ),
        (
            "orderlib/checkout.py",
            "lines = [(item.unit_price, item.quantity) for item in items]\n"
            "        total = self.pricing.cart_total(lines)",
            "lines = [(item.unit_price, item.quantity) for item in items]\n"
            "        # Broken: skips discount/tax path and under-charges large carts\n"
            "        total = sum(self.pricing.line_total(p, q) for p, q in lines)",
        ),
    ),
    uses_not_implemented=False,
)

PYTHON_ORDERS_HELD_OUT = HeldOutTest(
    relative_path="tests/test_held_out.py",
    content='''\
"""Held-out multi-module checkout contract (agent never sees this file)."""

from orderlib.checkout import CheckoutService, LineItem
from orderlib.inventory import Inventory
from orderlib.pricing import PricingEngine


def test_multi_module_checkout_contract() -> None:
    """Cross-module: discount + tax + atomic multi-sku reserve."""
    inv = Inventory({"a": 5, "b": 5})
    pricing = PricingEngine(tax_rate=0.10, discount_threshold=50.0)
    svc = CheckoutService(inv, pricing)
    result = svc.place_order(
        [
            LineItem(sku="a", unit_price=20.0, quantity=2),
            LineItem(sku="b", unit_price=20.0, quantity=1),
        ]
    )
    assert result.accepted is True
    # subtotal 60 → discounted 54 → taxed 59.4
    assert result.total == 59.4
    assert result.remaining["a"] == 3
    assert result.remaining["b"] == 4
    # Second order must still reserve atomically
    again = svc.place_order(
        [
            LineItem(sku="a", unit_price=10.0, quantity=3),
            LineItem(sku="b", unit_price=10.0, quantity=5),
        ]
    )
    assert again.accepted is False
    assert again.reason == "insufficient_stock"
    assert again.remaining["a"] == 3
    assert again.remaining["b"] == 4
''',
    f2p_node_ids=("tests.test_held_out.test_multi_module_checkout_contract",),
)

GO_KV_FAULT = FaultPlan(
    description=(
        "Store Get returns empty string without existence signal mishandling; "
        "Router Upsert double-writes and Remove treats missing as success."
    ),
    replacements=(
        (
            "store.go",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "", false\n'
            "\t}\n"
            "\tv, ok := s.data[key]\n"
            "\treturn v, ok\n"
            "}",
            "func (s *Store) Get(key string) (string, bool) {\n"
            "\tif s.data == nil {\n"
            '\t\treturn "", true\n'
            "\t}\n"
            "\tv := s.data[key]\n"
            "\t// Broken: always reports ok=true even for missing keys\n"
            "\treturn v, true\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            "\treturn r.store.Size(), nil\n"
            "}",
            "func (r *Router) Upsert(key, value string) (int, error) {\n"
            "\tif err := r.Put(key, value); err != nil {\n"
            "\t\treturn 0, err\n"
            "\t}\n"
            "\t// Broken: put twice and report size-1\n"
            "\t_ = r.Put(key, value+value)\n"
            "\treturn r.store.Size() - 1, nil\n"
            "}",
        ),
        (
            "router.go",
            "func (r *Router) Remove(key string) error {\n"
            "\tif !r.store.Delete(key) {\n"
            '\t\treturn fmt.Errorf("missing key %q", key)\n'
            "\t}\n"
            "\treturn nil\n"
            "}",
            "func (r *Router) Remove(key string) error {\n"
            "\t// Broken: silent success for missing keys\n"
            "\t_ = r.store.Delete(key)\n"
            "\treturn nil\n"
            "}",
        ),
    ),
    uses_not_implemented=False,
)

GO_KV_HELD_OUT = HeldOutTest(
    relative_path="held_out_test.go",
    content="""\
package kvstore

import "testing"

// Held-out cross-module contract: store existence + router upsert/remove fidelity.
func TestHeldOutStoreRouterContract(t *testing.T) {
	s := NewStore()
	r := NewRouter(s)
	if _, ok := s.Get("missing"); ok {
		t.Fatal("missing key must not report ok")
	}
	n, err := r.Upsert("k", "v")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Fatalf("upsert size want 1 got %d", n)
	}
	v, err := r.Fetch("k")
	if err != nil || v != "v" {
		t.Fatalf("fetch want v got %q err=%v", v, err)
	}
	if err := r.Remove("gone"); err == nil {
		t.Fatal("remove missing must error")
	}
}
""",
    f2p_node_ids=("TestHeldOutStoreRouterContract",),
)

TS_REGISTRY_FAULT = FaultPlan(
    description=(
        "Catalog findByTag matches incorrectly and Registry register id sequence "
        "skips / reuses, breaking multi-module registration contracts."
    ),
    replacements=(
        (
            "src/catalog.js",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      if (meta.tags.includes(tag)) {\n"
            "        out.push({ id, ...meta });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
            "findByTag(tag) {\n"
            "    const out = [];\n"
            "    for (const [id, meta] of this.items.entries()) {\n"
            "      // Broken: returns first tag match only and drops id\n"
            "      if (meta.tags[0] === tag) {\n"
            "        out.push({ name: meta.name, tags: meta.tags });\n"
            "      }\n"
            "    }\n"
            "    return out;\n"
            "  }",
        ),
        (
            "src/registry.js",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    this._seq += 1;\n"
            "    const id = `svc-${this._seq}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags: [...tags] };\n"
            "  }",
            "register(name, tags = []) {\n"
            "    if (!name || !String(name).trim()) {\n"
            '      throw new Error("name required");\n'
            "    }\n"
            "    // Broken: does not advance sequence; collides ids; mutates tags in place\n"
            "    const id = `svc-${this._seq || 1}`;\n"
            "    this.catalog.add(id, name, tags);\n"
            "    return { id, name, tags };\n"
            "  }",
        ),
    ),
    uses_not_implemented=False,
)

TS_REGISTRY_HELD_OUT = HeldOutTest(
    relative_path="tests/held_out.test.js",
    content="""\
import { Registry } from "../src/registry.js";
import { Catalog } from "../src/catalog.js";

test("held-out multi-module registry contract", () => {
  const catalog = new Catalog();
  const reg = new Registry(catalog);
  const a = reg.register("alpha", ["core", "edge"]);
  const b = reg.register("beta", ["edge"]);
  expect(a.id).toBe("svc-1");
  expect(b.id).toBe("svc-2");
  expect(a.id).not.toBe(b.id);
  const byEdge = reg.listByTag("edge");
  expect(byEdge).toHaveLength(2);
  expect(byEdge.map((x) => x.id).sort()).toEqual(["svc-1", "svc-2"]);
  expect(reg.lookup(a.id)?.name).toBe("alpha");
  expect(reg.count()).toBe(2);
});
""",
    f2p_node_ids=("held-out multi-module registry contract",),
)


MOTOR_SEEDS: tuple[HarborMotorSeed, ...] = (
    HarborMotorSeed(
        seed_id="harbor_python_orders",
        language="python",
        display_name="Multi-module order pipeline (pricing + inventory + checkout)",
        repo_label="fixtures/harbor_motors/python_orders",
        base_commit="a100000000000000000000000000000000000001",
        local_fixture="fixtures/harbor_motors/python_orders",
        green_modules=(
            "orderlib/pricing.py",
            "orderlib/inventory.py",
            "orderlib/checkout.py",
        ),
        fault=PYTHON_ORDERS_FAULT,
        held_out=PYTHON_ORDERS_HELD_OUT,
        # Advisory defaults only; produce path recomputes from dual-run (VAL-HARBOR-006).
        base_f2p_node_ids=(
            "tests.test_pricing.test_discount_threshold",
            "tests.test_inventory.test_reserve_rejects_partial",
            "tests.test_checkout.test_place_order_success",
        ),
        base_p2p_node_ids=(
            "tests.test_pricing.test_line_total_basic",
            "tests.test_inventory.test_reserve_atomic",
            "tests.test_checkout.test_place_order_insufficient_stock",
        ),
        grade_tool_label="pytest",
        hard_track=True,
    ),
    HarborMotorSeed(
        seed_id="harbor_go_kvstore",
        language="go",
        display_name="Go kv store + router multi-file contract",
        repo_label="fixtures/harbor_motors/go_kvstore",
        base_commit="b100000000000000000000000000000000000001",
        local_fixture="fixtures/harbor_motors/go_kvstore",
        green_modules=("store.go", "router.go"),
        fault=GO_KV_FAULT,
        held_out=GO_KV_HELD_OUT,
        # Advisory defaults; dual-run may reassign (e.g. TestRouterRemoveMissing is F2P).
        base_f2p_node_ids=("TestRouterUpsert", "TestRouterRemoveMissing"),
        base_p2p_node_ids=("TestStoreDelete", "TestStoreSetGet"),
        grade_tool_label="go-test",
        hard_track=True,
    ),
    HarborMotorSeed(
        seed_id="harbor_ts_registry",
        language="typescript",
        display_name="TypeScript registry + catalog multi-module service",
        repo_label="fixtures/harbor_motors/ts_registry",
        base_commit="c100000000000000000000000000000000000001",
        local_fixture="fixtures/harbor_motors/ts_registry",
        green_modules=("src/catalog.js", "src/registry.js"),
        fault=TS_REGISTRY_FAULT,
        held_out=TS_REGISTRY_HELD_OUT,
        # Advisory defaults; dual-run recomputes (catalog findByTag is F2P).
        base_f2p_node_ids=(
            "catalog findByTag",
            "register assigns sequential ids",
        ),
        base_p2p_node_ids=(
            "catalog add and get",
            "lookup missing returns null",
        ),
        grade_tool_label="jest",
        hard_track=True,
    ),
)


def get_motor_seed(seed_id: str) -> HarborMotorSeed:
    for seed in MOTOR_SEEDS:
        if seed.seed_id == seed_id:
            return seed
    raise KeyError(
        f"unknown harbor motor seed_id={seed_id!r}; known={sorted(s.seed_id for s in MOTOR_SEEDS)}"
    )


def list_motor_seeds(*, language: str | None = None) -> list[HarborMotorSeed]:
    if language is None:
        return list(MOTOR_SEEDS)
    lang = language.strip().lower()
    if lang in {"ts", "js", "javascript"}:
        lang = "typescript"
    if lang == "py":
        lang = "python"
    return [s for s in MOTOR_SEEDS if s.language == lang]


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "node_modules",
            ".pytest_cache",
            "*.egg-info",
        ),
    )


def _apply_fault_plan(workspace: Path, fault: FaultPlan) -> list[str]:
    touched: list[str] = []
    for rel, old, new in fault.replacements:
        path = workspace / rel
        if not path.is_file():
            raise HarborMotorError(f"fault target missing: {rel}")
        text = path.read_text(encoding="utf-8")
        if old not in text:
            raise HarborMotorError(
                f"fault old-span not found in {rel} (seed drift?). snippet starts: {old[:80]!r}"
            )
        updated = text.replace(old, new, 1)
        if updated == text:
            raise HarborMotorError(f"fault had no effect on {rel}")
        path.write_text(updated, encoding="utf-8")
        if rel not in touched:
            touched.append(rel)
    if len(touched) < HARD_MULTI_FILE_FLOOR:
        raise HarborMotorError(
            f"fault plan must touch ≥{HARD_MULTI_FILE_FLOOR} files; got {touched}"
        )
    return touched


def build_held_out_test_patch(held_out: HeldOutTest) -> str:
    """Create a unified diff that *adds* the held-out test file to the repo."""
    rel = held_out.relative_path.lstrip("/")
    body = held_out.content
    if not body.endswith("\n"):
        body += "\n"
    lines = body.splitlines(keepends=True)
    n = len(lines)
    header = (
        f"diff --git a/{rel} b/{rel}\n"
        f"new file mode 100644\n"
        f"index 0000000..e69de29\n"
        f"--- /dev/null\n"
        f"+++ b/{rel}\n"
        f"@@ -0,0 +1,{n} @@\n"
    )
    payload = "".join("+" + (ln if ln.endswith("\n") else ln + "\n") for ln in lines)
    # Preserve exact content lines without double newlines from split
    if not lines:
        payload = "+\n"
        header = (
            f"diff --git a/{rel} b/{rel}\n"
            f"new file mode 100644\n"
            f"index 0000000..e69de29\n"
            f"--- /dev/null\n"
            f"+++ b/{rel}\n"
            f"@@ -0,0 +1 @@\n"
        )
        return header + payload
    rebuilt = []
    for ln in body.splitlines():
        rebuilt.append("+" + ln + "\n")
    return header + "".join(rebuilt)


def build_long_horizon_instruction(
    *,
    seed: HarborMotorSeed,
    fault_files: Sequence[str],
    f2p: Sequence[str],
    p2p: Sequence[str],
) -> str:
    """Behavioral multi-paragraph instruction (not a patch dump)."""
    modules = ", ".join(f"`{m}`" for m in seed.green_modules)
    faults = ", ".join(f"`{f}`" for f in fault_files)
    f2p_list = "\n".join(f"- `{n}`" for n in f2p)
    p2p_list = "\n".join(f"- `{n}`" for n in p2p) or "- (none)"
    return f"""\
# {seed.display_name}

## Context

You are restoring a multi-module **{seed.language}** package under long-horizon
agent evaluation. The repository implements a modular product path where
behaviour is composed across independent modules rather than a single helper.

Green modules that participate in the contract:
{modules}

A regression was introduced spanning **at least two source files**
({faults}). Surface symptoms:

- Cross-module invariants no longer hold (callers of one module observe
  incorrect results after state changes performed in another).
- Public-suite regression tests that already exist in the tree fall on the
  fail-to-pass set below.
- Pass-to-pass regressions listed below must stay green (no plastic fixes that
  delete or weaken coverage).

## Behavioural requirements

1. Restore the original multi-module contracts so that every fail-to-pass node
   passes when the solution is applied.
2. Do **not** remove, skip, or rewrite existing tests as a "fix". The graded
   whitelist is forced by the verifier; missing or failed whitelist nodes score 0.
3. Prefer minimal multi-file unified diffs under the repository root. Paths must
   look like `--- a/<rel>` / `+++ b/<rel>` with relative product paths.
4. Keep pass-to-pass behaviour intact for unrelated modules/branches.
5. The hard track requires a multi-file solution (≥2 product source files).
   Single-hunk NotImplemented stubs are not acceptable as the sole hard fix.

Fault description (non-leaking summary):
{seed.fault.description}

## Fail-to-pass nodes (must go red → green)

{f2p_list}

## Pass-to-pass nodes (must remain green)

{p2p_list}

## Deliverable

Work on a new branch from the base checkout. Emit a unified multi-file
`model.patch` that restores the green behavioural contract. Commit when done.

IMPORTANT: Please work on this in a new branch from main and commit everything
when you are done.
"""


def _default_test_sh_for_lang(language: HarborLang) -> str:
    if language == "python":
        return default_test_sh(language="python")
    if language == "go":
        return (
            "#!/bin/bash\n"
            "set -uo pipefail\n"
            "trap '"
            "if [ ! -f /logs/verifier/reward.json ] "
            "&& [ ! -f /logs/verifier/reward.txt ]; "
            "then mkdir -p /logs/verifier; "
            "echo -1 > /logs/verifier/reward.txt; fi' EXIT\n"
            'log() { echo "[verifier] $*"; }\n'
            "cd /app || { mkdir -p /logs/verifier; exit 6; }\n"
            "\n"
            "python3 /tests/grader.py prepare || exit $?\n"
            "[ -f /logs/verifier/reward.json ] && exit 0\n"
            "\n"
            "set +e\n"
            # -json alone lacks --- FAIL lines; include -v for PASS/FAIL report lines
            "go test ./... -count=1 -v "
            "> /logs/verifier/new.log 2>&1\n"
            "new_rc=$?\n"
            "# Minimal junit wrapper for go test function names\n"
            "python3 - <<'PY'\n"
            "import re, pathlib\n"
            "from xml.etree.ElementTree import Element, SubElement, ElementTree\n"
            "raw = pathlib.Path('/logs/verifier/new.log').read_text(errors='replace')\n"
            "names = sorted(set(re.findall(r'--- (?:PASS|FAIL): (\\w+)', raw)))\n"
            "root = Element('testsuite', name='go')\n"
            "for n in names:\n"
            "    status = 'failed' if f'--- FAIL: {n}' in raw else 'passed'\n"
            "    tc = SubElement(root, 'testcase', classname='', name=n)\n"
            "    if status == 'failed':\n"
            "        SubElement(tc, 'failure', message='failed')\n"
            "ElementTree(root).write('/logs/verifier/new.xml')\n"
            "pathlib.Path('/logs/verifier/base.xml').write_text('<testsuite/>')\n"
            "PY\n"
            "set -e\n"
            'log "go test rc=$new_rc"\n'
            "python3 /tests/grader.py grade || exit $?\n"
        )
    # typescript / js via jest-compatible files using node assert if jest missing
    return (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "trap '"
        "if [ ! -f /logs/verifier/reward.json ] "
        "&& [ ! -f /logs/verifier/reward.txt ]; "
        "then mkdir -p /logs/verifier; "
        "echo -1 > /logs/verifier/reward.txt; fi' EXIT\n"
        'log() { echo "[verifier] $*"; }\n'
        "cd /app || { mkdir -p /logs/verifier; exit 6; }\n"
        "\n"
        "python3 /tests/grader.py prepare || exit $?\n"
        "[ -f /logs/verifier/reward.json ] && exit 0\n"
        "\n"
        "set +e\n"
        "if [ -f package.json ] && command -v npm >/dev/null 2>&1; then\n"
        "  npm test -- --json --outputFile=/logs/verifier/jest.json "
        "> /logs/verifier/new.log 2>&1\n"
        "  new_rc=$?\n"
        "else\n"
        "  node --test tests/ > /logs/verifier/new.log 2>&1\n"
        "  new_rc=$?\n"
        "fi\n"
        "python3 - <<'PY'\n"
        "import json, re, pathlib\n"
        "from xml.etree.ElementTree import Element, SubElement, ElementTree\n"
        "root = Element('testsuite', name='js')\n"
        "jest_path = pathlib.Path('/logs/verifier/jest.json')\n"
        "log = pathlib.Path('/logs/verifier/new.log').read_text(errors='replace')\n"
        "names_status = []  # list[(name, failed)]\n"
        "if jest_path.is_file():\n"
        "    try:\n"
        "        data = json.loads(jest_path.read_text(errors='replace') or '{}')\n"
        "    except Exception:\n"
        "        data = {}\n"
        "    for tr in (data.get('testResults') or []):\n"
        "        for a in (tr.get('assertionResults') or []):\n"
        "            nm = (a.get('title') or a.get('fullName') or '').strip()\n"
        "            if not nm:\n"
        "                continue\n"
        "            st = (a.get('status') or '').lower()\n"
        "            names_status.append((nm, st in ('failed', 'fail')))\n"
        "if not names_status:\n"
        "    for m in re.finditer(r'(?:✓|PASS|√)\\s+(.+)', log):\n"
        "        names_status.append((m.group(1).strip(), False))\n"
        "    for m in re.finditer(r'(?:✕|FAIL|×|x)\\s+(.+)', log):\n"
        "        names_status.append((m.group(1).strip(), True))\n"
        "    # jest verbose lines: FAIL path\\n  ● name\n"
        "    for m in re.finditer(r'[●•]\\s+(.+)', log):\n"
        "        names_status.append((m.group(1).strip(), True))\n"
        '    for m in re.finditer(r"test\\([\'\\"](.+?)[\'\\"]\\)", log):\n'
        "        pass  # names only from runtime output\n"
        "for nm, failed in names_status:\n"
        "    tc = SubElement(root, 'testcase', classname='', name=nm)\n"
        "    if failed:\n"
        "        SubElement(tc, 'failure', message='failed')\n"
        "ElementTree(root).write('/logs/verifier/new.xml')\n"
        "pathlib.Path('/logs/verifier/base.xml').write_text('<testsuite/>')\n"
        "PY\n"
        "set -e\n"
        'log "js/ts test rc=$new_rc"\n'
        "python3 /tests/grader.py grade || exit $?\n"
    )


def _offline_dockerfile_for_lang(language: HarborLang) -> str:
    if language == "python":
        return offline_environment_dockerfile()
    if language == "go":
        return """\
FROM golang:1.22-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git python3 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY repo/ /app/
RUN git init \\
 && git config user.email "fixture@local" \\
 && git config user.name "fixture" \\
 && git add -A \\
 && git commit -q -m "base" \\
 && git checkout -B main \\
 && git rev-parse HEAD > /app/.harbor_base_commit \\
 && printf '%s\\n' '.harbor_base_commit' >> /app/.git/info/exclude \\
 && true

ENV GOTOOLCHAIN=local
RUN git config core.hooksPath /dev/null || true
CMD ["/bin/bash"]
"""
    return """\
FROM node:20-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git python3 \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY repo/ /app/
RUN git init \\
 && git config user.email "fixture@local" \\
 && git config user.name "fixture" \\
 && git add -A \\
 && git commit -q -m "base" \\
 && git checkout -B main \\
 && git rev-parse HEAD > /app/.harbor_base_commit \\
 && printf '%s\\n' '.harbor_base_commit' >> /app/.git/info/exclude \\
 && true

# Offline fixtures may run without full lockfiles; install best-effort.
RUN if [ -f package.json ]; then npm install --no-audit --no-fund || true; fi
RUN git config core.hooksPath /dev/null || true
CMD ["/bin/bash"]
"""


def produce_harbor_materials(
    seed: HarborMotorSeed,
    *,
    work_root: Path | None = None,
    instance_suffix: str | None = None,
    package_root: Path | None = None,
    keep_green: bool = True,
    recompute_labels: bool = True,
) -> HarborMaterials:
    """Run one offline multi-lang motor → DeepSWE-ready materials (no pack write).

    By default node-id F2P/P2P cohorts are **recomputed** from dual-run
    broken-vs-green suite outcomes (VAL-HARBOR-006). Hand-authored
    ``base_*_node_ids`` are fallbacks only when ``recompute_labels=False``.
    """
    green_src = seed.green_repo(package_root)
    work = Path(work_root) if work_root else Path(tempfile.mkdtemp(prefix="sdf-harbor-motor-"))
    work.mkdir(parents=True, exist_ok=True)
    suffix = instance_suffix or "offline"
    case = work / f"{seed.seed_id}_{suffix}"
    if case.exists():
        shutil.rmtree(case)
    green_ws = case / "green"
    broken_ws = case / "broken"
    _copy_tree(green_src, green_ws)
    _copy_tree(green_src, broken_ws)

    fault_files = _apply_fault_plan(broken_ws, seed.fault)
    try:
        solution_patch = unified_diff_trees(broken_ws, green_ws)
    except SynthError as exc:
        raise HarborMotorError(str(exc)) from exc

    # Strip .sdf_keep noise if present
    solution_patch = (
        re.sub(
            r"diff --git a/\.sdf_keep b/\.sdf_keep[\s\S]*?(?=diff --git|\Z)",
            "",
            solution_patch,
        ).strip()
        + "\n"
    )

    sol_files = tuple(count_files_in_patch(solution_patch))
    # Hold out tests must not appear in solution files
    test_files = [p for p in sol_files if p.startswith("tests/") or p.endswith("_test.go")]
    product_files = tuple(p for p in sol_files if p not in test_files)
    if len(product_files) < HARD_MULTI_FILE_FLOOR and seed.hard_track:
        raise HarborMotorError(
            f"solution.patch multi-file product floor failed: files={list(sol_files)}"
        )

    test_patch = build_held_out_test_patch(seed.held_out)
    if not test_patch.strip():
        raise HarborMotorError("held-out test.patch must be non-empty")

    # Ensure held-out path is NOT present in agent (broken) workspace
    held_path = broken_ws / seed.held_out.relative_path
    if held_path.exists():
        held_path.unlink()

    dual: DualRunLabels | None = None
    if recompute_labels:
        try:
            dual = compute_dual_run_labels(
                language=seed.language,
                green_repo=green_ws,
                broken_repo=broken_ws,
                held_out_relative_path=seed.held_out.relative_path,
                held_out_content=seed.held_out.content,
            )
        except HarborLabelError as exc:
            raise HarborMotorError(f"dual-run F2P/P2P labeling failed: {exc}") from exc
        f2p = dual.f2p_node_ids
        p2p = dual.p2p_node_ids
    else:
        f2p = tuple(dict.fromkeys([*seed.base_f2p_node_ids, *seed.held_out.f2p_node_ids]))
        p2p = tuple(dict.fromkeys([*seed.base_p2p_node_ids, *seed.held_out.p2p_node_ids]))
    if not f2p:
        raise HarborMotorError("f2p_node_ids must be non-empty")

    instruction = build_long_horizon_instruction(
        seed=seed,
        fault_files=fault_files,
        f2p=f2p,
        p2p=p2p,
    )
    if len(instruction.strip()) < 400:
        raise HarborMotorError("instruction.md too short for long-horizon behavioral prompt")

    task_id = f"harbor-{seed.language}-{seed.seed_id.replace('harbor_', '')}-{suffix}"
    task_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", task_id).strip("-").lower()

    notes: dict[str, Any] = {
        "derivation": "harbor_motor_inverse",
        "fault": seed.fault.description,
        "fault_files": list(fault_files),
        "held_out_path": seed.held_out.relative_path,
        "uses_not_implemented": seed.fault.uses_not_implemented,
        "green_modules": list(seed.green_modules),
        "hard_multi_file_floor": HARD_MULTI_FILE_FLOOR,
        "label_method": "dual_run_broken_vs_green" if dual is not None else "seed_base_ids",
        "seed_base_f2p_node_ids": list(seed.base_f2p_node_ids),
        "seed_base_p2p_node_ids": list(seed.base_p2p_node_ids),
    }
    if dual is not None:
        notes["dual_run"] = dual.to_dict()
    try:
        notes["suite_reporter"] = reporter_info(seed.language).to_dict()
    except HarborLabelError:
        notes["suite_reporter"] = {"language": seed.language}

    materials = HarborMaterials(
        task_id=task_id,
        language=seed.language,
        seed_id=seed.seed_id,
        solution_patch=solution_patch,
        test_patch=test_patch,
        instruction_md=instruction,
        f2p_node_ids=f2p,
        p2p_node_ids=p2p,
        solution_files=product_files if product_files else sol_files,
        broken_workspace=broken_ws,
        green_workspace=green_ws if keep_green else None,
        multi_file_ok=len(product_files if product_files else sol_files) >= HARD_MULTI_FILE_FLOOR,
        hard_track=seed.hard_track,
        source_track="synthetic_grounded",
        base_commit=seed.base_commit,
        repository_url=f"file://{seed.repo_label}",
        license=seed.license,
        provider_calls=0,
        dual_run=dual,
        notes=notes,
    )
    materials.assert_hard_floor()
    (case / "solution.patch").write_text(solution_patch, encoding="utf-8")
    (case / "test.patch").write_text(test_patch, encoding="utf-8")
    (case / "instruction.md").write_text(instruction, encoding="utf-8")
    # Harbor-style tests/config.json beside materials (VAL-LABEL-003)
    tool = seed.grade_tool_label
    try:
        tool = grade_tool_label_for(seed.language)
    except HarborLabelError:
        tool = seed.grade_tool_label
    cfg_path = write_tests_config_json(
        case / "config.json",
        base_commit=seed.base_commit,
        f2p_node_ids=f2p,
        p2p_node_ids=p2p,
        grade={
            "format": seed.grade_format,
            "node_id": "name",
            "tool_label": tool,
            "reports": ["/logs/verifier/new.xml", "/logs/verifier/base.xml"],
        },
        extra={"label_method": notes["label_method"]},
    )
    nodes_payload: dict[str, Any] = {
        "f2p_node_ids": list(f2p),
        "p2p_node_ids": list(p2p),
        "label_method": notes["label_method"],
        "config_json": str(cfg_path.name),
    }
    if dual is not None:
        nodes_payload["dual_run_summary"] = {
            "green_passed": list(dual.green.passed),
            "broken_failed": list(dual.broken.failed),
            "broken_passed": list(dual.broken.passed),
        }
    (case / "nodes.json").write_text(
        json.dumps(nodes_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return materials


def materials_to_pack_spec(
    materials: HarborMaterials,
    seed: HarborMotorSeed,
    *,
    agent_image_ref: str | None = None,
) -> HarborPackSpec:
    """Assemble a validated HarborPackSpec from motor materials."""
    image = agent_image_ref or f"harbor-sdf-{materials.task_id}:motor"
    task_toml = HarborTaskToml(
        schema_version="1.1",
        task=HarborTaskIdentity(
            name=f"swe-factory/{materials.task_id}",
            description=seed.display_name,
            keywords=[materials.language, "hard", "multi-file", "harbor"],
        ),
        metadata=HarborMetadata(
            language=materials.language if materials.language != "typescript" else "typescript",
            repository_url=materials.repository_url,
            base_commit_hash=materials.base_commit,
            task_id=materials.task_id,
            ext_id=f"sdf-harbor-motor-{seed.seed_id}",
            display_title=seed.display_name,
            display_description=seed.fault.description[:200],
            original_title=seed.display_name,
            category="hard-multifile",
            source_track=materials.source_track,
            license=materials.license,
        ),
        verifier=HarborVerifier(environment_mode="separate", timeout_sec=900.0),
        environment=HarborEnvironment(
            docker_image=image,
            cpus=1,
            memory_mb=2048,
            storage_mb=4096,
            allow_internet=False,
        ),
    )
    tool_label = seed.grade_tool_label
    try:
        tool_label = grade_tool_label_for(materials.language)
    except HarborLabelError:
        tool_label = seed.grade_tool_label
    tests_config = TestsConfig(
        base_commit=materials.base_commit,
        f2p_node_ids=list(materials.f2p_node_ids),
        p2p_node_ids=list(materials.p2p_node_ids),
        grade=GradeConfig(
            format=seed.grade_format,
            node_id="name",
            tool_label=tool_label,
            reports=["/logs/verifier/new.xml", "/logs/verifier/base.xml"],
        ),
    )
    spec = HarborPackSpec(
        task_id=materials.task_id,
        instruction_md=materials.instruction_md,
        task_toml=task_toml,
        tests_config=tests_config,
        solution_patch=materials.solution_patch,
        test_patch=materials.test_patch,
        environment_dockerfile=_offline_dockerfile_for_lang(materials.language),
        tests_dockerfile=default_tests_dockerfile(agent_image_ref=image),
        test_sh=_default_test_sh_for_lang(materials.language),
    )
    return validate_pack_spec(spec)


def produce_harbor_pack(
    seed: HarborMotorSeed | str,
    *,
    out_dir: Path | str,
    work_root: Path | None = None,
    instance_suffix: str | None = None,
    package_root: Path | None = None,
    overwrite: bool = True,
) -> HarborMotorResult:
    """Produce materials for one motor and write a complete Harbor pack tree."""
    if isinstance(seed, str):
        seed = get_motor_seed(seed)
    materials = produce_harbor_materials(
        seed,
        work_root=work_root,
        instance_suffix=instance_suffix,
        package_root=package_root,
    )
    # Agent isolation: held-out file must not live in broken workspace before copy
    held = Path(materials.broken_workspace) / seed.held_out.relative_path
    if held.exists():
        held.unlink()

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    try:
        pack = export_harbor_pack(
            materials_to_pack_spec(materials, seed),
            dest=tasks / materials.task_id,
            overwrite=overwrite,
            copy_repo_into_environment=materials.broken_workspace,
        )
    except HarborExportError as exc:
        raise HarborMotorError(str(exc)) from exc

    missing = tuple(verify_pack_tree(pack.pack_dir))
    # Double-check agent isolation of held-out + multi-file solution
    env_repo = pack.pack_dir / "environment" / "repo"
    if env_repo.is_dir():
        leaked = env_repo / seed.held_out.relative_path
        if leaked.exists():
            raise HarborMotorError(
                f"held-out test file leaked into agent environment repo: {leaked}"
            )
        if (pack.pack_dir / "environment" / "solution").exists():
            raise HarborMotorError("solution/ leaked into environment context")
        isolation_hits = assert_held_out_verifier_only(
            agent_context=env_repo,
            test_patch_path=pack.pack_dir / "tests" / "test.patch",
            held_out_relative_paths=(seed.held_out.relative_path,),
        )
        if isolation_hits:
            raise HarborMotorError(f"held-out / solution leak into agent context: {isolation_hits}")
    sol_files = count_files_in_patch(
        (pack.pack_dir / "solution" / "solution.patch").read_text(encoding="utf-8")
    )
    product = [p for p in sol_files if not p.startswith("tests/")]
    if seed.hard_track and len(product) < HARD_MULTI_FILE_FLOOR:
        raise HarborMotorError(f"pack solution.patch multi-file floor failed: {product}")
    cfg = json.loads((pack.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
    if not cfg.get("f2p_node_ids"):
        raise HarborMotorError("config.json missing f2p_node_ids")
    if not (pack.pack_dir / "tests" / "test.patch").read_text(encoding="utf-8").strip():
        raise HarborMotorError("tests/test.patch empty")

    return HarborMotorResult(
        materials=materials,
        pack=pack,
        pack_dir=pack.pack_dir,
        missing=missing,
    )


def produce_all_offline_motors(
    *,
    out_dir: Path | str,
    work_root: Path | None = None,
    languages: Sequence[str] | None = None,
    instance_suffix: str = "offline",
) -> list[HarborMotorResult]:
    """Emit Harbor packs for all (or language-filtered) offline motors."""
    seeds = list(MOTOR_SEEDS)
    if languages:
        wanted: set[str] = set()
        for lang in languages:
            code = lang.strip().lower()
            if code in {"ts", "js", "javascript"}:
                code = "typescript"
            if code == "py":
                code = "python"
            wanted.add(code)
        seeds = [s for s in seeds if s.language in wanted]
    if not seeds:
        raise HarborMotorError("no motors match language filter")

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    results: list[HarborMotorResult] = []
    for seed in seeds:
        results.append(
            produce_harbor_pack(
                seed,
                out_dir=root,
                work_root=work_root,
                instance_suffix=instance_suffix,
            )
        )
    manifest = {
        "count": len(results),
        "task_ids": [r.materials.task_id for r in results],
        "languages": sorted({r.materials.language for r in results}),
        "multi_file": {r.materials.task_id: list(r.materials.solution_files) for r in results},
        "provider_calls": 0,
        "mode": "harbor_motors_offline",
        "hard_multi_file_floor": HARD_MULTI_FILE_FLOOR,
    }
    (root / "pack_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


def seed_repo_from_motor(seed: HarborMotorSeed) -> SeedRepo:
    """Project a HarborMotorSeed into the legacy SeedRepo shape for reuse."""
    return SeedRepo(
        seed_id=seed.seed_id,
        language=seed.language,
        repo=seed.repo_label,
        base_commit=seed.base_commit,
        license=seed.license,
        description=seed.display_name,
        local_fixture=f"{seed.local_fixture}/repo",
        modular=True,
        notes="DeepSWE Harbor multi-module motor seed (offline).",
    )


def ensure_motors_registered_in_allowlist() -> list[str]:
    """Return motor seed_ids; allowlist registration is optional for discovery."""
    return [s.seed_id for s in MOTOR_SEEDS]


# Sort mutations by legacy get_seed compatibility helper
def try_get_motor_or_legacy(seed_id: str) -> HarborMotorSeed | SeedRepo:
    try:
        return get_motor_seed(seed_id)
    except KeyError:
        return get_seed(seed_id)


__all__ = [
    "HARD_MULTI_FILE_FLOOR",
    "MOTOR_SEEDS",
    "FaultPlan",
    "HarborLang",
    "HarborMaterials",
    "HarborMotorError",
    "HarborMotorResult",
    "HarborMotorSeed",
    "HeldOutTest",
    "build_held_out_test_patch",
    "build_long_horizon_instruction",
    "ensure_motors_registered_in_allowlist",
    "get_motor_seed",
    "list_motor_seeds",
    "materials_to_pack_spec",
    "produce_all_offline_motors",
    "produce_harbor_materials",
    "produce_harbor_pack",
    "seed_repo_from_motor",
]
