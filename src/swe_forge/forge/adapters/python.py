"""Python language adapter.

Implements the language-specific behavior the env-first build and downstream
stages need for Python repositories: filesystem detection, the pinned base
image, dependency-install commands, the pytest test command (selection-aware),
test-file classification, AST parsing/mutation, and the in-Docker mutation run
(:meth:`mutation_tool_run`, via ``cosmic-ray``) used by the mutation-adequacy gate.
"""

from __future__ import annotations

import ast
import re
import shlex
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

from swe_forge.forge.adapters._fsdetect import has_root_marker, has_source_file
from swe_forge.forge.adapters._mutate import mutate_source
from swe_forge.forge.adapters.base import (
    LanguageAdapter,
    MutantStats,
    MutationExecutor,
    MutationOp,
    ParseError,
    Patch,
    PathLike,
    Symbol,
)


# Root manifests that unambiguously mark a Python project.
_PY_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "tox.ini",
)
_PY_EXTENSIONS = (".py",)
_REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "dev-requirements.txt",
)
_BUILD_MANIFESTS = ("pyproject.toml", "setup.py", "setup.cfg")
_TEST_FILE_RE = re.compile(r"(?:test_.*|.*_test)\.py$")
#: pytest summary lines: ``FAILED path::node - ...`` / ``ERROR path::node - ...``.
_PYTEST_FAILURE_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)

# Optional-dependency extras / dependency-group names that typically carry the
# repo's test dependencies, in priority order.
_TEST_GROUP_NAMES = ("test", "tests", "testing", "dev", "develop", "development")
# Dev/test requirement files preferred (and installed) before the runtime one.
_BASELINE_REQUIREMENT_FILES = (
    "requirements-dev.txt",
    "dev-requirements.txt",
    "requirements-test.txt",
    "test-requirements.txt",
    "requirements.txt",
)

_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


def _signature(node: _FuncDef) -> str:
    """Render ``def name(args)`` for a function/method, async-aware, no body."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    return f"{prefix} {node.name}({args})"


def _load_pyproject(root: Path) -> dict[str, object]:
    """Parse ``pyproject.toml`` to a dict; return ``{}`` if absent/unreadable."""
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError):
        return {}


def _test_extra(root: Path) -> str | None:
    """Return the first PEP 621 optional-dependency extra that carries tests."""
    project = _load_pyproject(root).get("project")
    if not isinstance(project, dict):
        return None
    optional = project.get("optional-dependencies")
    if not isinstance(optional, dict):
        return None
    for name in _TEST_GROUP_NAMES:
        if name in optional:
            return name
    return None


def _test_group_packages(root: Path) -> list[str]:
    """Return the package specs from the first PEP 735 test/dev dependency group."""
    groups = _load_pyproject(root).get("dependency-groups")
    if not isinstance(groups, dict):
        return []
    for name in _TEST_GROUP_NAMES:
        entries = groups.get(name)
        if isinstance(entries, list):
            # Skip non-string entries (e.g. {include-group = "..."}).
            return [entry for entry in entries if isinstance(entry, str)]
    return []


class _SymbolCollector(ast.NodeVisitor):
    """Collect every function/method declaration with its location.

    A function is classified ``"method"`` when its nearest enclosing scope is a
    class body and ``"function"`` otherwise; nested functions are included so
    every mutable definition in the file is locatable. Line spans are 1-based
    and inclusive, taken from the ``def`` line (decorators excluded) through the
    function's last line.
    """

    def __init__(self, file: str) -> None:
        self.file = file
        self.symbols: list[Symbol] = []
        self._class_depth = 0

    def _add(self, node: _FuncDef) -> None:
        self.symbols.append(
            Symbol(
                name=node.name,
                kind="method" if self._class_depth > 0 else "function",
                file=self.file,
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=_signature(node),
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: _FuncDef) -> None:
        self._add(node)
        # A function body is not a class scope: its nested defs are functions.
        saved = self._class_depth
        self._class_depth = 0
        for child in node.body:
            self.visit(child)
        self._class_depth = saved

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_depth += 1
        for child in node.body:
            self.visit(child)
        self._class_depth -= 1


class PythonAdapter(LanguageAdapter):
    """Adapter for Python repositories (``pyproject.toml`` / ``*.py``)."""

    name = "python"
    mutation_tool = "mutmut"
    mutation_tools = ("mutmut", "cosmic-ray")

    def detect(self, repo_path: PathLike) -> bool:
        return has_root_marker(repo_path, _PY_MARKERS) or has_source_file(
            repo_path, _PY_EXTENSIONS
        )

    def base_image(self) -> str:
        return "python:3.12-slim"

    def install_commands(self, repo_path: PathLike) -> list[str]:
        root = Path(repo_path)
        commands: list[str] = []
        for req in _REQUIREMENT_FILES:
            if (root / req).is_file():
                commands.append(f"pip install -r {req}")
        if any((root / manifest).is_file() for manifest in _BUILD_MANIFESTS):
            commands.append("pip install -e .")
        if not commands:
            commands.append("pip install pytest")
        return commands

    def test_command(self, selection: Sequence[str] | None = None) -> str:
        base = "python -m pytest"
        if selection:
            return f"{base} {' '.join(selection)}"
        return base

    def baseline_install_commands(self, repo_path: PathLike) -> list[str]:
        """Install the package plus its TEST/DEV dependencies for a green baseline.

        ``install_commands`` installs only the runtime package; a Python repo's
        test suite additionally needs its test dependencies, which live in PEP
        621 optional-dependency extras, PEP 735 ``[dependency-groups]``, or
        dev/test requirement files. This resolves whichever the repo uses and
        always guarantees ``pytest`` (the adapter standard runner) is present.
        """
        root = Path(repo_path)
        commands: list[str] = []

        for req in _BASELINE_REQUIREMENT_FILES:
            if (root / req).is_file():
                commands.append(f"pip install -r {req}")

        extra = _test_extra(root)
        has_manifest = any((root / m).is_file() for m in _BUILD_MANIFESTS)
        if has_manifest:
            if extra:
                commands.append(f"pip install -e '.[{extra}]'")
            else:
                commands.append("pip install -e .")

        if not extra:
            group_pkgs = _test_group_packages(root)
            if group_pkgs:
                commands.append(
                    "pip install " + " ".join(shlex.quote(p) for p in group_pkgs)
                )

        if not any("pytest" in cmd for cmd in commands):
            commands.append("pip install pytest")
        return commands

    def apply_p2p_exclusions(self, command: str, exclusions: Sequence[str]) -> str:
        """Append a pytest ``-k 'not (...)'`` filter that skips the named tests.

        pytest's ``-k`` matches a substring/expression against test node names,
        so the fix-independent self-tests are de-selected from the baseline/P2P
        run without touching any other test.
        """
        names = [e.strip() for e in exclusions if e.strip()]
        if not names:
            return command
        expr = "not (" + " or ".join(names) + ")"
        return f"{command} -k {shlex.quote(expr)}"

    def select_tests(self, command: str, names: Sequence[str]) -> str:
        """Append a pytest ``-k '(...)'`` filter that runs ONLY the named tests.

        pytest's ``-k`` expression de-selects everything that does not match, so
        the resulting command runs exactly the named (F2P-flipping) tests via the
        repo's own pytest runner.
        """
        selected = [n.strip() for n in names if n.strip()]
        if not selected:
            return command
        expr = "(" + " or ".join(selected) + ")"
        return f"{command} -k {shlex.quote(expr)}"

    def parse_test_failures(self, output: str) -> list[str]:
        """Return the ``-k``-usable names of the pytest tests that FAILED.

        Reads pytest's ``FAILED``/``ERROR`` summary lines and reduces each node id
        to a bare ``-k`` keyword: the last ``::`` segment with any ``[param]`` id
        stripped, and the trailing ``.func`` taken for a doctest node
        (``module.py::pkg.module.func``). Only valid identifiers are kept (a
        whole-file collection error has no per-test name to skip), de-duplicated
        in first-seen order. These feed :meth:`apply_p2p_exclusions` so a
        structural mutation's collateral failures are excluded from P2P.
        """
        names: list[str] = []
        seen: set[str] = set()
        for nodeid in _PYTEST_FAILURE_RE.findall(output):
            if "::" not in nodeid:
                # A whole-file collection error has no per-test name to skip.
                continue
            leaf = nodeid.split("::")[-1]
            leaf = leaf.split("[", 1)[0]
            leaf = leaf.rsplit(".", 1)[-1].strip()
            if leaf.isidentifier() and leaf not in seen:
                seen.add(leaf)
                names.append(leaf)
        return names

    def parse_collection_error_files(self, output: str) -> list[str]:
        """Return pytest test-file paths that failed at IMPORT/collection.

        A structural mutation that changes/removes a symbol a test module imports
        makes pytest fail to COLLECT that whole module: the summary carries an
        ``ERROR <file>`` line whose token has no per-test ``::`` node id (pytest
        exit 2). :meth:`parse_test_failures` skips those (there is no ``-k`` name),
        so this returns the erroring ``*.py`` module paths -- deduplicated in
        first-seen order -- for :meth:`apply_p2p_file_exclusions` to ``--ignore``
        wholesale from the P2P set. The synthesized F2P lives in a separate hidden
        file the baseline run never collects, so it can never appear here.
        """
        files: list[str] = []
        seen: set[str] = set()
        for token in _PYTEST_FAILURE_RE.findall(output):
            if "::" in token:
                continue
            path = token.strip()
            if path.endswith(".py") and path not in seen:
                seen.add(path)
                files.append(path)
        return files

    def apply_p2p_file_exclusions(self, command: str, files: Sequence[str]) -> str:
        """Append pytest ``--ignore=<path>`` flags skipping whole erroring modules.

        A test module the structural mutation makes uncollectable (import error,
        no per-test name) is ignored wholesale from the P2P/regression run. Only
        the derived collateral modules are ignored -- an unignored collection
        error still turns the P2P red so establish rejects rather than passing
        vacuously (no ``--continue-on-collection-errors``).
        """
        paths = [f.strip() for f in files if f.strip()]
        if not paths:
            return command
        flags = " ".join(f"--ignore={shlex.quote(p)}" for p in paths)
        return f"{command} {flags}"

    def is_test_file(self, path: PathLike) -> bool:
        return bool(_TEST_FILE_RE.fullmatch(Path(path).name))

    def parse_symbols(self, file: PathLike) -> list[Symbol]:
        path = Path(file)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            location = f"{path}:{exc.lineno or 0}"
            raise ParseError(
                f"failed to parse Python source {location}: {exc.msg}"
            ) from exc
        collector = _SymbolCollector(str(path))
        collector.visit(tree)
        return collector.symbols

    def mutate_ast(self, file: PathLike, symbol: Symbol, op: MutationOp) -> Patch:
        return mutate_source(self.name, file, symbol, op)

    async def mutation_tool_run(
        self,
        executor: MutationExecutor,
        *,
        target_files: Sequence[str],
        timeout: float = 1200.0,
        target_regions: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    ) -> MutantStats:
        """Run cosmic-ray against the gold target file(s) inside ``executor``.

        cosmic-ray (one of this adapter's ``mutation_tools``) gives structured,
        per-mutant output: it mutates each target module, runs the established
        suite (``pytest``, which collects the hidden test files the gate wrote)
        and reports killed/surviving mutants. Counts are aggregated across target
        files. The Python ``.pyc`` re-test determinism invariant is honored
        (``PYTHONDONTWRITEBYTECODE=1`` + a ``__pycache__`` purge before each run).

        When ``target_regions`` maps a target file to changed-symbol line ranges
        (1-based, inclusive), the cosmic-ray session for that file is SCOPED to
        those ranges after ``init`` and before ``exec`` -- bounding a difficulty-
        amplifier (``bug_combination`` / ``multi_file``) run on a large modular
        module to the actually-mutated region so it finishes within ``timeout``
        (the same file/region scoping the Go/JS tooling already do). Files with no
        entry are mutated whole, preserving the default behavior.
        """
        from swe_forge.forge.adapters._mutation_tools import (
            COSMIC_RAY_CONFIG,
            COSMIC_RAY_SCOPE_SCRIPT,
            COSMIC_RAY_SESSION,
            COSMIC_RAY_SETUP,
            MutationToolError,
            cosmicray_config,
            cosmicray_report_command,
            cosmicray_run_commands,
            cosmicray_scope_command,
            cosmicray_scope_script,
            parse_cosmicray_report,
        )

        sources = [str(f) for f in target_files if not self.is_test_file(f)]
        if not sources:
            raise MutationToolError("no non-test target files to mutate (python)")
        regions = dict(target_regions or {})

        for cmd in COSMIC_RAY_SETUP:
            res = await executor.run_command(cmd, timeout=timeout)
            if res.exit_code != 0:
                raise MutationToolError(
                    f"cosmic-ray install failed (exit {res.exit_code}): "
                    f"{(res.stderr or res.stdout)[:400]}"
                )

        test_command = "python -m pytest -x -q -p no:cacheprovider"
        env = {"PYTHONDONTWRITEBYTECODE": "1"}
        init_command, exec_command = cosmicray_run_commands()
        total = 0
        killed = 0
        survivors: list[str] = []
        for src in sources:
            await executor.run_command(
                "find . -name '__pycache__' -type d -prune -exec rm -rf {} + "
                "2>/dev/null; find . -name '*.pyc' -delete 2>/dev/null; "
                f"rm -f {COSMIC_RAY_CONFIG} {COSMIC_RAY_SESSION}; true",
                timeout=timeout,
            )
            await executor.write_file(
                COSMIC_RAY_CONFIG,
                cosmicray_config(src, test_command, timeout=max(30.0, timeout / 6)),
            )
            await executor.run_command(init_command, timeout=timeout, env=env)
            ranges = regions.get(src)
            if ranges:
                await executor.write_file(
                    COSMIC_RAY_SCOPE_SCRIPT, cosmicray_scope_script(ranges)
                )
                await executor.run_command(
                    cosmicray_scope_command(), timeout=timeout, env=env
                )
            await executor.run_command(exec_command, timeout=timeout, env=env)
            report = await executor.run_command(
                cosmicray_report_command(), timeout=timeout, env=env
            )
            counts = parse_cosmicray_report(report.stdout)
            total += counts.total
            killed += counts.killed
            survivors.extend(counts.survivors)

        return MutantStats(
            total=total,
            killed=killed,
            survived=total - killed,
            tool="cosmic-ray",
            survivors=tuple(survivors),
        )
