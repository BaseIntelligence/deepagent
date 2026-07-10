"""Pure, language-specific mutation-tool command construction and output parsing.

The in-Docker mutation run (``LanguageAdapter.mutation_tool_run``) is the only
async/Docker piece of the mutation-adequacy gate; everything tool-specific that
can be tested *offline* lives here as pure functions:

* the install commands for each tool,
* the config files / run commands the adapter writes and executes in the sandbox,
* and the parsers that turn each tool's textual/JSON output into a tool-agnostic
  :class:`ToolCounts` (``total``/``killed`` plus survivor descriptions).

Tools, per language adapter:

* **Python** -> ``cosmic-ray`` (structured ``cr-report`` output; one of the
  Python adapter's ``mutation_tools = ("mutmut", "cosmic-ray")``).
* **JS/TS** -> ``@stryker-mutator/core`` (JSON mutation report).
* **Go**    -> ``go-mutesting`` (``The mutation score is ...`` summary line).

Keeping this logic pure means the parsers are unit-tested against captured real
tool output without spinning a container, while the adapter only orchestrates the
sandbox I/O.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass


class MutationToolError(RuntimeError):
    """Raised when a mutation tool cannot run or its output cannot be parsed."""


@dataclass(frozen=True)
class ToolCounts:
    """Tool-agnostic mutation result: totals plus surviving-mutant descriptions.

    ``total`` is the number of *detectable* mutants the tool generated,
    ``killed`` the number the suite caught, and ``survivors`` short
    human-readable descriptions of the mutants that escaped (used to guide the
    teacher during auto-synthesis).
    """

    total: int
    killed: int
    survivors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.total < 0 or self.killed < 0:
            raise MutationToolError("mutation counts must be non-negative")
        if self.killed > self.total:
            raise MutationToolError(
                f"killed ({self.killed}) cannot exceed total ({self.total})"
            )

    @property
    def survived(self) -> int:
        return self.total - self.killed

    @property
    def kill_ratio(self) -> float:
        return self.killed / self.total if self.total else 0.0


# --------------------------------------------------------------------------- #
# Python: cosmic-ray
# --------------------------------------------------------------------------- #
COSMIC_RAY_CONFIG = "swe_forge_cr.toml"
COSMIC_RAY_SESSION = "swe_forge_cr.sqlite"
COSMIC_RAY_SETUP: tuple[str, ...] = ("pip install --quiet cosmic-ray",)


def cosmicray_config(
    module_path: str, test_command: str, *, timeout: float = 60.0
) -> str:
    """Render a cosmic-ray config TOML mutating ``module_path``.

    ``test-command`` is the suite cosmic-ray runs against each mutant (a mutant
    is *killed* when the suite exits non-zero). The local distributor keeps the
    run in-process so it works inside a single throwaway container.
    """
    return (
        "[cosmic-ray]\n"
        f"module-path = {json.dumps(module_path)}\n"
        f"timeout = {float(timeout)}\n"
        "excluded-modules = []\n"
        f"test-command = {json.dumps(test_command)}\n"
        "\n"
        "[cosmic-ray.distributor]\n"
        'name = "local"\n'
    )


def cosmicray_run_commands() -> tuple[str, ...]:
    """The init + exec commands that populate the cosmic-ray session db."""
    cfg = shlex.quote(COSMIC_RAY_CONFIG)
    sess = shlex.quote(COSMIC_RAY_SESSION)
    return (
        f"cosmic-ray init {cfg} {sess}",
        f"cosmic-ray exec {cfg} {sess}",
    )


def cosmicray_report_command() -> str:
    return f"cr-report {shlex.quote(COSMIC_RAY_SESSION)}"


#: The scope-filter script the Python adapter writes into the sandbox.
COSMIC_RAY_SCOPE_SCRIPT = "swe_forge_cr_scope.py"


def cosmicray_scope_script(ranges: Sequence[tuple[int, int]]) -> str:
    """Return a self-contained script that scopes a cosmic-ray session to RANGES.

    ``cosmic-ray init`` enumerates a work item per mutable AST node in the WHOLE
    module, which on a large modular file (e.g. a boltons ``*utils.py``) is
    hundreds of mutants -- each run through the established suite -- and does not
    finish within ``mutation_timeout`` for a difficulty-amplifier
    (``bug_combination`` / ``multi_file``) candidate. Run AFTER ``cosmic-ray
    init`` and BEFORE ``cosmic-ray exec``, this filter keeps only the work items
    whose mutation falls inside one of the changed-symbol line ``ranges``
    (1-based, inclusive), bounding the run to the actually-mutated region -- the
    same file/region scoping the Go (file-scoped) and JS (mutate-list) tooling
    already do. The session path is ``argv[1]``.

    Safety: if filtering would empty the session (no mutable node lies in range),
    the original work items are kept untouched so the gate still measures the
    suite rather than spuriously rejecting on ``0 mutants`` -- scoping only ever
    *narrows* an over-large run, never fabricates a vacuous pass.
    """
    pairs = [(int(lo), int(hi)) for lo, hi in ranges]
    return (
        "import sys\n"
        "from cosmic_ray.work_db import WorkDB\n"
        f"RANGES = {pairs!r}\n"
        "db = WorkDB(sys.argv[1], WorkDB.Mode.open)\n"
        "try:\n"
        "    total = db.num_work_items\n"
        "    keep = [\n"
        "        wi\n"
        "        for wi in db.work_items\n"
        "        if any(\n"
        "            lo <= m.start_pos[0] <= hi\n"
        "            for m in wi.mutations\n"
        "            for (lo, hi) in RANGES\n"
        "        )\n"
        "    ]\n"
        "    if keep and len(keep) != total:\n"
        "        db.clear()\n"
        "        db.add_work_items(keep)\n"
        "    print('cr-scope kept', len(keep), 'of', total)\n"
        "finally:\n"
        "    db.close()\n"
    )


def cosmicray_scope_command() -> str:
    """Run the scope-filter script against the written session."""
    return f"python {shlex.quote(COSMIC_RAY_SCOPE_SCRIPT)} {shlex.quote(COSMIC_RAY_SESSION)}"


_CR_TOTAL_RE = re.compile(r"total jobs:\s*(\d+)")
_CR_SURVIVING_RE = re.compile(r"surviving mutants:\s*(\d+)")
_CR_SURVIVED_OUTCOME = "test outcome: TestOutcome.SURVIVED"


def parse_cosmicray_report(text: str) -> ToolCounts:
    """Parse ``cr-report`` output into :class:`ToolCounts`.

    The report lists each mutant (``<module> <operator> <occurrence>`` followed
    by a ``worker outcome: ..., test outcome: TestOutcome.<X>`` line) and ends
    with ``total jobs: N`` / ``surviving mutants: K (..)``. Survivor descriptions
    are the mutant header lines preceding a ``SURVIVED`` outcome.
    """
    lines = text.splitlines()
    total: int | None = None
    survived: int | None = None
    survivors: list[str] = []
    for idx, raw in enumerate(lines):
        line = raw.strip()
        total_match = _CR_TOTAL_RE.search(line)
        if total_match:
            total = int(total_match.group(1))
        surviving_match = _CR_SURVIVING_RE.search(line)
        if surviving_match:
            survived = int(surviving_match.group(1))
        if _CR_SURVIVED_OUTCOME in line:
            for back in range(idx - 1, -1, -1):
                prev = lines[back].strip()
                if (
                    prev
                    and not prev.startswith("[job-id]")
                    and not prev.startswith("worker outcome:")
                ):
                    survivors.append(prev)
                    break
    if total is None:
        raise MutationToolError(
            "could not parse cosmic-ray report (no 'total jobs:' line)"
        )
    if survived is None:
        survived = 0
    return ToolCounts(total=total, killed=total - survived, survivors=tuple(survivors))


# --------------------------------------------------------------------------- #
# JavaScript / TypeScript: Stryker
# --------------------------------------------------------------------------- #
STRYKER_CONFIG = "stryker.conf.json"
STRYKER_REPORT = "reports/mutation/mutation.json"

# Stryker's command test runner manages the test process tree with the system
# ``ps`` utility, which the official ``node:*-slim`` base images do NOT ship; its
# absence crashes Stryker with ``spawn ps ENOENT`` AFTER a successful dry run. We
# best-effort install ``procps`` (the same apt pattern the env builder uses for
# git) and verify ``ps`` is present so the gate fails with a clear reason rather
# than an opaque Node stack trace.
STRYKER_SETUP: tuple[str, ...] = (
    "set +e; export DEBIAN_FRONTEND=noninteractive; "
    "if ! command -v ps >/dev/null 2>&1; then "
    "apt-get update -qq >/dev/null 2>&1 && "
    "apt-get install -y -qq procps >/dev/null 2>&1; fi; "
    "command -v ps >/dev/null 2>&1",
)

# Stryker statuses that count as a kill / as a survivor / as out-of-scope.
_STRYKER_KILLED = {"Killed", "Timeout"}
_STRYKER_SURVIVED = {"Survived", "NoCoverage"}


def stryker_config(mutate: Sequence[str], *, test_command: str = "npm test") -> str:
    """Render a Stryker config that mutates ``mutate`` via the command runner.

    The ``command`` test runner just runs ``test_command`` and treats a non-zero
    exit as a kill, so it works with any JS/TS suite (``npm test`` / the
    repo-configured runner) without a Stryker framework plugin.
    """
    config = {
        "$schema": "./node_modules/@stryker-mutator/core/schema/stryker-schema.json",
        "testRunner": "command",
        "commandRunner": {"command": test_command},
        "reporters": ["json", "clear-text"],
        "mutate": list(mutate),
        "coverageAnalysis": "off",
        "concurrency": 2,
    }
    return json.dumps(config, indent=2)


def stryker_run_command() -> str:
    """Run Stryker via ``npx`` (fetched on demand) against the written config."""
    return f"npx --yes @stryker-mutator/core run {shlex.quote(STRYKER_CONFIG)}"


# npm/yarn/pnpm runners whose lifecycle hooks we must suppress.
_JS_PACKAGE_RUNNERS = ("npm", "yarn", "pnpm")


def robust_js_test_command(test_command: str = "npm test") -> str:
    """Make a JS test command safe to run as Stryker's command-runner target.

    Stryker copies an INSTRUMENTED copy of the mutate file into its sandbox (every
    mutant inlined behind a switch), so a babel/ESM repo whose ``test`` script runs
    a ``pretest`` lifecycle hook (e.g. validator.js's ``npm run build && npm run
    lint``) crashes Stryker's *dry run* with exit 1 BEFORE any test executes: the
    instrumented source trips eslint with hundreds of style errors. ``--ignore-
    scripts`` skips those pre/post hooks; the baseline build artifacts already
    persist in the ``EnvImage`` and the suite tests the babel-transformed ``src``,
    so no rebuild is needed and the per-mutant run is also faster (coverage tools
    such as nyc embedded in the ``test`` script keep working and pass their exit
    code through). Non-npm commands are returned unchanged.
    """
    cmd = test_command.strip()
    if not cmd:
        return cmd
    if cmd.split()[0] not in _JS_PACKAGE_RUNNERS or "--ignore-scripts" in cmd:
        return cmd
    # Place the flag before any ``--`` runner-argument separator so npm consumes it.
    head, sep, tail = cmd.partition(" -- ")
    if sep:
        return f"{head} --ignore-scripts{sep}{tail}"
    return f"{cmd} --ignore-scripts"


def parse_stryker_json(report_text: str) -> ToolCounts:
    """Parse a Stryker ``mutation.json`` report into :class:`ToolCounts`.

    Mutation score denominator excludes compile/ignored mutants, so ``total`` is
    only Killed/Timeout/Survived/NoCoverage; ``killed`` is Killed+Timeout.
    """
    try:
        data = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise MutationToolError(f"invalid Stryker JSON report: {exc}") from exc

    files = data.get("files")
    if not isinstance(files, dict):
        raise MutationToolError("Stryker report has no 'files' map")

    total = 0
    killed = 0
    survivors: list[str] = []
    for path, file_obj in files.items():
        if not isinstance(file_obj, dict):
            continue
        for mutant in file_obj.get("mutants", []) or []:
            if not isinstance(mutant, dict):
                continue
            status = str(mutant.get("status", ""))
            if status in _STRYKER_KILLED:
                total += 1
                killed += 1
            elif status in _STRYKER_SURVIVED:
                total += 1
                name = mutant.get("mutatorName", "mutant")
                loc = mutant.get("location", {})
                line = ""
                if isinstance(loc, dict):
                    start = loc.get("start", {})
                    if isinstance(start, dict):
                        line = f":{start.get('line', '?')}"
                survivors.append(f"{path}{line} {name} [{status}]")
    return ToolCounts(total=total, killed=killed, survivors=tuple(survivors))


# --------------------------------------------------------------------------- #
# Go: go-mutesting
# --------------------------------------------------------------------------- #
# go-mutesting's ``@latest`` tracks a fork whose newer commits raise the required
# Go version (e.g. >= 1.25.5) past the ``golang:1.22`` base image. The base image
# pins ``GOTOOLCHAIN=local``, so the install fails with a toolchain mismatch;
# ``GOTOOLCHAIN=auto`` lets ``go install`` download the toolchain the module needs
# (the Go module/toolchain registry is reachable) and keeps ``@latest`` working as
# the fork evolves. Target tests still run under the module's own ``go`` directive.
GO_MUTESTING_SETUP: tuple[str, ...] = (
    "GOTOOLCHAIN=auto go install "
    "github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest",
)


def gomutesting_command(target: str) -> str:
    """Run go-mutesting against one target (a Go FILE or a package directory).

    Scoping to the single gold *file* (rather than its whole package) is what
    keeps the run bounded and parseable: on a repo whose package has many sources
    (e.g. google/uuid's root package: 250+ mutants in ``uuid.go`` alone) a
    package-scoped run does not finish within the timeout, so its truncated output
    has no ``mutation score`` summary and looks like a tool crash. go-mutesting
    accepts a ``*.go`` file argument and still runs that file's package tests, so
    a file-scoped run completes and yields a real kill ratio. A package dir is
    given a ``./`` prefix; a file path is passed through as-is.
    """
    if not target.endswith(".go") and not target.startswith((".", "/")):
        target = f"./{target}"
    return f"go-mutesting {shlex.quote(target)}"


_GO_SCORE_RE = re.compile(
    r"^[ \t]*(?:the\s+)?mutation score is\s+[\d.]+\s*\("
    r"(\d+)\s+passed,\s*(\d+)\s+failed(?:,[^\r\n]*)?\)",
    re.IGNORECASE | re.MULTILINE,
)
# go-mutesting's own per-mutant surviving verdict lines are quoted. Anchoring on
# the quote avoids miscounting an interleaved ``go test`` ``FAIL\tpkg`` line.
_GO_FAIL_LINE_RE = re.compile(r'^\s*FAIL\s+"([^"]+)"', re.MULTILINE)
# Emitted (with a non-zero exit) when a target has no buildable Go source / nothing
# to mutate: a SOUND "0 eligible mutants" result, NOT a tool crash.
_GO_NO_MUTANTS_SIGNALS = (
    "could not find any suitable go source files",
    "found 0 mutations",
)


def parse_gomutesting(output: str, *, exit_code: int | None = None) -> ToolCounts:
    """Parse go-mutesting output into :class:`ToolCounts`.

    go-mutesting prints ``PASS`` when a mutant is caught (killed) and ``FAIL``
    when it survives, ending with ``The mutation score is X (P passed, F failed,
    ...)`` (``passed`` == killed, ``failed`` == survived). Trusted adequacy
    counts require a completed exit-zero run with exactly one terminal summary
    followed only by whitespace. A partial stream can omit remaining survivors,
    so PASS/FAIL lines alone are never admissible as a measurement, whether the
    process timed out, was signaled, or exited nonzero.

    The one exception is the tool's recognized "no suitable Go source" output:
    that is a sound zero-eligible-mutants result even though the tool may return
    nonzero. It becomes ``ToolCounts(0, 0)`` and the adequacy gate rejects it as
    under-determined rather than treating it as a tool crash.
    """
    if exit_code is None or exit_code < 0 or exit_code == 124 or exit_code >= 128:
        raise MutationToolError(
            "untrusted go-mutesting result: missing, timed-out, or signaled "
            f"execution (exit={exit_code}): {output.strip()[-400:]}"
        )

    lowered = output.lower()
    if any(signal in lowered for signal in _GO_NO_MUTANTS_SIGNALS):
        return ToolCounts(total=0, killed=0)

    summaries = tuple(_GO_SCORE_RE.finditer(output))
    if (
        exit_code == 0
        and len(summaries) == 1
        and not output[summaries[0].end() :].strip()
    ):
        match = summaries[0]
        survivors = tuple(m.group(1).strip() for m in _GO_FAIL_LINE_RE.finditer(output))
        killed = int(match.group(1))
        survived = int(match.group(2))
        return ToolCounts(total=killed + survived, killed=killed, survivors=survivors)

    raise MutationToolError(
        "untrusted go-mutesting result: require a completed exit-zero run with "
        f"a terminal 'mutation score' summary (exit={exit_code}): "
        f"{output.strip()[-400:]}"
    )


__all__ = [
    "COSMIC_RAY_CONFIG",
    "COSMIC_RAY_SCOPE_SCRIPT",
    "COSMIC_RAY_SESSION",
    "COSMIC_RAY_SETUP",
    "GO_MUTESTING_SETUP",
    "STRYKER_CONFIG",
    "STRYKER_REPORT",
    "STRYKER_SETUP",
    "MutationToolError",
    "ToolCounts",
    "cosmicray_config",
    "cosmicray_report_command",
    "cosmicray_run_commands",
    "cosmicray_scope_command",
    "cosmicray_scope_script",
    "gomutesting_command",
    "parse_cosmicray_report",
    "parse_gomutesting",
    "parse_stryker_json",
    "robust_js_test_command",
    "stryker_config",
    "stryker_run_command",
]
