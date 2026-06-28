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
GO_MUTESTING_SETUP: tuple[str, ...] = (
    "go install github.com/avito-tech/go-mutesting/cmd/go-mutesting@latest",
)


def gomutesting_command(package: str) -> str:
    """Run go-mutesting against one package directory (scoped to bound runtime)."""
    target = package if package.startswith((".", "/")) else f"./{package}"
    return f"go-mutesting {shlex.quote(target)}"


_GO_SCORE_RE = re.compile(
    r"mutation score is\s+[\d.]+\s*\((\d+)\s+passed,\s*(\d+)\s+failed", re.IGNORECASE
)
_GO_SURVIVOR_RE = re.compile(r"^FAIL\s+(.*)$")


def parse_gomutesting(output: str) -> ToolCounts:
    """Parse go-mutesting output into :class:`ToolCounts`.

    go-mutesting prints ``PASS`` when a mutant is caught (killed) and ``FAIL``
    when it survives, ending with ``The mutation score is X (P passed, F
    failed, ...)``; ``passed`` == killed, ``failed`` == survived.
    """
    match = _GO_SCORE_RE.search(output)
    if not match:
        raise MutationToolError(
            "could not parse go-mutesting output (no 'mutation score' summary)"
        )
    killed = int(match.group(1))
    survived = int(match.group(2))
    survivors = [
        m.group(1).strip()
        for m in (_GO_SURVIVOR_RE.match(line.strip()) for line in output.splitlines())
        if m
    ]
    return ToolCounts(
        total=killed + survived, killed=killed, survivors=tuple(survivors)
    )


__all__ = [
    "COSMIC_RAY_CONFIG",
    "COSMIC_RAY_SESSION",
    "COSMIC_RAY_SETUP",
    "GO_MUTESTING_SETUP",
    "STRYKER_CONFIG",
    "STRYKER_REPORT",
    "MutationToolError",
    "ToolCounts",
    "cosmicray_config",
    "cosmicray_report_command",
    "cosmicray_run_commands",
    "gomutesting_command",
    "parse_cosmicray_report",
    "parse_gomutesting",
    "parse_stryker_json",
    "stryker_config",
    "stryker_run_command",
]
