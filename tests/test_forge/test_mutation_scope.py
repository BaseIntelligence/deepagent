"""Offline tests for amplifier mutation-run scoping (m6-pilot-difficulty).

The difficulty-amplifier generators (``bug_combination`` / ``multi_file``) mutate
>=2 symbols on large MODULAR modules; mutating each WHOLE module is hundreds of
mutants and does not finish within ``mutation_timeout``. The mutation runner
therefore scopes cosmic-ray to the changed-symbol line ranges. This module covers,
with no Docker and no live LLM:

- ``patch_old_side_regions`` parses GOLD-side hunk line ranges from a unified diff;
- ``candidate_mutation_regions`` unions provenance symbol spans + patch hunks and
  is only derived for amplifier generators;
- the cosmic-ray scope script REALLY narrows a session (driven against a real
  host cosmic-ray ``WorkDB``), and falls back to keeping all items when nothing is
  in range (never a spurious 0-mutant reject);
- the Python adapter writes+runs the scope step between ``init`` and ``exec`` when
  regions are given, and skips it otherwise;
- the amplifier generators record each fault's enclosing-symbol span in provenance.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from swe_forge.forge.adapters import PythonAdapter
from swe_forge.forge.adapters._mutation_tools import (
    COSMIC_RAY_SCOPE_SCRIPT,
    cosmicray_scope_command,
    cosmicray_scope_script,
)
from swe_forge.forge.models import (
    Candidate,
    CandidateTarget,
    Provenance,
)
from swe_forge.forge.oracle.mutation import (
    AMPLIFIER_GENERATORS,
    candidate_mutation_regions,
    patch_old_side_regions,
)


def _candidate(
    generator: str,
    *,
    mutation_patch: str = "placeholder-no-hunks",
    details: dict | None = None,
) -> Candidate:
    return Candidate(
        language="python",
        generator=generator,
        target=CandidateTarget(files=("pkg/a.py", "pkg/b.py"), symbols=("f", "g")),
        mutation_patch=mutation_patch,
        oracle_patch="oracle",
        difficulty_hint="high",
        provenance=Provenance(
            generator=generator,
            seed=0,
            language="python",
            details=details or {},
        ),
    )


_TWO_FILE_PATCH = (
    "diff --git a/pkg/a.py b/pkg/a.py\n"
    "--- a/pkg/a.py\n"
    "+++ b/pkg/a.py\n"
    "@@ -10,3 +10,3 @@ def f():\n"
    "-    return x + 1\n"
    "+    return x - 1\n"
    "diff --git a/pkg/b.py b/pkg/b.py\n"
    "--- a/pkg/b.py\n"
    "+++ b/pkg/b.py\n"
    "@@ -40,2 +40,2 @@ def g():\n"
    "-    return y\n"
    "+    return z\n"
)


# --------------------------------------------------------------------------- #
# patch_old_side_regions
# --------------------------------------------------------------------------- #
class TestPatchOldSideRegions:
    def test_multi_file_multi_hunk_old_side_ranges(self) -> None:
        regions = patch_old_side_regions(_TWO_FILE_PATCH)
        assert regions == {"pkg/a.py": [(10, 12)], "pkg/b.py": [(40, 41)]}

    def test_single_line_hunk_count_defaults_to_one(self) -> None:
        patch = "--- a/m.py\n+++ b/m.py\n@@ -5 +5 @@\n-    a = 1\n+    a = 2\n"
        assert patch_old_side_regions(patch) == {"m.py": [(5, 5)]}

    def test_pure_insertion_hunk_contributes_no_range(self) -> None:
        # old_count == 0 -> nothing on the gold side to mutate.
        patch = "--- a/m.py\n+++ b/m.py\n@@ -0,0 +1,2 @@\n+line1\n+line2\n"
        assert patch_old_side_regions(patch) == {}

    def test_dev_null_target_is_skipped(self) -> None:
        patch = "--- a/m.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-line1\n-line2\n"
        assert patch_old_side_regions(patch) == {}

    def test_empty_patch_is_empty(self) -> None:
        assert patch_old_side_regions("") == {}


# --------------------------------------------------------------------------- #
# candidate_mutation_regions
# --------------------------------------------------------------------------- #
class TestCandidateMutationRegions:
    def test_unions_provenance_spans_with_patch_hunks(self) -> None:
        details = {
            "faults": [
                {"file": "pkg/a.py", "symbol": "f", "start_line": 8, "end_line": 14},
                {"file": "pkg/b.py", "symbol": "g", "start_line": 38, "end_line": 45},
            ]
        }
        cand = _candidate(
            "bug_combination", mutation_patch=_TWO_FILE_PATCH, details=details
        )
        regions = candidate_mutation_regions(cand)
        # function spans (provenance) PLUS the diff-line hunks (patch), unioned.
        assert (8, 14) in regions["pkg/a.py"] and (10, 12) in regions["pkg/a.py"]
        assert (38, 45) in regions["pkg/b.py"] and (40, 41) in regions["pkg/b.py"]

    def test_multi_file_edits_records_are_read(self) -> None:
        details = {
            "edits": [
                {"file": "pkg/a.py", "symbol": "f", "start_line": 8, "end_line": 14},
            ]
        }
        cand = _candidate("multi_file", details=details)
        assert candidate_mutation_regions(cand) == {"pkg/a.py": ((8, 14),)}

    def test_falls_back_to_patch_when_no_provenance_spans(self) -> None:
        cand = _candidate("bug_combination", mutation_patch=_TWO_FILE_PATCH)
        assert candidate_mutation_regions(cand) == {
            "pkg/a.py": ((10, 12),),
            "pkg/b.py": ((40, 41),),
        }

    def test_empty_when_nothing_derivable(self) -> None:
        assert candidate_mutation_regions(_candidate("bug_combination")) == {}

    def test_malformed_records_are_ignored(self) -> None:
        details = {
            "faults": [
                {"file": "pkg/a.py", "start_line": "x", "end_line": 4},  # bad type
                {"start_line": 1, "end_line": 4},  # no file
                {"file": "pkg/a.py", "start_line": 9, "end_line": 3},  # end < start
            ]
        }
        cand = _candidate("bug_combination", details=details)
        assert candidate_mutation_regions(cand) == {}

    def test_amplifier_set_is_exactly_the_two_amplifiers(self) -> None:
        assert AMPLIFIER_GENERATORS == frozenset({"bug_combination", "multi_file"})


# --------------------------------------------------------------------------- #
# cosmicray_scope_script against a REAL host cosmic-ray WorkDB
# --------------------------------------------------------------------------- #
def _build_session(path: Path, rows: list[int]) -> None:
    from cosmic_ray.work_db import WorkDB
    from cosmic_ray.work_item import MutationSpec, WorkItem

    db = WorkDB(str(path), WorkDB.Mode.create)
    try:
        items = []
        for index, row in enumerate(rows):
            spec = MutationSpec(
                module_path="m.py",
                operator_name="core/NumberReplacer",
                occurrence=0,
                start_pos=(row, 0),
                end_pos=(row, 5),
            )
            items.append(WorkItem(job_id=f"job-{index}", mutations=(spec,)))
        db.add_work_items(items)
    finally:
        db.close()


def _surviving_rows(path: Path) -> list[int]:
    from cosmic_ray.work_db import WorkDB

    db = WorkDB(str(path), WorkDB.Mode.open)
    try:
        return sorted(m.start_pos[0] for wi in db.work_items for m in wi.mutations)
    finally:
        db.close()


def _run_scope(session: Path, script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), str(session)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


class TestCosmicrayScopeScriptReal:
    def test_scope_narrows_session_to_ranges(self, tmp_path: Path) -> None:
        session = tmp_path / "cr.sqlite"
        _build_session(session, rows=[5, 11, 12, 40, 99])
        script = tmp_path / COSMIC_RAY_SCOPE_SCRIPT
        script.write_text(cosmicray_scope_script([(10, 12), (40, 41)]))
        _run_scope(session, script)
        # only rows in [10,12] or [40,41] remain.
        assert _surviving_rows(session) == [11, 12, 40]

    def test_scope_keeps_all_when_nothing_in_range(self, tmp_path: Path) -> None:
        # No mutable node in range must NOT empty the session (avoids a spurious
        # 0-mutant reject); the original items are kept untouched.
        session = tmp_path / "cr.sqlite"
        _build_session(session, rows=[5, 6, 7])
        script = tmp_path / COSMIC_RAY_SCOPE_SCRIPT
        script.write_text(cosmicray_scope_script([(100, 200)]))
        _run_scope(session, script)
        assert _surviving_rows(session) == [5, 6, 7]


# --------------------------------------------------------------------------- #
# Python adapter mutation_tool_run: scope step ordering
# --------------------------------------------------------------------------- #
@dataclass
class _FakeExec:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


_CR_REPORT = (
    "job-0 core/NumberReplacer 0\n"
    "worker outcome: normal, test outcome: TestOutcome.KILLED\n"
    "total jobs: 2\n"
    "complete: 2\n"
    "surviving mutants: 0 (0.00%)\n"
)


class _ScriptedPyExecutor:
    """Scriptable executor recording commands + files for cosmic-ray runs."""

    workspace_dir = "/workspace/repo"

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: list[str] = []

    async def run_command(self, cmd, *, cwd=None, timeout=None, env=None):  # type: ignore[no-untyped-def]
        self.commands.append(cmd)
        if "cr-report" in cmd:
            return _FakeExec(0, _CR_REPORT, "")
        return _FakeExec(0, "", "")

    async def write_file(self, path: str, content: str) -> None:
        self.files.append(path)

    async def read_file(self, path: str) -> str:
        return ""


class TestPythonAdapterScopeOrdering:
    async def test_scope_runs_between_init_and_exec_for_regions(self) -> None:
        ex = _ScriptedPyExecutor()
        await PythonAdapter().mutation_tool_run(
            ex,
            target_files=("pkg/a.py",),
            timeout=30.0,
            target_regions={"pkg/a.py": ((10, 12),)},
        )
        assert COSMIC_RAY_SCOPE_SCRIPT in ex.files
        scope = cosmicray_scope_command()
        init_idx = next(i for i, c in enumerate(ex.commands) if "cosmic-ray init" in c)
        scope_idx = next(i for i, c in enumerate(ex.commands) if c == scope)
        exec_idx = next(i for i, c in enumerate(ex.commands) if "cosmic-ray exec" in c)
        assert init_idx < scope_idx < exec_idx

    async def test_no_scope_when_regions_absent(self) -> None:
        ex = _ScriptedPyExecutor()
        await PythonAdapter().mutation_tool_run(
            ex, target_files=("pkg/a.py",), timeout=30.0
        )
        assert COSMIC_RAY_SCOPE_SCRIPT not in ex.files
        assert not any(c == cosmicray_scope_command() for c in ex.commands)

    async def test_no_scope_for_file_without_region_entry(self) -> None:
        ex = _ScriptedPyExecutor()
        await PythonAdapter().mutation_tool_run(
            ex,
            target_files=("pkg/a.py",),
            timeout=30.0,
            target_regions={"pkg/other.py": ((1, 2),)},
        )
        assert COSMIC_RAY_SCOPE_SCRIPT not in ex.files
