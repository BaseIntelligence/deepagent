"""Soft solvers for panel rollouts: apply candidate patch and score resolve.

Used by hardness panel to decide solve/not-solve without embedding gold.

Backends:
- local_pytest: host-side git-apply + pytest (fast fixture path)
- oracle_runner: Docker / FakeOracleRunner via harness.score_candidate
- text_patch_heuristic: offline fallback — never solves (fail-safe)
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_factory.openrouter import ChatResult
from swe_factory.panel.runner import SoftSolverFn
from swe_factory.schema import TaskRecord

_DIFF_HEADER_RE = re.compile(r"(?m)^(?:diff --git |--- |\+\+\+ |@@ )")


class SoftSolveError(RuntimeError):
    """Raised when soft scoring infra fails (scored as unsolved by callers)."""


def normalize_patch_paths(patch: str) -> str:
    """Rewrite ---/+++ /diff --git paths so repo-relative names may apply.

    Frontier models often invent prefixes like ``fixtures/tiny_green/`` or
    ``repo/``; strip common junk so git apply can target the broken workspace.
    """
    if not patch:
        return patch
    lines: list[str] = []
    junk_prefixes = (
        "fixtures/tiny_green/",
        "fixtures/tiny_offline/",
        "fixtures/",
        "repo/",
        "src/",
        "./",
    )

    def _clean_path(path: str) -> str:
        cleaned = path.strip()
        # strip a/ or b/ already handled by callers where needed
        for prefix in junk_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        # drop absolute-looking leading /
        cleaned = cleaned.lstrip("/")
        return cleaned

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            # diff --git a/FOO b/BAR
            parts = line.split()
            if len(parts) >= 4:
                a = parts[2]
                b = parts[3]
                a_body = a[2:] if a.startswith("a/") else a
                b_body = b[2:] if b.startswith("b/") else b
                a_body = _clean_path(a_body)
                b_body = _clean_path(b_body)
                line = f"diff --git a/{a_body} b/{b_body}"
        elif line.startswith("--- ") or line.startswith("+++ "):
            prefix = line[:4]
            rest = line[4:]
            if rest in {"/dev/null", "dev/null"}:
                lines.append(line)
                continue
            # optional a/ or b/
            side = ""
            body = rest
            if rest.startswith("a/") or rest.startswith("b/"):
                side = rest[:2]
                body = rest[2:]
            # strip timestamps tab-separated
            if "\t" in body:
                body = body.split("\t", 1)[0]
            body = _clean_path(body)
            line = f"{prefix}{side}{body}" if side else f"{prefix}{body}"
            # ensure a/ b/ present for standard git apply
            if (
                prefix == "--- "
                and not line.startswith("--- a/")
                and line
                not in {
                    "--- /dev/null",
                    "--- dev/null",
                }
            ):
                line = f"--- a/{_clean_path(rest if not side else body)}"
            if (
                prefix == "+++ "
                and not line.startswith("+++ b/")
                and line
                not in {
                    "+++ /dev/null",
                    "+++ dev/null",
                }
            ):
                line = f"+++ b/{_clean_path(rest if not side else body)}"
        lines.append(line)
    out = "\n".join(lines)
    return out if out.endswith("\n") else out + "\n"


def extract_unified_diff(text: str) -> str:
    """Best-effort extract of a unified diff from model text."""
    if not text or not text.strip():
        return ""
    raw = text.strip()
    candidate = ""
    # Prefer fenced ```diff blocks
    fence = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        body = fence.group(1).strip()
        if _DIFF_HEADER_RE.search(body) or body.startswith("--- ") or body.startswith("diff "):
            candidate = body if body.endswith("\n") else body + "\n"
    if not candidate:
        # Strip leading prose before first diff header
        for marker in ("diff --git ", "--- a/", "--- /", "--- fixtures/", "--- repo/"):
            idx = raw.find(marker)
            if idx >= 0:
                body = raw[idx:].strip()
                # trim trailing fencing residue
                if "```" in body:
                    body = body.split("```", 1)[0].rstrip()
                candidate = body if body.endswith("\n") else body + "\n"
                break
    if not candidate and _DIFF_HEADER_RE.search(raw):
        candidate = raw if raw.endswith("\n") else raw + "\n"
    if not candidate:
        return ""
    return normalize_patch_paths(candidate)


def _rewrite_hunk_headers(patch: str) -> str:
    """Fill missing @@ line numbers so git apply can parse model patches."""
    lines = patch.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            # Count subsequent hunk body until next header.
            body: list[str] = []
            j = i + 1
            while j < len(lines) and not (
                lines[j].startswith("@@")
                or lines[j].startswith("diff --git")
                or lines[j].startswith("--- ")
                or lines[j].startswith("+++ ")
            ):
                body.append(lines[j])
                j += 1
            old_count = 0
            new_count = 0
            for b in body:
                if b.startswith("\\"):  # "\ No newline"
                    continue
                if b.startswith("+") and not b.startswith("+++"):
                    new_count += 1
                elif b.startswith("-") and not b.startswith("---"):
                    old_count += 1
                else:
                    old_count += 1
                    new_count += 1
            old_count = max(old_count, 1)
            new_count = max(new_count, 1)
            # default start at 1 if model omitted numbers
            m = re.match(r"@@\s*-?(\d+)?(?:,\d+)?\s*\+?(\d+)?(?:,\d+)?\s*@@(.*)$", line)
            old_start = 1
            new_start = 1
            trail = ""
            if m:
                if m.group(1):
                    old_start = int(m.group(1))
                if m.group(2):
                    new_start = int(m.group(2))
                trail = m.group(3) or ""
            out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{trail}")
            out.extend(body)
            i = j
            continue
        out.append(line)
        i += 1
    rewritten = "\n".join(out)
    return rewritten if rewritten.endswith("\n") else rewritten + "\n"


def _function_body_replace_apply(workspace: Path, patch: str) -> bool:
    """Fallback: replace entire functions present as -raise NotImplemented / +new body."""
    # crude: look for `def name` in +/ - lines
    changed = False
    # Map path -> sequence of intended new file content is hard; instead apply
    # per-file unified by reconstructing from markers.
    current_path: str | None = None
    chunks: dict[str, list[tuple[str, list[str], list[str]]]] = {}
    old_buf: list[str] = []
    new_buf: list[str] = []
    for raw in patch.splitlines():
        if raw.startswith("--- "):
            if current_path and (old_buf or new_buf):
                chunks.setdefault(current_path, []).append(("hunk", old_buf, new_buf))
            old_buf, new_buf = [], []
            rest = raw[4:]
            if rest.startswith("a/"):
                rest = rest[2:]
            current_path = rest.split("\t")[0]
            continue
        if raw.startswith("+++ "):
            continue
        if raw.startswith("@@"):
            if current_path and (old_buf or new_buf):
                chunks.setdefault(current_path, []).append(("hunk", old_buf, new_buf))
            old_buf, new_buf = [], []
            continue
        if current_path is None:
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            new_buf.append(raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
            old_buf.append(raw[1:])
        elif raw.startswith(" "):
            old_buf.append(raw[1:])
            new_buf.append(raw[1:])
    if current_path and (old_buf or new_buf):
        chunks.setdefault(current_path, []).append(("hunk", old_buf, new_buf))

    for rel, hunks in chunks.items():
        path = workspace / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for _kind, old_lines, new_lines in hunks:
            if not old_lines or not new_lines:
                continue
            old_block = "\n".join(old_lines)
            new_block = "\n".join(new_lines)
            if old_block in text:
                text = text.replace(old_block, new_block, 1)
                changed = True
                continue
            # function-level: find def name from old or new and replace whole def body
            m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", "\n".join(old_lines + new_lines))
            if not m:
                continue
            name = m.group(1)
            pat = re.compile(
                rf"(^[ \t]*def\s+{re.escape(name)}\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\n)"
                rf"((?:(?:[ \t]+|\s*#).*\n|\s*\n)*)",
                re.MULTILINE,
            )
            mm = pat.search(text)
            if not mm:
                continue
            # Rebuild function from new_lines if they include def header, else body only.
            if any(ln.strip().startswith(f"def {name}") for ln in new_lines):
                # use full new_lines joined as function
                # find indent of def
                new_fn = "\n".join(new_lines)
                if not new_fn.endswith("\n"):
                    new_fn += "\n"
                text = text[: mm.start()] + new_fn + text[mm.end() :]
            else:
                # new_lines are body lines; preserve def header
                # ensure body indent
                indent = "    "
                body_lines = []
                for ln in new_lines:
                    if ln.strip() == "":
                        body_lines.append("")
                    elif ln.startswith(" ") or ln.startswith("\t"):
                        body_lines.append(ln)
                    else:
                        body_lines.append(indent + ln)
                body = "\n".join(body_lines) + "\n"
                text = text[: mm.start(2)] + body + text[mm.end(2) :]
            changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
    return changed


def _git_apply(workspace: Path, patch: str) -> bool:
    if not patch.strip():
        return False
    patch = _rewrite_hunk_headers(normalize_patch_paths(patch))
    if not (workspace / ".git").exists():
        subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=False)
        for key, value in (("user.email", "panel@localhost"), ("user.name", "panel")):
            subprocess.run(
                ["git", "config", key, value],
                cwd=workspace,
                capture_output=True,
                check=False,
            )
        subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, check=False)
        subprocess.run(
            ["git", "commit", "-m", "broken", "--allow-empty"],
            cwd=workspace,
            capture_output=True,
            check=False,
        )
    patch_path = workspace / ".panel_candidate.patch"
    patch_path.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")
    try:
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if applied.returncode == 0:
            return True
        # retry with 3-way
        applied = subprocess.run(
            ["git", "apply", "--3way", "--whitespace=nowarn", str(patch_path)],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        if applied.returncode == 0:
            return True
        # fallback for poorly numbered hunks
        return _function_body_replace_apply(workspace, patch)
    finally:
        patch_path.unlink(missing_ok=True)


def _run_commands(workspace: Path, commands: Sequence[str], *, timeout: float = 120.0) -> bool:
    if not commands:
        return True
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return False
    return True


def local_pytest_soft_solver(
    *,
    broken_workspace: Path,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str] = (),
    install_commands: Sequence[str] = (),
) -> SoftSolverFn:
    """Host-side soft solver for small fixtures (no Docker).

    Copies broken tree, applies extracted patch, install (optional), then F2P+P2P.
    Resolve metric matches harness: all F2P pass AND all P2P pass.
    """
    broken = Path(broken_workspace)
    if not broken.is_dir():
        raise SoftSolveError(f"broken_workspace missing: {broken}")

    def solver(
        model: str,
        messages: Sequence[dict[str, str]],
        chat: ChatResult | None,
    ) -> bool:
        del model, messages
        if chat is None or not (chat.text or "").strip():
            return False
        patch = extract_unified_diff(chat.text)
        if not patch.strip():
            return False
        with tempfile.TemporaryDirectory(prefix="sdf-soft-") as tmp:
            work = Path(tmp) / "repo"
            shutil.copytree(
                broken,
                work,
                ignore=shutil.ignore_patterns(
                    ".git",
                    "__pycache__",
                    "*.pyc",
                    ".venv",
                    "node_modules",
                    ".pytest_cache",
                ),
            )
            if not _git_apply(work, patch):
                return False
            # Best-effort install: host pytest stacks often already provide deps.
            # Fail only if subsequent tests fail — not solely on install stderr.
            if install_commands:
                try:
                    _run_commands(work, install_commands, timeout=180.0)
                except subprocess.TimeoutExpired:
                    return False
            try:
                f2p_ok = _run_commands(work, fail_to_pass, timeout=120.0)
                if not f2p_ok:
                    return False
                if pass_to_pass:
                    return _run_commands(work, pass_to_pass, timeout=120.0)
                return True
            except subprocess.TimeoutExpired:
                return False

    return solver


def oracle_runner_soft_solver(
    *,
    task: TaskRecord,
    broken_workspace: Path,
    runner: Any,
) -> SoftSolverFn:
    """Soft solver that uses OracleRunnerBackend via harness.score_candidate."""
    from swe_factory.harness.score import HarnessError, score_candidate

    broken = Path(broken_workspace)

    def solver(
        model: str,
        messages: Sequence[dict[str, str]],
        chat: ChatResult | None,
    ) -> bool:
        del model, messages
        if chat is None or not (chat.text or "").strip():
            return False
        patch = extract_unified_diff(chat.text)
        if not patch.strip():
            return False
        try:
            result = score_candidate(
                task=task,
                workspace=broken,
                patch=patch,
                runner=runner,
                label="panel-candidate",
            )
            return bool(result.resolve)
        except (HarnessError, SoftSolveError, Exception):  # noqa: BLE001 — unsolved on infra error
            return False

    return solver


def never_solve_soft_solver() -> SoftSolverFn:
    """Fail-safe soft solver (never marks a rollout solved)."""

    def solver(
        model: str,
        messages: Sequence[dict[str, str]],
        chat: ChatResult | None,
    ) -> bool:
        del model, messages, chat
        return False

    return solver


@dataclass(frozen=True, slots=True)
class SoftSolverMeta:
    backend: str
    workspaces: str


__all__ = [
    "SoftSolveError",
    "SoftSolverMeta",
    "extract_unified_diff",
    "local_pytest_soft_solver",
    "never_solve_soft_solver",
    "normalize_patch_paths",
    "oracle_runner_soft_solver",
]
