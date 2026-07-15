"""Docker evaluation backend for mechanical oracle gates (G1–G3 + flake).

Owns only ``sdf-*`` containers; always tears them down. Off-limits containers are
never named or removed. Reuses the envbuild DockerCLI seam so unit tests inject
a FakeDocker without a daemon.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from swe_factory.envbuild.builder import (
    DockerBackend,
    DockerCLI,
    EnvBuildError,
    image_tag_for,
    remove_leftover_sdf_containers,
    scoped_container_name,
)

_CONTAINER_PREFIX = "sdf-"
_DEFAULT_WORKSPACE = "/workspace/repo"
_TIMEOUT_EXIT = 124


class OracleDockerError(RuntimeError):
    """Unrecoverable docker evaluation failure."""


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Result of one test command inside a container."""

    command: str
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def timed_out(self) -> bool:
        return self.exit_code == _TIMEOUT_EXIT


@dataclass(frozen=True, slots=True)
class SuiteOutcome:
    """Aggregated F2P/P2P results for one evaluation phase."""

    phase: str
    f2p: tuple[CommandOutcome, ...]
    p2p: tuple[CommandOutcome, ...] = ()
    patch_applied: bool = False
    container_name: str = ""
    logs: dict[str, str] = field(default_factory=dict)

    def all_f2p_failed(self) -> bool:
        return bool(self.f2p) and all(not item.passed for item in self.f2p)

    def all_f2p_passed(self) -> bool:
        return bool(self.f2p) and all(item.passed for item in self.f2p)

    def all_p2p_passed(self) -> bool:
        if not self.p2p:
            return True
        return all(item.passed for item in self.p2p)

    def resolve(self) -> bool:
        """Strict resolve: every F2P pass AND every P2P pass."""
        return self.all_f2p_passed() and self.all_p2p_passed()

    def outcome_signature(self) -> tuple[bool, bool, tuple[int, ...], tuple[int, ...]]:
        """Deterministic signature for flake comparison."""
        return (
            self.all_f2p_passed(),
            self.all_p2p_passed(),
            tuple(c.exit_code for c in self.f2p),
            tuple(c.exit_code for c in self.p2p),
        )


class OracleRunnerBackend(Protocol):
    """Injectable evaluation backend (Docker or fake)."""

    def run_broken(
        self,
        *,
        workspace: Path,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str] = (),
    ) -> SuiteOutcome: ...

    def run_with_patch(
        self,
        *,
        workspace: Path,
        patch: str,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str] = (),
        phase: str = "gold",
    ) -> SuiteOutcome: ...

    def cleanup(self) -> None: ...


def _tail(text: str, *, max_lines: int = 30, max_chars: int = 3000) -> str:
    trimmed = text.strip()
    if not trimmed:
        return ""
    out = "\n".join(trimmed.splitlines()[-max_lines:])
    return out[-max_chars:]


def _copy_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", "node_modules"),
    )


def _init_git(path: Path) -> None:
    import subprocess

    if (path / ".git").exists():
        return
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=False)
    subprocess.run(
        ["git", "config", "user.email", "oracle@localhost"],
        cwd=path,
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ["git", "config", "user.name", "oracle"],
        cwd=path,
        capture_output=True,
        check=False,
    )
    subprocess.run(["git", "add", "-A"], cwd=path, capture_output=True, check=False)
    subprocess.run(
        ["git", "commit", "-m", "oracle broken baseline"],
        cwd=path,
        capture_output=True,
        check=False,
    )


_PREP = (
    "set +e; export DEBIAN_FRONTEND=noninteractive; "
    "if ! command -v git >/dev/null 2>&1; then "
    "apt-get update -qq >/dev/null 2>&1 && "
    "apt-get install -y -qq git >/dev/null 2>&1; fi; "
    "git config --global --add safe.directory '*' >/dev/null 2>&1 || true; "
    "true"
)


@dataclass
class OracleDockerRunner:
    """Evaluate FAIL→PASS phases inside throwaway ``sdf-*`` containers.

    When ``prebuilt_image`` is set, still copies the (broken) workspace into a
    fresh container so the golden baseline image is not mutated permanently.
    When unset, builds a short-lived eval image: pull base → install → commit
    (no baseline-green requirement; oracle G1 covers the broken suite).
    """

    docker: DockerBackend | None = None
    base_image: str = "python:3.12-slim"
    install_commands: list[str] = field(default_factory=lambda: ["pip install -q pytest"])
    workspace_dir: str = _DEFAULT_WORKSPACE
    prebuilt_image: str | None = None
    run_id: str | None = None
    memory_mb: int = 1024
    cpus: float = 1.0
    pids_limit: int = 512
    prep_timeout: float = 300.0
    install_timeout: float = 600.0
    command_timeout: float = 300.0
    cleanup_containers: bool = True
    keep_eval_image: bool = False

    def __post_init__(self) -> None:
        self._docker: DockerBackend = self.docker or DockerCLI()
        self._run_id = self.run_id or uuid.uuid4().hex[:8]
        self._owned_containers: list[str] = []
        self._eval_image: str | None = self.prebuilt_image
        self._built_eval_image: str | None = None
        self._install_done_images: set[str] = set()

    def _track(self, name: str) -> str:
        self._owned_containers.append(name)
        return name

    def _unique(self, role: str) -> str:
        return scoped_container_name(role, self._run_id)

    def _teardown(self) -> None:
        if not self.cleanup_containers:
            return
        for name in reversed(list(self._owned_containers)):
            if name.startswith(_CONTAINER_PREFIX):
                with contextlib.suppress(EnvBuildError):
                    self._docker.remove_container(name)
        self._owned_containers.clear()
        with contextlib.suppress(Exception):
            remove_leftover_sdf_containers(self._docker)
        if (
            not self.keep_eval_image
            and self._built_eval_image is not None
            and self.prebuilt_image is None
        ):
            with contextlib.suppress(EnvBuildError):
                self._docker.remove_image(self._built_eval_image)
            self._built_eval_image = None
            self._eval_image = None

    def cleanup(self) -> None:
        self._teardown()

    def _workspace_local_install_cmds(self) -> list[str]:
        """Commands that need a staged workspace (e.g. pip -e .)."""
        local: list[str] = []
        for cmd in self.install_commands:
            low = cmd.lower()
            pip_local = "pip install" in low and (
                "-e " in low or "--editable" in low or " ." in low
            )
            if pip_local or "npm install" in low or "go rebuild" in low:
                local.append(cmd)
        return local

    def _base_only_install_cmds(self) -> list[str]:
        """Install commands that can run on an empty base image (wheels only)."""
        local = set(self._workspace_local_install_cmds())
        base = [c for c in self.install_commands if c not in local]
        # Always provide pytest for python suites if nothing else declares it.
        if not any("pytest" in c for c in base + list(local)):
            base = [*base, "pip install -q pytest"]
        return base

    def _ensure_eval_image(self) -> str:
        if self._eval_image:
            return self._eval_image
        # Build install-only image (global deps; workspace-local installs deferred).
        tag = image_tag_for("oracle-eval", self._run_id, namespace="sdf-oracle")
        cname = self._track(self._unique("orac-build"))
        try:
            self._docker.ensure_image(self.base_image, timeout=900.0)
            self._docker.run_detached(
                name=cname,
                image=self.base_image,
                workdir=self.workspace_dir,
                memory_mb=self.memory_mb,
                cpus=self.cpus,
                pids_limit=self.pids_limit,
            )
            prep = self._docker.exec(
                cname, _PREP, workdir=self.workspace_dir, timeout=self.prep_timeout
            )
            if prep.exit_code != 0 and "apt-get" not in prep.combined:
                # Best-effort prep; continue if package manager unavailable
                pass
            base_cmds = self._base_only_install_cmds()
            if base_cmds:
                install_script = "set -e\n" + "\n".join(base_cmds)
                install = self._docker.exec(
                    cname,
                    install_script,
                    workdir=self.workspace_dir,
                    timeout=self.install_timeout,
                )
                if install.exit_code != 0:
                    raise OracleDockerError(
                        f"oracle install failed (exit {install.exit_code}): "
                        f"{_tail(install.combined)}"
                    )
            self._docker.commit(cname, tag, workdir=self.workspace_dir)
            self._eval_image = tag
            self._built_eval_image = tag
            # Workspace-local install still required per run when declared.
            if not self._workspace_local_install_cmds():
                self._install_done_images.add(tag)
            return tag
        finally:
            with contextlib.suppress(EnvBuildError):
                self._docker.remove_container(cname)

    def _run_phase(
        self,
        *,
        workspace: Path,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str],
        patch: str | None,
        phase: str,
    ) -> SuiteOutcome:
        if not workspace.is_dir():
            raise OracleDockerError(f"workspace not found: {workspace}")

        image = self._ensure_eval_image()
        cname = self._track(self._unique(phase))
        logs: dict[str, str] = {}
        tmp: Path | None = None
        patch_applied = False
        f2p_out: list[CommandOutcome] = []
        p2p_out: list[CommandOutcome] = []

        try:
            self._docker.run_detached(
                name=cname,
                image=image,
                workdir=self.workspace_dir,
                memory_mb=self.memory_mb,
                cpus=self.cpus,
                pids_limit=self.pids_limit,
            )
            # Stage a clean workspace tree with git so gold can apply.
            tmp = Path(tempfile.mkdtemp(prefix="sdf-oracle-ws-"))
            _copy_tree(workspace, tmp)
            _init_git(tmp)
            if patch is not None and patch.strip():
                patch_path = tmp / ".oracle_candidate.patch"
                body = patch if patch.endswith("\n") else patch + "\n"
                patch_path.write_text(body, encoding="utf-8")
            self._docker.copy_into(cname, tmp, self.workspace_dir)

            prep = self._docker.exec(
                cname, _PREP, workdir=self.workspace_dir, timeout=self.prep_timeout
            )
            logs["prep"] = _tail(prep.combined)

            # Install into this container if image is prebuilt base without deps
            if self.prebuilt_image and image == self.prebuilt_image:
                # Prebuilt green envbuild images already have deps.
                self._install_done_images.add(image)
            elif image not in self._install_done_images:
                # Run workspace-local installs (pip -e .) now that the tree is staged.
                local_cmds = self._workspace_local_install_cmds()
                # Also re-run full install_commands when they were entirely workspace-local.
                cmds = local_cmds or list(self.install_commands)
                if cmds:
                    # Help pure-source packages import without full packaging drop-ins.
                    env_prefix = 'export PYTHONPATH="${PYTHONPATH:-}:' + self.workspace_dir + '"; '
                    install_script = "set -e\n" + env_prefix + "\n".join(cmds)
                    install = self._docker.exec(
                        cname,
                        install_script,
                        workdir=self.workspace_dir,
                        timeout=self.install_timeout,
                    )
                    logs["install"] = _tail(install.combined)
                    if install.exit_code != 0:
                        raise OracleDockerError(
                            f"oracle workspace install failed (exit {install.exit_code}): "
                            f"{_tail(install.combined)}"
                        )
                self._install_done_images.add(image)

            if patch is not None and patch.strip():
                apply_script = (
                    "set -e; "
                    "if [ -f .oracle_candidate.patch ]; then "
                    "git apply --whitespace=nowarn .oracle_candidate.patch "
                    "|| patch -p1 --forward < .oracle_candidate.patch; "
                    "fi"
                )
                applied = self._docker.exec(
                    cname, apply_script, workdir=self.workspace_dir, timeout=60.0
                )
                logs["apply"] = _tail(applied.combined)
                if applied.exit_code != 0:
                    raise OracleDockerError(
                        f"failed to apply patch in phase {phase!r}: {_tail(applied.combined)}"
                    )
                patch_applied = True

            # Always ensure repo is importable for pure-source layouts.
            path_export = f'export PYTHONPATH="${{PYTHONPATH:-}}:{self.workspace_dir}"; '
            for cmd in fail_to_pass:
                outcome = self._docker.exec(
                    cname,
                    path_export + cmd,
                    workdir=self.workspace_dir,
                    timeout=self.command_timeout,
                )
                f2p_out.append(
                    CommandOutcome(
                        command=cmd,
                        exit_code=outcome.exit_code,
                        stdout_tail=_tail(outcome.stdout),
                        stderr_tail=_tail(outcome.stderr),
                    )
                )
            for cmd in pass_to_pass:
                outcome = self._docker.exec(
                    cname,
                    path_export + cmd,
                    workdir=self.workspace_dir,
                    timeout=self.command_timeout,
                )
                p2p_out.append(
                    CommandOutcome(
                        command=cmd,
                        exit_code=outcome.exit_code,
                        stdout_tail=_tail(outcome.stdout),
                        stderr_tail=_tail(outcome.stderr),
                    )
                )
            return SuiteOutcome(
                phase=phase,
                f2p=tuple(f2p_out),
                p2p=tuple(p2p_out),
                patch_applied=patch_applied,
                container_name=cname,
                logs=logs,
            )
        except EnvBuildError as exc:
            raise OracleDockerError(str(exc)) from exc
        finally:
            with contextlib.suppress(EnvBuildError):
                self._docker.remove_container(cname)
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    def run_broken(
        self,
        *,
        workspace: Path,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str] = (),
    ) -> SuiteOutcome:
        return self._run_phase(
            workspace=workspace,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            patch=None,
            phase="broken",
        )

    def run_with_patch(
        self,
        *,
        workspace: Path,
        patch: str,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str] = (),
        phase: str = "gold",
    ) -> SuiteOutcome:
        return self._run_phase(
            workspace=workspace,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            patch=patch,
            phase=phase,
        )


# ---------------------------------------------------------------------------
# In-memory backend for unit tests (no daemon)
# ---------------------------------------------------------------------------


@dataclass
class ScriptedSuite:
    """Pre-scripted exits for one backend call."""

    f2p_exits: list[int]
    p2p_exits: list[int] = field(default_factory=list)
    apply_ok: bool = True


@dataclass
class FakeOracleRunner:
    """Deterministic OracleRunnerBackend for unit tests."""

    broken: ScriptedSuite
    gold_runs: list[ScriptedSuite] = field(default_factory=list)
    null: ScriptedSuite | None = None
    _gold_i: int = 0
    cleaned: bool = False

    def run_broken(
        self,
        *,
        workspace: Path,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str] = (),
    ) -> SuiteOutcome:
        del workspace
        return self._suite("broken", self.broken, fail_to_pass, pass_to_pass, False)

    def run_with_patch(
        self,
        *,
        workspace: Path,
        patch: str,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str] = (),
        phase: str = "gold",
    ) -> SuiteOutcome:
        del workspace
        if not patch.strip():
            suit = self.null or self.broken
            return self._suite("null", suit, fail_to_pass, pass_to_pass, False)
        if self._gold_i >= len(self.gold_runs):
            suit = self.gold_runs[-1] if self.gold_runs else ScriptedSuite([0], [0])
        else:
            suit = self.gold_runs[self._gold_i]
            self._gold_i += 1
        if not suit.apply_ok:
            raise OracleDockerError(f"fake apply fail phase={phase}")
        return self._suite(phase, suit, fail_to_pass, pass_to_pass, True)

    def cleanup(self) -> None:
        self.cleaned = True

    @staticmethod
    def _suite(
        phase: str,
        script: ScriptedSuite,
        f2p_cmds: Sequence[str],
        p2p_cmds: Sequence[str],
        applied: bool,
    ) -> SuiteOutcome:
        f2p = tuple(
            CommandOutcome(command=cmd, exit_code=script.f2p_exits[i % len(script.f2p_exits)])
            for i, cmd in enumerate(f2p_cmds)
        )
        p2p = tuple(
            CommandOutcome(
                command=cmd,
                exit_code=(script.p2p_exits[i % len(script.p2p_exits)] if script.p2p_exits else 0),
            )
            for i, cmd in enumerate(p2p_cmds)
        )
        return SuiteOutcome(phase=phase, f2p=f2p, p2p=p2p, patch_applied=applied)


_LEAK_HINTS = re.compile(
    r"(gold\.patch|gold_patch|oracle_hidden|\.oracle_candidate\.patch)",
    re.IGNORECASE,
)


def scan_agent_workspace_leak(
    workspace: Path | None,
    *,
    gold_patch: str,
) -> list[str]:
    """Return human-readable leak findings for agent-visible trees (G5)."""
    if workspace is None or not workspace.is_dir():
        return []
    findings: list[str] = []
    gold_strips = [
        ln.strip()
        for ln in gold_patch.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    gold_body_markers = [ln.lstrip("+").strip() for ln in gold_strips if len(ln) > 8][:5]
    diff_suffixes = {".patch", ".diff", ".md", ".txt", ".json"}
    prompt_names = {
        "problem_statement.md",
        "task_meta.agent.json",
        "README.md",
        "prompt.md",
    }

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(workspace))
        if _LEAK_HINTS.search(rel):
            findings.append(f"forbidden path name: {rel}")
            continue
        # Skip large binary-ish
        try:
            if path.stat().st_size > 200_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "diff --git" in text and "+++ b/" in text and path.suffix in diff_suffixes:
            findings.append(f"unified diff content in agent file: {rel}")
        for marker in gold_body_markers:
            if marker and marker in text and path.name in prompt_names:
                findings.append(f"gold body marker in agent prompt file {rel}: {marker[:40]!r}")
                break
    return findings


__all__ = [
    "CommandOutcome",
    "FakeOracleRunner",
    "OracleDockerError",
    "OracleDockerRunner",
    "OracleRunnerBackend",
    "ScriptedSuite",
    "SuiteOutcome",
    "scan_agent_workspace_leak",
]
