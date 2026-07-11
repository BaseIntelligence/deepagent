"""Stage 1 env-first builder: one green-baseline Docker image per repo.

The builder is deliberately language-agnostic: it asks the :class:`LanguageAdapter`
for the base image, the baseline install commands (runtime + test/dev deps), and
the repo's own baseline test command, then drives Docker to:

1. check out exactly the pinned commit (on the host) and copy it into a fresh
   throwaway container off the language-correct base image,
2. run the baseline install so the repo's deps are present in the container,
3. run the repo's baseline suite and require it GREEN,
4. commit the container to a persisted image (the :class:`EnvImage`), then
5. re-run the recorded baseline command in a FRESH throwaway container off that
   persisted image to prove reproducibility independent of build-time state.

A RED baseline (or an install/build failure, reported distinctly) yields NO
usable image: the build is rejected with an explicit reason, every container is
torn down, and no ``baseline_green`` artifact is emitted. Docker hygiene is
strict: every container uses a unique run-scoped name and is removed by id even
on failure, intermediate/non-reproducible images are deleted, and only resources
this builder created are ever touched (off-limits containers are never named or
removed).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from swe_forge.forge.adapters import (
    AdapterRegistry,
    NoAdapterFoundError,
    build_default_registry,
)
from swe_forge.execution.sandbox import scoped_docker_name
from swe_forge.forge.models import EnvImage, RepoSpec

# Failure kinds (recorded on a rejected build so install/build vs test failures
# are never conflated).
DETECT_FAILED = "detect_failed"
CHECKOUT_FAILED = "checkout_failed"
IMAGE_PULL_FAILED = "image_pull_failed"
CONTAINER_FAILED = "container_failed"
INSTALL_FAILED = "install_failed"
BASELINE_FAILED = "baseline_failed"
REPRODUCE_FAILED = "reproduce_failed"
COMMIT_FAILED = "commit_failed"

_WORKSPACE_DIR = "/workspace/repo"
_TIMEOUT_EXIT = 124

# Best-effort system prep run inside the build container before install: ensure
# git is present (Python's setuptools-scm and other VCS-versioned builds need
# it) and mark the copied tree a safe git directory. Language-agnostic and
# tolerant of failure (a genuinely needed step failing surfaces later as an
# install failure with its own reason).
_PREP_SCRIPT = (
    "set +e; export DEBIAN_FRONTEND=noninteractive; "
    "git config --global --add safe.directory '*' >/dev/null 2>&1 || true; "
    "if ! command -v git >/dev/null 2>&1; then "
    "apt-get update -qq >/dev/null 2>&1 && "
    "apt-get install -y -qq git >/dev/null 2>&1; fi; "
    "git config --global --add safe.directory '*' >/dev/null 2>&1 || true; "
    "true"
)
_PREP_COMMANDS = (
    "ensure git present (apt-get install -y git if missing)",
    "git config --global --add safe.directory '*'",
)


class EnvBuildError(RuntimeError):
    """Raised for an unrecoverable Docker/CLI failure during an env build."""


@dataclass
class ExecOutcome:
    """Result of one command executed via the docker CLI."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""

    @property
    def combined(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)

    @property
    def timed_out(self) -> bool:
        return self.exit_code == _TIMEOUT_EXIT


def _tail(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    """Return the trailing lines/chars of ``text`` (for compact log capture)."""
    trimmed = text.strip()
    if not trimmed:
        return ""
    out = "\n".join(trimmed.splitlines()[-max_lines:])
    return out[-max_chars:]


def _last_line(text: str) -> str:
    """Return the last non-empty line of ``text`` (used as a green-proof snippet)."""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DockerCLI:
    """Thin synchronous wrapper over the ``docker`` CLI (via ``subprocess``).

    Each method maps to a single docker invocation. Command execution
    (:meth:`exec`, :meth:`run_ephemeral`) returns an :class:`ExecOutcome` and
    converts a timeout into exit code 124 rather than raising, so the builder can
    classify a hung install/baseline as a normal failure. Image/container
    lifecycle calls raise :class:`EnvBuildError` on failure.
    """

    def __init__(self, *, binary: str = "docker", default_timeout: float = 900.0):
        self._binary = binary
        self._default_timeout = default_timeout

    def _run(
        self, args: list[str], *, timeout: float | None = None, check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                [self._binary, *args],
                capture_output=True,
                text=True,
                timeout=timeout or self._default_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise EnvBuildError(
                f"docker {args[0]} timed out after {timeout or self._default_timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise EnvBuildError(f"docker binary {self._binary!r} not found") from exc
        if check and completed.returncode != 0:
            raise EnvBuildError(
                f"docker {' '.join(args)} failed: "
                f"{_tail(completed.stderr or completed.stdout)}"
            )
        return completed

    def _run_outcome(self, args: list[str], *, timeout: float) -> ExecOutcome:
        try:
            completed = subprocess.run(
                [self._binary, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            out = _as_text(exc.stdout)
            err = _as_text(exc.stderr)
            return ExecOutcome(
                _TIMEOUT_EXIT, out, f"{err}\n[timed out after {timeout}s]".strip()
            )
        return ExecOutcome(completed.returncode, completed.stdout, completed.stderr)

    def version(self) -> str:
        completed = self._run(["version", "--format", "{{.Server.Version}}"])
        return completed.stdout.strip() if completed.returncode == 0 else ""

    def image_id(self, ref: str) -> str | None:
        completed = self._run(["image", "inspect", "--format", "{{.Id}}", ref])
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    def image_exists(self, ref: str) -> bool:
        return self.image_id(ref) is not None

    def ensure_image(self, ref: str, *, timeout: float) -> None:
        if self.image_exists(ref):
            return
        completed = self._run(["pull", ref], timeout=timeout)
        if completed.returncode != 0:
            raise EnvBuildError(
                f"failed to pull base image {ref!r}: "
                f"{_tail(completed.stderr or completed.stdout)}"
            )

    def run_detached(
        self,
        *,
        name: str,
        image: str,
        workdir: str,
        memory_mb: int,
        cpus: float,
        pids_limit: int,
    ) -> str:
        args = [
            "run",
            "-d",
            "--name",
            name,
            "--memory",
            f"{memory_mb}m",
            "--cpus",
            str(cpus),
            "--pids-limit",
            str(pids_limit),
            "-w",
            workdir,
            image,
            "sleep",
            "infinity",
        ]
        completed = self._run(args)
        if completed.returncode != 0:
            raise EnvBuildError(
                f"failed to start build container {name!r}: "
                f"{_tail(completed.stderr or completed.stdout)}"
            )
        return completed.stdout.strip()

    def copy_into(self, container: str, src_dir: Path | str, dest_dir: str) -> None:
        src = f"{str(src_dir).rstrip('/')}/."
        completed = self._run(["cp", src, f"{container}:{dest_dir}"])
        if completed.returncode != 0:
            raise EnvBuildError(
                f"failed to copy repo into container {container!r}: "
                f"{_tail(completed.stderr or completed.stdout)}"
            )

    def exec(
        self, container: str, script: str, *, workdir: str, timeout: float
    ) -> ExecOutcome:
        return self._run_outcome(
            ["exec", "-w", workdir, container, "bash", "-c", script], timeout=timeout
        )

    def commit(self, container: str, tag: str, *, workdir: str) -> str:
        completed = self._run(["commit", "-c", f"WORKDIR {workdir}", container, tag])
        if completed.returncode != 0:
            raise EnvBuildError(
                f"failed to commit image {tag!r}: "
                f"{_tail(completed.stderr or completed.stdout)}"
            )
        return completed.stdout.strip()

    def run_ephemeral(
        self,
        *,
        name: str,
        image: str,
        script: str,
        workdir: str,
        timeout: float,
        memory_mb: int,
        cpus: float,
        pids_limit: int,
    ) -> ExecOutcome:
        args = [
            "run",
            "--rm",
            "--name",
            name,
            "--memory",
            f"{memory_mb}m",
            "--cpus",
            str(cpus),
            "--pids-limit",
            str(pids_limit),
            "-w",
            workdir,
            image,
            "bash",
            "-c",
            script,
        ]
        return self._run_outcome(args, timeout=timeout)

    def remove_container(self, ref: str) -> None:
        self._run(["rm", "-f", ref], timeout=60.0)

    def remove_image(self, ref: str) -> None:
        self._run(["rmi", "-f", ref], timeout=120.0)


def _as_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    return str(value) if value else ""


@dataclass
class _Checkout:
    ok: bool
    head: str = ""
    reason: str = ""


@dataclass
class EnvBuildResult:
    """Outcome of one env build (success carries the persisted :class:`EnvImage`)."""

    repo_id: str
    language: str
    success: bool
    stage: str
    failure_kind: str = ""
    reason: str = ""
    image_tag: str = ""
    env_image: EnvImage | None = None
    install_exit_code: int | None = None
    baseline_exit_code: int | None = None
    logs: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_id": self.repo_id,
            "language": self.language,
            "success": self.success,
            "stage": self.stage,
            "failure_kind": self.failure_kind,
            "reason": self.reason,
            "image_tag": self.image_tag,
            "install_exit_code": self.install_exit_code,
            "baseline_exit_code": self.baseline_exit_code,
            "env_image": self.env_image.to_dict() if self.env_image else None,
            "logs": dict(self.logs),
        }


class EnvBuilder:
    """Builds one green-baseline Docker image per repo via the language adapter."""

    def __init__(
        self,
        *,
        docker: DockerCLI | None = None,
        registry: AdapterRegistry | None = None,
        run_id: str | None = None,
        image_namespace: str = "swe-forge-env",
        workspace_dir: str = _WORKSPACE_DIR,
        memory_mb: int = 4096,
        cpus: float = 4.0,
        pids_limit: int = 2048,
        clone_timeout: float = 600.0,
        pull_timeout: float = 900.0,
        prep_timeout: float = 300.0,
        install_timeout: float = 1800.0,
        baseline_timeout: float = 1800.0,
    ) -> None:
        self._docker = docker or DockerCLI()
        self._registry = registry or build_default_registry()
        self._run_id = run_id or uuid.uuid4().hex[:8]
        self._namespace = image_namespace
        self._workspace_dir = workspace_dir
        self._memory_mb = memory_mb
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._clone_timeout = clone_timeout
        self._pull_timeout = pull_timeout
        self._prep_timeout = prep_timeout
        self._install_timeout = install_timeout
        self._baseline_timeout = baseline_timeout

    # -- public API -------------------------------------------------------- #
    def build(
        self, spec: RepoSpec, *, workdir: str | Path | None = None
    ) -> EnvBuildResult:
        """Build the env image for a registry :class:`RepoSpec`.

        Checks out exactly ``spec.commit`` (host side, unless an already-checked-
        out ``workdir`` is supplied) and drives the Docker build/baseline/commit
        flow. The pinned-SHA checkout is verified so the image is built from the
        recorded commit, never a moving branch tip. The spec's optional per-repo
        overrides (baseline install, baseline/P2P test command, P2P exclusions)
        are threaded in; unset overrides fall back to the adapter defaults.
        """
        install_override = list(spec.baseline_install) or None
        baseline_command_override = spec.baseline_test or None
        p2p_exclusions = spec.p2p_exclusions

        if workdir is not None:
            return self._build_in_docker(
                repo_path=Path(workdir),
                repo_id=spec.repo_id,
                language=spec.language,
                commit=spec.commit,
                url=spec.url,
                install_override=install_override,
                baseline_command_override=baseline_command_override,
                p2p_exclusions=p2p_exclusions,
            )

        tmp = Path(tempfile.mkdtemp(prefix="swe-forge-envbuild-"))
        try:
            checkout = self._checkout(spec, tmp)
            if not checkout.ok:
                return EnvBuildResult(
                    repo_id=spec.repo_id,
                    language=spec.language,
                    success=False,
                    stage="checkout",
                    failure_kind=CHECKOUT_FAILED,
                    reason=checkout.reason,
                )
            return self._build_in_docker(
                repo_path=tmp,
                repo_id=spec.repo_id,
                language=spec.language,
                commit=spec.commit,
                url=spec.url,
                install_override=install_override,
                baseline_command_override=baseline_command_override,
                p2p_exclusions=p2p_exclusions,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def build_from_path(
        self,
        repo_path: str | Path,
        *,
        repo_id: str,
        commit: str = "",
        url: str = "",
    ) -> EnvBuildResult:
        """Build the env image for a local repo checkout (no registry/network).

        Used for local fixtures (including deliberately RED ones). The recorded
        ``commit`` defaults to the local ``git rev-parse HEAD`` when available.
        """
        path = Path(repo_path)
        resolved_commit = commit or _git_head(path)
        return self._build_in_docker(
            repo_path=path,
            repo_id=repo_id,
            language="",
            commit=resolved_commit,
            url=url,
        )

    # -- internals --------------------------------------------------------- #
    def _checkout(self, spec: RepoSpec, dest: Path) -> _Checkout:
        for command in spec.checkout_commands():
            completed = subprocess.run(
                command,
                shell=True,
                cwd=dest,
                capture_output=True,
                text=True,
                timeout=self._clone_timeout,
            )
            if completed.returncode != 0:
                return _Checkout(
                    ok=False,
                    reason=(
                        f"checkout failed running {command!r}: "
                        f"{_tail(completed.stderr or completed.stdout)}"
                    ),
                )
        head = _git_head(dest)
        if head.lower() != spec.commit.lower():
            return _Checkout(
                ok=False,
                reason=(
                    f"checked-out HEAD {head!r} does not match pinned commit "
                    f"{spec.commit!r}"
                ),
            )
        return _Checkout(ok=True, head=head)

    def _unique_name(self, role: str) -> str:
        return scoped_docker_name(
            f"{self._namespace}-{role}-{self._run_id}-{uuid.uuid4().hex[:8]}"
        )

    def _image_tag(self, repo_id: str, commit: str) -> str:
        name = re.sub(r"[^a-z0-9._-]+", "_", repo_id.strip().lower()).strip("._-")
        fragment = re.sub(r"[^a-z0-9]+", "", commit.strip().lower())[:12] or "local"
        return f"{self._namespace}-{name or 'repo'}:{fragment}"

    def _build_in_docker(
        self,
        *,
        repo_path: Path,
        repo_id: str,
        language: str,
        commit: str,
        url: str,
        install_override: Sequence[str] | None = None,
        baseline_command_override: str | None = None,
        p2p_exclusions: Sequence[str] = (),
    ) -> EnvBuildResult:
        logs: dict[str, str] = {}

        try:
            adapter = self._registry.detect(repo_path)
        except NoAdapterFoundError as exc:
            return EnvBuildResult(
                repo_id=repo_id,
                language=language,
                success=False,
                stage="detect",
                failure_kind=DETECT_FAILED,
                reason=f"no language adapter matched the repo: {exc}",
            )

        base_image = adapter.base_image()
        # Per-repo overrides take precedence; otherwise fall back to the adapter
        # defaults. Overrides live on the RepoSpec so the stage stays
        # language-agnostic (no `if language ==` here).
        install_commands = (
            list(install_override)
            if install_override
            else adapter.baseline_install_commands(repo_path)
        )
        original_public_cmd = (
            baseline_command_override or adapter.baseline_test_command(repo_path)
        )
        baseline_cmd = original_public_cmd
        if p2p_exclusions:
            baseline_cmd = adapter.apply_p2p_exclusions(baseline_cmd, p2p_exclusions)
        workdir = self._workspace_dir

        try:
            self._docker.ensure_image(base_image, timeout=self._pull_timeout)
        except EnvBuildError as exc:
            return EnvBuildResult(
                repo_id=repo_id,
                language=adapter.name,
                success=False,
                stage="image_pull",
                failure_kind=IMAGE_PULL_FAILED,
                reason=str(exc),
            )

        tag = self._image_tag(repo_id, commit)
        container: str | None = None
        committed = False
        try:
            try:
                container = self._docker.run_detached(
                    name=self._unique_name("build"),
                    image=base_image,
                    workdir=workdir,
                    memory_mb=self._memory_mb,
                    cpus=self._cpus,
                    pids_limit=self._pids_limit,
                )
                self._docker.copy_into(container, repo_path, workdir)
            except EnvBuildError as exc:
                return EnvBuildResult(
                    repo_id=repo_id,
                    language=adapter.name,
                    success=False,
                    stage="container",
                    failure_kind=CONTAINER_FAILED,
                    reason=str(exc),
                    logs=logs,
                )

            prep = self._docker.exec(
                container, _PREP_SCRIPT, workdir=workdir, timeout=self._prep_timeout
            )
            logs["prep"] = _tail(prep.combined)

            install = self._docker.exec(
                container,
                "set -e\n" + "\n".join(install_commands),
                workdir=workdir,
                timeout=self._install_timeout,
            )
            logs["install"] = _tail(install.combined)
            if install.exit_code != 0:
                detail = " (timed out)" if install.timed_out else ""
                return EnvBuildResult(
                    repo_id=repo_id,
                    language=adapter.name,
                    success=False,
                    stage="install",
                    failure_kind=INSTALL_FAILED,
                    reason=(
                        f"install/build failed{detail} (exit {install.exit_code}); "
                        f"repo dependencies could not be installed"
                    ),
                    install_exit_code=install.exit_code,
                    logs=logs,
                )

            baseline = self._docker.exec(
                container, baseline_cmd, workdir=workdir, timeout=self._baseline_timeout
            )
            logs["baseline"] = _tail(baseline.combined)
            if baseline.exit_code != 0:
                detail = " (timed out)" if baseline.timed_out else ""
                return EnvBuildResult(
                    repo_id=repo_id,
                    language=adapter.name,
                    success=False,
                    stage="baseline",
                    failure_kind=BASELINE_FAILED,
                    reason=(
                        f"baseline tests failed{detail} (exit {baseline.exit_code}); "
                        f"baseline suite {baseline_cmd!r} is not green"
                    ),
                    install_exit_code=install.exit_code,
                    baseline_exit_code=baseline.exit_code,
                    logs=logs,
                )

            try:
                old_id = self._docker.image_id(tag)
                image_id = self._docker.commit(container, tag, workdir=workdir)
                committed = True
                if old_id and old_id != image_id:
                    self._docker.remove_image(old_id)
            except EnvBuildError as exc:
                return EnvBuildResult(
                    repo_id=repo_id,
                    language=adapter.name,
                    success=False,
                    stage="persist",
                    failure_kind=COMMIT_FAILED,
                    reason=str(exc),
                    install_exit_code=install.exit_code,
                    baseline_exit_code=baseline.exit_code,
                    logs=logs,
                )
        finally:
            if container is not None:
                self._docker.remove_container(container)

        # Reproducibility: re-run the recorded baseline in a FRESH throwaway
        # container off the persisted image, independent of build-time state.
        reproduce = self._docker.run_ephemeral(
            name=self._unique_name("baseline"),
            image=tag,
            script=baseline_cmd,
            workdir=workdir,
            timeout=self._baseline_timeout,
            memory_mb=self._memory_mb,
            cpus=self._cpus,
            pids_limit=self._pids_limit,
        )
        logs["reproduce"] = _tail(reproduce.combined)
        if reproduce.exit_code != 0:
            if committed:
                self._docker.remove_image(tag)
            detail = " (timed out)" if reproduce.timed_out else ""
            return EnvBuildResult(
                repo_id=repo_id,
                language=adapter.name,
                success=False,
                stage="reproduce",
                failure_kind=REPRODUCE_FAILED,
                reason=(
                    f"baseline not reproducible{detail} in a fresh container "
                    f"(exit {reproduce.exit_code}); image discarded"
                ),
                baseline_exit_code=reproduce.exit_code,
                logs=logs,
            )

        env_image = EnvImage(
            repo_id=repo_id,
            language=adapter.name,
            image_tag=tag,
            base_image=base_image,
            commit=commit,
            workspace_dir=workdir,
            install_commands=install_commands,
            baseline_test_command=baseline_cmd,
            baseline_green=True,
            baseline_exit_code=reproduce.exit_code,
            original_public_test_command=original_public_cmd,
            baseline_summary=_last_line(reproduce.combined),
            prep_commands=list(_PREP_COMMANDS),
            built_at=_now_iso(),
            provenance={
                "built_by": "swe-forge envbuild",
                "run_id": self._run_id,
                "docker_version": self._docker.version(),
                "source_url": url,
            },
        )
        return EnvBuildResult(
            repo_id=repo_id,
            language=adapter.name,
            success=True,
            stage="complete",
            image_tag=tag,
            env_image=env_image,
            install_exit_code=0,
            baseline_exit_code=reproduce.exit_code,
            logs=logs,
        )


def _git_head(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""
