"""Separate-verifier oracle for Harbor packs (solution reward=1, null reward=0).

VAL-HARBOR-003 / VAL-HARBOR-004 / VAL-CROSS-006:
- Build agent image from environment/ (no solution, no held-out test.patch)
- Build tests/ verifier image FROM agent image with tests/* baked in
- Run solution => reward.json reward == 1
- Run null/empty model.patch => reward == 0
- Fake backend for offline unit tests; Docker backend for integration
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from swe_factory.envbuild.builder import (
    DockerBackend,
    DockerCLI,
    EnvBuildError,
    remove_leftover_sdf_containers,
    scoped_container_name,
)
from swe_factory.harbor.harbor_docker import (
    HarborDockerError,
    assert_certified_test_patch,
    assert_certified_tests_config,
    build_agent_and_tests_images,
    remove_images,
    scan_agent_context_forbidden,
    stage_agent_context,
    summarize_agent_context,
)

_CONTAINER_PREFIX = "sdf-"


class HarborOracleError(RuntimeError):
    """Unrecoverable Harbor verifier oracle failure."""


@dataclass(frozen=True, slots=True)
class VerifierRunResult:
    """One separate-verifier evaluation (solution or null)."""

    phase: str
    reward: int | float | None
    reward_json: dict[str, Any] = field(default_factory=dict)
    logs: str = ""
    container_name: str = ""
    ok: bool = False

    @property
    def passed(self) -> bool:
        return self.ok and self.reward == 1


@dataclass(frozen=True, slots=True)
class HarborOracleResult:
    """Aggregate oracle outcome for a Harbor pack."""

    passed: bool
    task_id: str
    solution: VerifierRunResult
    null: VerifierRunResult
    agent_isolated: bool
    agent_context_summary: dict[str, Any] = field(default_factory=dict)
    config_ok: bool = False
    test_patch_ok: bool = False
    agent_image: str = ""
    tests_image: str = ""
    mode: str = "docker"
    reasons: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "task_id": self.task_id,
            "mode": self.mode,
            "solution_reward": self.solution.reward,
            "null_reward": self.null.reward,
            "agent_isolated": self.agent_isolated,
            "config_ok": self.config_ok,
            "test_patch_ok": self.test_patch_ok,
            "agent_image": self.agent_image,
            "tests_image": self.tests_image,
            "reasons": list(self.reasons),
            "agent_context_summary": dict(self.agent_context_summary),
            "details": dict(self.details),
        }


class HarborVerifierBackend(Protocol):
    """Injectable evaluator for solution/null runs against a pack."""

    def run_solution(self, pack_dir: Path) -> VerifierRunResult: ...

    def run_null(self, pack_dir: Path) -> VerifierRunResult: ...

    def cleanup(self) -> None: ...


@dataclass
class FakeHarborVerifier:
    """Offline fake: scripted rewards + isolation/config structural checks."""

    solution_reward: int | float = 1
    null_reward: int | float = 0
    cleaned: bool = False
    force_fail_isolation: bool = False
    force_fail_config: bool = False

    def run_solution(self, pack_dir: Path) -> VerifierRunResult:
        del pack_dir
        reward = self.solution_reward
        payload = {
            "reward": reward,
            "f2p_total": 2,
            "f2p_passed": 2 if reward == 1 else 0,
            "p2p_total": 1,
            "p2p_passed": 1 if reward == 1 else 0,
            "partial": 1.0 if reward == 1 else 0.0,
        }
        return VerifierRunResult(
            phase="solution",
            reward=reward,
            reward_json=payload,
            logs="fake solution verifier",
            ok=True,
        )

    def run_null(self, pack_dir: Path) -> VerifierRunResult:
        del pack_dir
        reward = self.null_reward
        payload = {
            "reward": reward,
            "f2p_total": 2,
            "f2p_passed": 0,
            "p2p_total": 1,
            "p2p_passed": 1 if reward == 0 else 0,
            "partial": 0.0 if reward == 0 else 1.0,
        }
        return VerifierRunResult(
            phase="null",
            reward=reward,
            reward_json=payload,
            logs="fake null verifier",
            ok=True,
        )

    def cleanup(self) -> None:
        self.cleaned = True


def _read_reward_from_logs(logs_dir: Path) -> tuple[int | float | None, dict[str, Any], str]:
    reward_path = logs_dir / "verifier" / "reward.json"
    reward_txt = logs_dir / "verifier" / "reward.txt"
    if reward_path.is_file():
        try:
            data = json.loads(reward_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, {}, "reward.json unparseable"
        if not isinstance(data, dict):
            return None, {}, "reward.json not object"
        reward = data.get("reward")
        return reward if isinstance(reward, int | float) else None, dict(data), "reward.json"
    if reward_txt.is_file():
        raw = reward_txt.read_text(encoding="utf-8").strip()
        try:
            return float(raw) if "." in raw else int(raw), {"reward": raw}, "reward.txt"
        except ValueError:
            return None, {}, f"reward.txt invalid: {raw!r}"
    return None, {}, "no reward file"


def _host_copy_paths(
    *,
    tests_image: str,
    model_patch: str | None,
    solution_patch: str | None,
    work: Path,
) -> tuple[Path, Path]:
    """Prepare host directories to mount as /logs/artifacts and /tmp patches."""
    artifacts = work / "artifacts"
    logs = work / "logs"
    artifacts.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "verifier").mkdir(parents=True, exist_ok=True)
    # model.patch (null = empty or missing)
    model_path = artifacts / "model.patch"
    if model_patch is not None and model_patch.strip():
        body = model_patch if model_patch.endswith("\n") else model_patch + "\n"
        model_path.write_text(body, encoding="utf-8")
    elif model_path.exists():
        model_path.unlink()
    if solution_patch is not None:
        # solution is applied to produce model.patch for the solution phase
        pass
    del tests_image
    return artifacts, logs


@dataclass
class HarborDockerVerifier:
    """Docker separate-verifier backend: build images, run test.sh with model.patch."""

    docker: DockerBackend | None = None
    run_id: str | None = None
    memory_mb: int = 1024
    cpus: float = 1.0
    pids_limit: int = 512
    build_timeout: float = 600.0
    run_timeout: float = 300.0
    cleanup_containers: bool = True
    remove_images_on_cleanup: bool = True
    agent_image: str | None = None
    tests_image: str | None = None
    binary: str = "docker"

    def __post_init__(self) -> None:
        self._docker: DockerBackend = self.docker or DockerCLI()
        self._run_id = self.run_id or uuid.uuid4().hex[:8]
        self._owned: list[str] = []
        self._images: list[str] = []
        self._work: Path | None = None
        self._built = False
        self._pack_dir: Path | None = None

    def _track(self, name: str) -> str:
        self._owned.append(name)
        return name

    def prepare_images(self, pack_dir: Path) -> tuple[str, str]:
        """Build (or reuse) agent + tests images for the pack."""
        if self._built and self.agent_image and self.tests_image:
            return self.agent_image, self.tests_image
        self._pack_dir = pack_dir
        self._work = Path(tempfile.mkdtemp(prefix="sdf-harbor-oracle-"))
        agent_tag = self.agent_image or f"harbor-sdf-agent-{self._run_id}:oracle"
        tests_tag = self.tests_image or f"harbor-sdf-tests-{self._run_id}:oracle"
        try:
            pair = build_agent_and_tests_images(
                pack_dir,
                work_dir=self._work / "contexts",
                agent_tag=agent_tag,
                tests_tag=tests_tag,
                binary=self.binary,
                build_timeout=self.build_timeout,
                stage_only=False,
            )
        except HarborDockerError as exc:
            raise HarborOracleError(str(exc)) from exc
        self.agent_image = pair.agent_image
        self.tests_image = pair.tests_image
        self._images = [pair.agent_image, pair.tests_image]
        self._built = True
        return pair.agent_image, pair.tests_image

    def _run_phase(
        self,
        pack_dir: Path,
        *,
        phase: str,
        model_patch: str,
    ) -> VerifierRunResult:
        import subprocess

        try:
            _, tests_image = self.prepare_images(pack_dir)
        except HarborOracleError as exc:
            return VerifierRunResult(
                phase=phase,
                reward=None,
                reward_json={},
                logs=str(exc),
                ok=False,
            )

        phase_work = Path(tempfile.mkdtemp(prefix=f"sdf-harbor-{phase}-"))
        artifacts, logs_dir = _host_copy_paths(
            tests_image=tests_image,
            model_patch=model_patch,
            solution_patch=None,
            work=phase_work,
        )
        # When model_patch is empty for null, leave no patch file (or empty)
        cname = self._track(scoped_container_name(f"horacle-{phase}", self._run_id))
        logs_blob = ""
        try:
            # Mount:
            #  - artifacts -> /logs/artifacts (model.patch)
            #  - logs -> /logs (verifier writes reward.json)
            # Image already has /app (repo), /tests/* (grader, test.patch, config)
            cmd = [
                self.binary,
                "run",
                "--rm",
                "--name",
                cname,
                "--memory",
                f"{self.memory_mb}m",
                "--cpus",
                str(self.cpus),
                "--pids-limit",
                str(self.pids_limit),
                "-v",
                f"{artifacts}:/logs/artifacts",
                "-v",
                f"{logs_dir}:/logs",
                "-w",
                "/app",
                "-e",
                "TESTS_DIR=/tests",
                "-e",
                "VERIFIER_DIR=/logs/verifier",
                "-e",
                "APP_DIR=/app",
                "-e",
                "ARTIFACTS_DIR=/logs/artifacts",
                "-e",
                "PYTHONPATH=/app",
                tests_image,
                "bash",
                "-c",
                "mkdir -p /logs/verifier /logs/artifacts && bash /tests/test.sh; "
                "ec=$?; "
                "ls -la /logs/verifier 2>/dev/null || true; "
                "cat /logs/verifier/reward.json 2>/dev/null || true; "
                "exit $ec",
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.run_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                logs_blob = (
                    f"timed out after {self.run_timeout}s\n{_as(exc.stdout)}\n{_as(exc.stderr)}"
                )
                return VerifierRunResult(
                    phase=phase,
                    reward=None,
                    reward_json={},
                    logs=logs_blob,
                    container_name=cname,
                    ok=False,
                )
            logs_blob = "\n".join(part for part in (completed.stdout, completed.stderr) if part)[
                -8000:
            ]
            reward, payload, _src = _read_reward_from_logs(logs_dir)
            # Fallback: parse reward.json printed to stdout
            if reward is None and "reward" in logs_blob:
                for line in reversed(logs_blob.splitlines()):
                    line = line.strip()
                    if line.startswith("{") and "reward" in line:
                        with contextlib.suppress(json.JSONDecodeError, TypeError):
                            data = json.loads(line)
                            if isinstance(data, dict) and "reward" in data:
                                reward = data["reward"]
                                payload = data
                                break
            ok = reward is not None
            return VerifierRunResult(
                phase=phase,
                reward=reward if isinstance(reward, int | float) else None,
                reward_json=payload,
                logs=logs_blob,
                container_name=cname,
                ok=ok,
            )
        finally:
            # container uses --rm; still force-kill if stuck
            with contextlib.suppress(Exception):
                subprocess.run(
                    [self.binary, "rm", "-f", cname],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            shutil.rmtree(phase_work, ignore_errors=True)

    def run_solution(self, pack_dir: Path) -> VerifierRunResult:
        sol = pack_dir / "solution" / "solution.patch"
        if not sol.is_file():
            return VerifierRunResult(
                phase="solution",
                reward=None,
                logs="solution/solution.patch missing",
                ok=False,
            )
        body = sol.read_text(encoding="utf-8")
        return self._run_phase(pack_dir, phase="solution", model_patch=body)

    def run_null(self, pack_dir: Path) -> VerifierRunResult:
        return self._run_phase(pack_dir, phase="null", model_patch="")

    def cleanup(self) -> None:
        if self.cleanup_containers:
            for name in reversed(list(self._owned)):
                if name.startswith(_CONTAINER_PREFIX):
                    with contextlib.suppress(EnvBuildError):
                        self._docker.remove_container(name)
            self._owned.clear()
            with contextlib.suppress(Exception):
                remove_leftover_sdf_containers(self._docker)
        if self.remove_images_on_cleanup and self._images:
            remove_images(self._images, binary=self.binary)
            self._images.clear()
        if self._work is not None:
            shutil.rmtree(self._work, ignore_errors=True)
            self._work = None


def _as(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    return str(value)


def _structural_certified_checks(
    pack_dir: Path,
    *,
    work_dir: Path | None = None,
) -> tuple[bool, bool, bool, dict[str, Any], list[str]]:
    """Config + test.patch + agent isolation structural checks."""
    reasons: list[str] = []
    config_ok = False
    patch_ok = False
    isolated = False
    summary: dict[str, Any] = {}
    try:
        assert_certified_tests_config(pack_dir / "tests" / "config.json")
        config_ok = True
    except HarborDockerError as exc:
        reasons.append(f"config: {exc}")
    try:
        assert_certified_test_patch(pack_dir / "tests" / "test.patch")
        patch_ok = True
    except HarborDockerError as exc:
        reasons.append(f"test_patch: {exc}")

    tmp_parent = work_dir or Path(tempfile.mkdtemp(prefix="sdf-harbor-agentctx-"))
    agent_dest = Path(tmp_parent) / "agent_context"
    own_tmp = work_dir is None
    try:
        staged = stage_agent_context(pack_dir, agent_dest, overwrite=True)
        hits = list(staged.forbidden_hits) or scan_agent_context_forbidden(staged.context_dir)
        summary = summarize_agent_context(staged.context_dir)
        if hits:
            reasons.append(f"agent isolation: {hits}")
            isolated = False
        else:
            isolated = True
    except HarborDockerError as exc:
        reasons.append(f"agent context: {exc}")
        isolated = False
    finally:
        if own_tmp:
            shutil.rmtree(tmp_parent, ignore_errors=True)
    return config_ok, patch_ok, isolated, summary, reasons


def run_harbor_oracle(
    pack_dir: Path | str,
    *,
    backend: HarborVerifierBackend | None = None,
    task_id: str | None = None,
    mode: str | None = None,
    cleanup: bool = True,
) -> HarborOracleResult:
    """Run separate-verifier oracle: solution reward=1 and null reward=0.

    Also enforces certified config f2p_node_ids + non-empty test.patch and
    agent context isolation (no solution/, no tests/test.patch).
    """
    root = Path(pack_dir)
    if not root.is_dir():
        raise HarborOracleError(f"pack dir not found: {root}")
    tid = task_id or root.name
    config_ok, patch_ok, isolated, summary, struct_reasons = _structural_certified_checks(root)

    runner: HarborVerifierBackend
    resolved_mode: str
    if backend is not None:
        runner = backend
        if isinstance(backend, FakeHarborVerifier):
            resolved_mode = mode or "fake"
            if backend.force_fail_isolation:
                isolated = False
                struct_reasons = [*struct_reasons, "agent isolation: forced fail"]
            if backend.force_fail_config:
                config_ok = False
                struct_reasons = [*struct_reasons, "config: forced fail"]
        else:
            resolved_mode = mode or "docker"
    else:
        runner = HarborDockerVerifier()
        resolved_mode = mode or "docker"

    solution = VerifierRunResult(phase="solution", reward=None, ok=False)
    null = VerifierRunResult(phase="null", reward=None, ok=False)
    run_reasons: list[str] = list(struct_reasons)
    agent_image = ""
    tests_image = ""
    try:
        solution = runner.run_solution(root)
        null = runner.run_null(root)
        if isinstance(runner, HarborDockerVerifier):
            agent_image = runner.agent_image or ""
            tests_image = runner.tests_image or ""
    except Exception as exc:  # noqa: BLE001 — surface as oracle reject
        run_reasons.append(f"oracle run error: {exc}")
    finally:
        if cleanup:
            with contextlib.suppress(Exception):
                runner.cleanup()

    if solution.reward != 1:
        run_reasons.append(
            f"solution reward expected 1, got {solution.reward!r} (logs: {solution.logs[-400:]})"
        )
    if null.reward != 0:
        run_reasons.append(
            f"null reward expected 0, got {null.reward!r} (logs: {null.logs[-400:]})"
        )
    if not isolated:
        run_reasons.append("agent context not isolated from solution/held-out tests")
    if not config_ok:
        run_reasons.append("tests/config.json f2p_node_ids/base_commit invalid")
    if not patch_ok:
        run_reasons.append("tests/test.patch empty or missing")

    passed = solution.reward == 1 and null.reward == 0 and isolated and config_ok and patch_ok
    return HarborOracleResult(
        passed=passed,
        task_id=tid,
        solution=solution,
        null=null,
        agent_isolated=isolated,
        agent_context_summary=summary,
        config_ok=config_ok,
        test_patch_ok=patch_ok,
        agent_image=agent_image,
        tests_image=tests_image,
        mode=resolved_mode,
        reasons=tuple(run_reasons),
        details={
            "solution_json": solution.reward_json,
            "null_json": null.reward_json,
        },
    )


def run_offline_harbor_oracle_fixture(
    *,
    out_dir: Path | str | None = None,
    fixture_root: Path | None = None,
    backend: HarborVerifierBackend | None = None,
) -> tuple[Any, HarborOracleResult]:
    """Emit offline Harbor pack, then run oracle (fake by default).

    Returns ``(OfflineHarborResult, HarborOracleResult)``.
    """
    from swe_factory.harbor.offline_fixture import run_offline_harbor_fixture

    dest = Path(out_dir) if out_dir is not None else Path("datasets/harbor_fixture")
    pack = run_offline_harbor_fixture(out_dir=dest, fixture_root=fixture_root)
    runner = backend or FakeHarborVerifier()
    result = run_harbor_oracle(
        pack.pack_dir,
        backend=runner,
        task_id=pack.task_id,
        mode="fake" if isinstance(runner, FakeHarborVerifier) else "docker",
    )
    # Persist oracle evidence next to the pack
    evidence = dest / "oracle_evidence.json"
    evidence.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return pack, result


# Convenience alias used by CLI wiring
OracleBackendFactory = Callable[[], HarborVerifierBackend]


__all__ = [
    "FakeHarborVerifier",
    "HarborDockerVerifier",
    "HarborOracleError",
    "HarborOracleResult",
    "HarborVerifierBackend",
    "VerifierRunResult",
    "run_harbor_oracle",
    "run_offline_harbor_oracle_fixture",
]
