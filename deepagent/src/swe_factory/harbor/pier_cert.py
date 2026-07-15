"""Pier certification adapter for DeepAgent Harbor packs (VAL-PIER-001..005).

Certifies Pier-loadable packs by:

- loading pack TaskConfig / tree (structural smoke; pier / harbor load path)
- running Pier ``-a oracle`` (solution) expect reward.json reward=1
- running Pier ``-a nop`` (null / no-patch) expect reward.json reward=0
- refusing oracle_mode=fake on deepagent keeps
- re-checking agent isolation before promote

Jobs default under ``/tmp/harbor-deepagent-jobs*``. Offline unit tests inject
backends and/or pre-seeded job trees so pier/docker need not execute.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from swe_factory.harbor.deepagent_cert import (
    FakeBackendRejected,
    IsolationEvidence,
    PackMetaEvidence,
    PierReadyHooks,
    build_pier_ready_hooks,
    is_real_base_sha,
    is_real_repository_url,
    read_pack_meta,
    refuse_fake_backend,
    scan_pack_agent_isolation,
)
from swe_factory.harbor.export_pack import verify_pack_tree
from swe_factory.oracle import codes as C

PierAgent = Literal["oracle", "nop"]
PierCertDisposition = Literal["accept", "reject"]

DEFAULT_JOBS_ROOT = Path("/tmp/harbor-deepagent-jobs")
DEFAULT_PIER_BIN = Path("/tmp/pier-venv/bin/pier")
_REWARD_NAME = "reward.json"
_FAKE_MODE_TOKENS = frozenset({"fake", "stub", "mock", "offline"})
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


class PierCertError(RuntimeError):
    """Unrecoverable Pier certification failure (surfaces clearly to CLI)."""


class PierInvokeError(PierCertError):
    """Pier process failed to start or returned an unusable job tree."""


@dataclass(frozen=True, slots=True)
class RewardEvidence:
    """Parsed pier verifier reward.json (or equivalent) content."""

    reward: int | float | None
    path: str | None
    raw: dict[str, Any] = field(default_factory=dict)
    agent: str = ""
    parse_ok: bool = False
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "path": self.path,
            "agent": self.agent,
            "parse_ok": self.parse_ok,
            "raw": dict(self.raw),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class PierRunEvidence:
    """One pier job (oracle or nop) run and its evidence surface."""

    agent: str
    job_dir: str
    reward: RewardEvidence
    exit_code: int | None = None
    command: tuple[str, ...] = ()
    stdout_tail: str = ""
    stderr_tail: str = ""
    trial_dir: str | None = None
    result_json: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "job_dir": self.job_dir,
            "reward": self.reward.to_dict(),
            "exit_code": self.exit_code,
            "command": list(self.command),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "trial_dir": self.trial_dir,
            "result_json": dict(self.result_json),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class PierCertResult:
    """Aggregate Pier certification outcome for one pack."""

    certified: bool
    disposition: PierCertDisposition
    task_id: str
    pack_dir: str
    jobs_root: str
    structural_ok: bool
    pier_ready: PierReadyHooks
    oracle_run: PierRunEvidence | None
    null_run: PierRunEvidence | None
    isolation: IsolationEvidence
    pack_meta: PackMetaEvidence
    backend: str
    oracle_mode: str
    reason_codes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "certified": self.certified,
            "disposition": self.disposition,
            "task_id": self.task_id,
            "pack_dir": self.pack_dir,
            "jobs_root": self.jobs_root,
            "structural_ok": self.structural_ok,
            "pier_ready": self.pier_ready.to_dict(),
            "oracle_run": self.oracle_run.to_dict() if self.oracle_run else None,
            "null_run": self.null_run.to_dict() if self.null_run else None,
            "sol_reward": self.oracle_run.reward.reward if self.oracle_run else None,
            "null_reward": self.null_run.reward.reward if self.null_run else None,
            "isolation": self.isolation.to_dict(),
            "isolation_status": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "pack_meta": self.pack_meta.to_dict(),
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "backend": self.backend,
            "oracle_mode": self.oracle_mode,
            "reason_codes": list(self.reason_codes),
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }

    def to_audit_row(self) -> dict[str, Any]:
        return {
            "instance_id": self.task_id,
            "task_id": self.task_id,
            "disposition": self.disposition,
            "certified": self.certified,
            "backend": self.backend,
            "oracle_mode": self.oracle_mode,
            "sol": self.oracle_run.reward.reward if self.oracle_run else None,
            "null": self.null_run.reward.reward if self.null_run else None,
            "isolation": "clean" if self.isolation.clean else "leak",
            "agent_isolated": self.isolation.clean,
            "structural_ok": self.structural_ok,
            "repository_url": self.pack_meta.repository_url,
            "base_commit_hash": self.pack_meta.base_commit_hash,
            "jobs_root": self.jobs_root,
            "reason_codes": list(self.reason_codes),
            "oracle_reward_path": (self.oracle_run.reward.path if self.oracle_run else None),
            "null_reward_path": self.null_run.reward.path if self.null_run else None,
        }


class PierRunner(Protocol):
    """Injectable Pier job runner (real subprocess or offline scripted)."""

    def run(
        self,
        *,
        pack_dir: Path,
        agent: PierAgent,
        jobs_dir: Path,
        job_name: str | None = None,
        n_concurrent: int = 1,
        force_build: bool = True,
        extra_args: Sequence[str] | None = None,
    ) -> PierRunEvidence: ...


# ---------------------------------------------------------------------------
# Evidence parsers
# ---------------------------------------------------------------------------


def parse_reward_json(
    path: Path | str,
    *,
    agent: str = "",
) -> RewardEvidence:
    """Parse a pier verifier ``reward.json`` into structured evidence.

    Accepts common shapes:
    - ``{"reward": 1}``
    - ``{"reward": 0.0, "f2p": ...}``
    - pure number JSON ``1`` / ``0``
    - text file with a single number (legacy reward.txt-ish misname)
    """
    p = Path(path)
    errors: list[str] = []
    if not p.is_file():
        return RewardEvidence(
            reward=None,
            path=str(p),
            agent=agent,
            parse_ok=False,
            errors=(f"reward file missing: {p}",),
        )
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return RewardEvidence(
            reward=None,
            path=str(p),
            agent=agent,
            parse_ok=False,
            errors=(f"reward read failed: {exc}",),
        )
    if not text:
        return RewardEvidence(
            reward=None,
            path=str(p),
            agent=agent,
            parse_ok=False,
            errors=("reward file empty",),
        )
    raw: dict[str, Any] = {}
    reward: int | float | None = None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        # bare number text
        try:
            reward = float(text) if "." in text else int(text)
            raw = {"reward": reward, "_text": text}
        except ValueError as exc:
            errors.append(f"reward parse failed: {exc}")
            return RewardEvidence(
                reward=None,
                path=str(p),
                raw={"_text": text},
                agent=agent,
                parse_ok=False,
                errors=tuple(errors),
            )
    else:
        if isinstance(loaded, int | float) and not isinstance(loaded, bool):
            reward = loaded
            raw = {"reward": reward}
        elif isinstance(loaded, Mapping):
            raw = dict(loaded)
            if "reward" in loaded:
                reward = _coerce_reward(loaded["reward"])
            elif "score" in loaded:
                reward = _coerce_reward(loaded["score"])
            elif "pass" in loaded and isinstance(loaded["pass"], bool):
                reward = 1 if loaded["pass"] else 0
            else:
                errors.append("reward.json missing reward/score/pass field")
        else:
            errors.append(f"unsupported reward.json type: {type(loaded).__name__}")

    parse_ok = reward is not None and not errors
    if reward is None and not errors:
        errors.append("could not extract numeric reward")
    return RewardEvidence(
        reward=reward,
        path=str(p),
        raw=raw,
        agent=agent,
        parse_ok=parse_ok,
        errors=tuple(errors),
    )


def _coerce_reward(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned) if "." in cleaned else int(cleaned)
        except ValueError:
            return None
    return None


def find_reward_jsons(job_or_trial_dir: Path | str) -> list[Path]:
    """Locate verifier/reward.json files under a pier job or trial directory."""
    root = Path(job_or_trial_dir)
    if not root.exists():
        return []
    matches: list[Path] = []
    # Prefer conventional pier trial layout: <trial>/verifier/reward.json
    for path in root.rglob(_REWARD_NAME):
        if path.is_file() and path.parent.name == "verifier":
            matches.append(path)
    if matches:
        return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    # Fallback: any reward.json
    for path in root.rglob(_REWARD_NAME):
        if path.is_file():
            matches.append(path)
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)


def find_latest_trial_dir(job_dir: Path | str) -> Path | None:
    """Return the most recently modified trial directory under a pier job."""
    root = Path(job_dir)
    if not root.is_dir():
        return None
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        # trial dirs have verifier/ or result.json
        if (child / "verifier").is_dir() or (child / "result.json").is_file():
            candidates.append(child)
        else:
            # nested: job_dir / <timestamp> / <trial>
            for sub in child.iterdir() if child.is_dir() else []:
                if sub.is_dir() and (
                    (sub / "verifier").is_dir() or (sub / "result.json").is_file()
                ):
                    candidates.append(sub)
    if not candidates:
        # timestamp wrapper without classifying yet
        for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir():
                nested = [
                    s
                    for s in child.iterdir()
                    if s.is_dir()
                    and (
                        (s / "verifier").is_dir()
                        or (s / "result.json").is_file()
                        or (s / "config.json").is_file()
                    )
                ]
                if nested:
                    return max(nested, key=lambda p: p.stat().st_mtime)
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_pier_job_result(job_dir: Path | str, *, agent: str = "") -> RewardEvidence:
    """Parse the latest reward under a pier jobs dir / job output.

    Prefers trial ``verifier/reward.json``; falls back to trial ``result.json``
    ``verifier_result.rewards.reward``; then top-level job ``result.json`` metrics.
    """
    root = Path(job_dir)
    if not root.exists():
        return RewardEvidence(
            reward=None,
            path=None,
            agent=agent,
            parse_ok=False,
            errors=(f"job dir missing: {root}",),
        )

    rewards = find_reward_jsons(root)
    if rewards:
        return parse_reward_json(rewards[0], agent=agent)

    trial = find_latest_trial_dir(root)
    if trial is not None:
        result_path = trial / "result.json"
        if result_path.is_file():
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return RewardEvidence(
                    reward=None,
                    path=str(result_path),
                    agent=agent,
                    parse_ok=False,
                    errors=(f"result.json parse failed: {exc}",),
                )
            rewards_map = ((data.get("verifier_result") or {}).get("rewards")) or {}
            if isinstance(rewards_map, Mapping) and "reward" in rewards_map:
                reward = _coerce_reward(rewards_map["reward"])
                return RewardEvidence(
                    reward=reward,
                    path=str(result_path),
                    raw=dict(rewards_map),
                    agent=agent,
                    parse_ok=reward is not None,
                    errors=() if reward is not None else ("result.json rewards.reward missing",),
                )

    job_result = root / "result.json"
    if job_result.is_file():
        try:
            data = json.loads(job_result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return RewardEvidence(
                reward=None,
                path=str(job_result),
                agent=agent,
                parse_ok=False,
                errors=(f"job result.json parse failed: {exc}",),
            )
        # stats.evals.*.metrics[0].reward
        evals = ((data.get("stats") or {}).get("evals")) or {}
        if isinstance(evals, Mapping):
            for _name, eval_body in evals.items():
                metrics = (eval_body or {}).get("metrics") or []
                if metrics and isinstance(metrics[0], Mapping) and "reward" in metrics[0]:
                    reward = _coerce_reward(metrics[0]["reward"])
                    return RewardEvidence(
                        reward=reward,
                        path=str(job_result),
                        raw=dict(metrics[0]),
                        agent=agent,
                        parse_ok=reward is not None,
                        errors=() if reward is not None else ("job result metrics.reward missing",),
                    )

    return RewardEvidence(
        reward=None,
        path=None,
        agent=agent,
        parse_ok=False,
        errors=(f"no reward.json found under {root}",),
    )


def load_trial_result_json(trial_dir: Path | str) -> dict[str, Any]:
    """Load trial result.json when present (empty dict otherwise)."""
    path = Path(trial_dir) / "result.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Pier runners
# ---------------------------------------------------------------------------


def resolve_pier_bin(
    pier_bin: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Locate pier executable (PIER_BIN env, explicit path, pier-venv, PATH)."""
    environ = env if env is not None else os.environ
    if pier_bin is not None:
        candidate = Path(pier_bin)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise PierInvokeError(f"pier binary not executable: {candidate}")

    env_bin = (environ.get("PIER_BIN") or "").strip()
    if env_bin:
        candidate = Path(env_bin)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise PierInvokeError(f"PIER_BIN not executable: {candidate}")

    if DEFAULT_PIER_BIN.is_file() and os.access(DEFAULT_PIER_BIN, os.X_OK):
        return DEFAULT_PIER_BIN

    which = shutil.which("pier")
    if which:
        return Path(which)
    raise PierInvokeError(
        "pier 0.3+ not found; install via services pier_venv_ensure "
        f"or set PIER_BIN (looked for {DEFAULT_PIER_BIN} and PATH)"
    )


def ensure_jobs_root(jobs_root: Path | str | None = None) -> Path:
    """Ensure jobs root lives under /tmp/harbor-deepagent-jobs* and exists."""
    root = Path(jobs_root) if jobs_root is not None else DEFAULT_JOBS_ROOT
    text = str(root)
    # Allow /tmp/harbor-deepagent-jobs and /tmp/harbor-deepagent-jobs-<suffix>
    if not (
        text == str(DEFAULT_JOBS_ROOT)
        or text.startswith(str(DEFAULT_JOBS_ROOT) + "/")
        or text.startswith(str(DEFAULT_JOBS_ROOT) + "-")
        or text.startswith("/tmp/harbor-deepagent-jobs")
    ):
        raise PierCertError(
            f"pier jobs root must be under /tmp/harbor-deepagent-jobs* (got {root})"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


@dataclass
class SubprocessPierRunner:
    """Invoke ``pier run -a <agent> -p <pack> -o <jobs_dir>`` via subprocess."""

    pier_bin: Path | str | None = None
    timeout_sec: float = 1800.0
    env: Mapping[str, str] | None = None

    def run(
        self,
        *,
        pack_dir: Path,
        agent: PierAgent,
        jobs_dir: Path,
        job_name: str | None = None,
        n_concurrent: int = 1,
        force_build: bool = True,
        extra_args: Sequence[str] | None = None,
    ) -> PierRunEvidence:
        binary = resolve_pier_bin(self.pier_bin, env=self.env)
        jobs_dir.mkdir(parents=True, exist_ok=True)
        name = job_name or f"{agent}-{int(time.time())}"
        cmd: list[str] = [
            str(binary),
            "run",
            "-a",
            agent,
            "-p",
            str(Path(pack_dir).resolve()),
            "-o",
            str(Path(jobs_dir).resolve()),
            "--job-name",
            name,
            "-n",
            str(max(1, int(n_concurrent))),
            "-y",
        ]
        if force_build:
            cmd.append("--force-build")
        if extra_args:
            cmd.extend(list(extra_args))

        errors: list[str] = []
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
                env=dict(self.env) if self.env is not None else None,
            )
        except FileNotFoundError as exc:
            raise PierInvokeError(f"pier executable missing: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PierInvokeError(
                f"pier run timed out after {self.timeout_sec}s (agent={agent}): {exc}"
            ) from exc

        job_out = Path(jobs_dir) / name
        # pier may nest under jobs_dir/name; fall back to latest under jobs_dir
        search_root = job_out if job_out.exists() else Path(jobs_dir)
        reward = parse_pier_job_result(search_root, agent=agent)
        trial = find_latest_trial_dir(search_root)
        result_json = load_trial_result_json(trial) if trial else {}
        if proc.returncode != 0:
            errors.append(f"pier exit {proc.returncode}")
        if not reward.parse_ok:
            errors.extend(reward.errors)
            errors.append(
                f"pier agent={agent} did not produce parseable reward under {search_root}"
            )
        stdout_tail = (proc.stdout or "")[-2000:]
        stderr_tail = (proc.stderr or "")[-2000:]
        if proc.returncode != 0 and stderr_tail.strip():
            # surface pier stderr clearly in reasons
            errors.append(f"pier stderr: {stderr_tail.strip()[-500:]}")
        return PierRunEvidence(
            agent=agent,
            job_dir=str(search_root),
            reward=reward,
            exit_code=proc.returncode,
            command=tuple(cmd),
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            trial_dir=str(trial) if trial else None,
            result_json=result_json,
            errors=tuple(errors),
        )


@dataclass
class ScriptedPierRunner:
    """Offline pier runner for unit tests (no subprocess / docker)."""

    oracle_reward: int | float = 1
    null_reward: int | float = 0
    fail_agent: str | None = None
    force_missing_reward: bool = False
    call_log: list[str] = field(default_factory=list)

    def run(
        self,
        *,
        pack_dir: Path,
        agent: PierAgent,
        jobs_dir: Path,
        job_name: str | None = None,
        n_concurrent: int = 1,
        force_build: bool = True,
        extra_args: Sequence[str] | None = None,
    ) -> PierRunEvidence:
        del pack_dir, n_concurrent, force_build, extra_args
        self.call_log.append(agent)
        name = job_name or f"scripted-{agent}"
        job_out = Path(jobs_dir) / name
        trial = job_out / f"scripted-trial-{agent}"
        verifier = trial / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        reward_val: int | float | None = (
            self.oracle_reward if agent == "oracle" else self.null_reward
        )
        reward_path = verifier / _REWARD_NAME
        if self.force_missing_reward or self.fail_agent == agent:
            # leave reward missing so parsers surface clear errors
            reward = RewardEvidence(
                reward=None,
                path=str(reward_path),
                agent=agent,
                parse_ok=False,
                errors=(f"scripted missing reward for agent={agent}",),
            )
            errors.append(f"scripted pier agent={agent} reward missing")
            exit_code = 1
        else:
            payload = {
                "reward": reward_val,
                "f2p_total": 2,
                "f2p_passed": 2 if reward_val == 1 else 0,
                "p2p_total": 1,
                "p2p_passed": 1 if reward_val in (0, 1) else 0,
            }
            reward_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            reward = parse_reward_json(reward_path, agent=agent)
            exit_code = 0
        return PierRunEvidence(
            agent=agent,
            job_dir=str(job_out),
            reward=reward,
            exit_code=exit_code,
            command=("scripted-pier", "run", "-a", agent),
            stdout_tail="scripted pier ok",
            stderr_tail="",
            trial_dir=str(trial),
            result_json={
                "verifier_result": {
                    "rewards": {"reward": reward.reward},
                }
            },
            errors=tuple(errors),
        )


# ---------------------------------------------------------------------------
# Cert gates
# ---------------------------------------------------------------------------


def refuse_fake_oracle_mode(
    oracle_mode: str | None,
    *,
    certified: bool = True,
    pack_or_dest: Path | str | None = None,
) -> None:
    """VAL-PIER-004: certified deepagent path rejects oracle_mode=fake."""
    mode = (oracle_mode or "docker").strip().lower()
    text = str(pack_or_dest or "").replace("\\", "/").lower()
    is_deepagent = any(
        token in text
        for token in ("deepagent_v1", "datasets/deepagent", "deepagent-v1", "deepagent")
    )
    if not certified and not is_deepagent:
        return
    if mode in _FAKE_MODE_TOKENS:
        raise FakeBackendRejected(
            "pier cert / deepagent keeps refuse oracle_mode=fake "
            f"(got oracle_mode={mode!r}); docker backend required (VAL-PIER-004)"
        )
    # also block via refuse_fake_backend for shared wording
    refuse_fake_backend(mode if mode != "docker" else "docker", certified=True)


def structural_load_pack(
    pack_dir: Path | str,
    *,
    run_load_smoke: bool = True,
    pier_job_prefix: str = str(DEFAULT_JOBS_ROOT),
) -> PierReadyHooks:
    """VAL-PIER-001: Pier/Harbor structural load without schema structure error."""
    return build_pier_ready_hooks(
        pack_dir,
        run_load_smoke=run_load_smoke,
        pier_job_prefix=pier_job_prefix,
    )


def write_pier_evidence(path: Path | str, result: PierCertResult) -> Path:
    """Persist full pier cert evidence JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def append_pier_audit(path: Path | str, result: PierCertResult) -> Path:
    """Append one pier cert audit row (gate_audit style)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_audit_row(), sort_keys=True) + "\n")
    return out


def certify_pier_pack(
    pack_dir: Path | str,
    *,
    runner: PierRunner | None = None,
    jobs_root: Path | str | None = None,
    task_id: str | None = None,
    oracle_mode: str = "docker",
    run_oracle: bool = True,
    run_null: bool = True,
    run_load_smoke: bool = True,
    force_build: bool = True,
    pier_bin: Path | str | None = None,
    evidence_out: Path | str | None = None,
    audit_out: Path | str | None = None,
    isolation: IsolationEvidence | None = None,
    n_concurrent: int = 1,
    timeout_sec: float = 1800.0,
) -> PierCertResult:
    """Pier-certify one Harbor pack (VAL-PIER-001..005).

    - Structural load must succeed (no schema/tree structure error)
    - Oracle agent reward must parse and equal 1
    - Null/nop agent reward must parse and equal 0
    - oracle_mode must not be fake on cert/deepagent path
    - Agent isolation re-check must be clean
    """
    root = Path(pack_dir)
    if not root.is_dir():
        raise PierCertError(f"pack dir not found: {root}")

    refuse_fake_oracle_mode(oracle_mode, certified=True, pack_or_dest=root)

    pack_meta = read_pack_meta(root)
    tid = task_id or pack_meta.task_id or root.name
    jobs = ensure_jobs_root(jobs_root)

    codes: list[str] = []
    reasons: list[str] = []
    codes.extend(pack_meta.reason_codes)
    reasons.extend(pack_meta.reasons)

    pier_ready = structural_load_pack(
        root,
        run_load_smoke=run_load_smoke,
        pier_job_prefix=str(jobs),
    )
    structural_ok = pier_ready.structural_ok and pier_ready.required_relpaths_ok
    if not pier_ready.required_relpaths_ok:
        codes.append("PACK_TREE_INCOMPLETE")
        reasons.append(f"incomplete pack tree: {list(pier_ready.missing_relpaths)}")
    if not pier_ready.structural_ok:
        codes.append("PIER_STRUCTURAL_LOAD_FAIL")
        reasons.append(
            "pier/harbor structural load failed: "
            + "; ".join(pier_ready.notes or list((pier_ready.load_smoke or {}).get("errors") or []))
        )
    else:
        codes.append("PIER_STRUCTURAL_OK")

    isol = isolation if isolation is not None else scan_pack_agent_isolation(root)
    if not isol.clean:
        codes.append(C.G5_LEAK)
        reasons.append(f"agent isolation leak at pier cert stage: {list(isol.hits)}")
    else:
        codes.append(C.G5_LEAK_CLEAN)

    backend = (
        "docker"
        if (oracle_mode or "docker").strip().lower() not in _FAKE_MODE_TOKENS
        else (oracle_mode or "docker").strip().lower()
    )
    if backend in _FAKE_MODE_TOKENS:
        # Should already have raised; belt-and-suspenders
        raise FakeBackendRejected(f"pier cert refuses backend/oracle_mode={backend!r}")

    active_runner: PierRunner
    if runner is not None:
        active_runner = runner
    else:
        active_runner = SubprocessPierRunner(pier_bin=pier_bin, timeout_sec=timeout_sec)

    # M15: pack-scoped agent mintag (deepagent-agent-<digest>:local) painted onto
    # tests/Dockerfile FROM. Live Subprocess pier needs that mintag present;
    # stage+build from environment/ before oracle/nop (skip for injected/
    # scripted runners). FAIL CLOSED on mintag prepare — do not continue into
    # pier oracle/nop with a wrong or missing base image.
    agent_mintag: str | None = None
    mintag_ok = True
    if (
        isinstance(active_runner, SubprocessPierRunner)
        and (run_oracle or run_null)
        and (oracle_mode or "docker").strip().lower() not in _FAKE_MODE_TOKENS
    ):
        try:
            from swe_factory.harbor.harbor_docker import ensure_deepagent_agent_local

            agent_mintag = ensure_deepagent_agent_local(
                root,
                work_dir=jobs / "_agent_mintag" / tid,
                force_rebuild=False,
                build_timeout=max(600.0, float(timeout_sec)),
                paint_tests_from=True,
            )
            codes.append("PIER_AGENT_MINTAG_OK")
            reasons.append(f"agent mintag ready: {agent_mintag}")
        except Exception as exc:  # noqa: BLE001 — fail closed as cert reject
            mintag_ok = False
            codes.append("PIER_AGENT_MINTAG_FAIL")
            reasons.append(f"agent mintag ensure failed (fail-closed): {exc}")

    oracle_run: PierRunEvidence | None = None
    null_run: PierRunEvidence | None = None

    # Skip pier oracle/nop when mintag prepare failed (fail closed).
    if mintag_ok and run_oracle:
        try:
            oracle_run = active_runner.run(
                pack_dir=root,
                agent="oracle",
                jobs_dir=jobs / "oracle",
                job_name=f"oracle-{tid}",
                n_concurrent=n_concurrent,
                force_build=force_build,
            )
        except PierInvokeError as exc:
            oracle_run = PierRunEvidence(
                agent="oracle",
                job_dir=str(jobs / "oracle"),
                reward=RewardEvidence(
                    reward=None,
                    path=None,
                    agent="oracle",
                    parse_ok=False,
                    errors=(str(exc),),
                ),
                exit_code=None,
                errors=(str(exc),),
            )
            codes.append("PIER_ORACLE_INVOKE_FAIL")
            reasons.append(f"pier oracle invoke failed: {exc}")
        else:
            if not oracle_run.reward.parse_ok:
                codes.append("PIER_ORACLE_REWARD_UNPARSEABLE")
                reasons.append(
                    "oracle reward not parseable: "
                    + "; ".join(oracle_run.reward.errors or oracle_run.errors)
                )
            elif oracle_run.reward.reward != 1 and oracle_run.reward.reward != 1.0:
                codes.append("PIER_ORACLE_REWARD_NOT_1")
                reasons.append(
                    f"pier oracle reward expected 1, got {oracle_run.reward.reward!r} "
                    f"(path={oracle_run.reward.path})"
                )
            else:
                codes.append("PIER_ORACLE_REWARD_1")
            if oracle_run.errors and oracle_run.reward.reward != 1:
                reasons.extend(list(oracle_run.errors)[:6])
    elif not mintag_ok and run_oracle:
        codes.append("PIER_ORACLE_SKIPPED_MINTAG_FAIL")
        reasons.append("pier oracle skipped because agent mintag prepare failed")

    if mintag_ok and run_null:
        try:
            null_run = active_runner.run(
                pack_dir=root,
                agent="nop",
                jobs_dir=jobs / "nop",
                job_name=f"nop-{tid}",
                n_concurrent=n_concurrent,
                force_build=force_build,
            )
        except PierInvokeError as exc:
            null_run = PierRunEvidence(
                agent="nop",
                job_dir=str(jobs / "nop"),
                reward=RewardEvidence(
                    reward=None,
                    path=None,
                    agent="nop",
                    parse_ok=False,
                    errors=(str(exc),),
                ),
                exit_code=None,
                errors=(str(exc),),
            )
            codes.append("PIER_NULL_INVOKE_FAIL")
            reasons.append(f"pier null/nop invoke failed: {exc}")
        else:
            if not null_run.reward.parse_ok:
                codes.append("PIER_NULL_REWARD_UNPARSEABLE")
                reasons.append(
                    "null reward not parseable: "
                    + "; ".join(null_run.reward.errors or null_run.errors)
                )
            elif null_run.reward.reward != 0 and null_run.reward.reward != 0.0:
                codes.append("PIER_NULL_REWARD_NOT_0")
                reasons.append(
                    f"pier null/nop reward expected 0, got {null_run.reward.reward!r} "
                    f"(path={null_run.reward.path})"
                )
            else:
                codes.append("PIER_NULL_REWARD_0")
            if null_run.errors and null_run.reward.reward != 0:
                reasons.extend(list(null_run.errors)[:6])
    elif not mintag_ok and run_null:
        codes.append("PIER_NULL_SKIPPED_MINTAG_FAIL")
        reasons.append("pier null/nop skipped because agent mintag prepare failed")

    sol_ok = (
        oracle_run is not None
        and oracle_run.reward.parse_ok
        and oracle_run.reward.reward in (1, 1.0)
    )
    null_ok = (
        null_run is not None and null_run.reward.parse_ok and null_run.reward.reward in (0, 0.0)
    )
    if not run_oracle:
        sol_ok = True  # not required this call
    if not run_null:
        null_ok = True
    # Mintag fail-closed forces reject even when oracle/null not run
    if not mintag_ok:
        sol_ok = False
        null_ok = False

    # Extra meta honesty for deepagent cert
    if not pack_meta.real_url_ok:
        reasons.append("repository_url not real public remote")
    if not pack_meta.real_sha_ok:
        reasons.append("base_commit_hash not real 40-char SHA")

    certified = (
        structural_ok
        and isol.clean
        and sol_ok
        and null_ok
        and mintag_ok
        and pack_meta.real_url_ok
        and pack_meta.real_sha_ok
        and backend == "docker"
        and (oracle_mode or "docker").strip().lower() not in _FAKE_MODE_TOKENS
        and not any(c in C.HARD_REJECT_CODES for c in codes)
        and "PIER_AGENT_MINTAG_FAIL" not in codes
    )
    if certified:
        codes.append("PIER_CERT_PASS")
    else:
        codes.append("PIER_CERT_REJECT")

    seen: set[str] = set()
    uniq_codes: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            uniq_codes.append(code)

    result = PierCertResult(
        certified=certified,
        disposition="accept" if certified else "reject",
        task_id=tid,
        pack_dir=str(root.resolve()),
        jobs_root=str(jobs),
        structural_ok=structural_ok,
        pier_ready=pier_ready,
        oracle_run=oracle_run,
        null_run=null_run,
        isolation=isol,
        pack_meta=pack_meta,
        backend=backend,
        oracle_mode=(oracle_mode or "docker").strip().lower(),
        reason_codes=tuple(uniq_codes),
        reasons=tuple(dict.fromkeys(reasons)),
        evidence={
            "sol_reward_path": oracle_run.reward.path if oracle_run else None,
            "null_reward_path": null_run.reward.path if null_run else None,
            "missing_relpaths": list(verify_pack_tree(root)),
            "meta_real_url": is_real_repository_url(pack_meta.repository_url),
            "meta_real_sha": is_real_base_sha(pack_meta.base_commit_hash),
            "sha40_ok": bool(_HEX40_RE.match(pack_meta.base_commit_hash or "")),
            "agent_mintag": agent_mintag,
            "mintag_ok": mintag_ok,
        },
    )
    if evidence_out is not None:
        write_pier_evidence(evidence_out, result)
    if audit_out is not None:
        append_pier_audit(audit_out, result)
    return result


__all__ = [
    "DEFAULT_JOBS_ROOT",
    "DEFAULT_PIER_BIN",
    "PierAgent",
    "PierCertDisposition",
    "PierCertError",
    "PierCertResult",
    "PierInvokeError",
    "PierRunEvidence",
    "PierRunner",
    "RewardEvidence",
    "ScriptedPierRunner",
    "SubprocessPierRunner",
    "append_pier_audit",
    "certify_pier_pack",
    "ensure_jobs_root",
    "find_latest_trial_dir",
    "find_reward_jsons",
    "load_trial_result_json",
    "parse_pier_job_result",
    "parse_reward_json",
    "refuse_fake_oracle_mode",
    "resolve_pier_bin",
    "structural_load_pack",
    "write_pier_evidence",
]
