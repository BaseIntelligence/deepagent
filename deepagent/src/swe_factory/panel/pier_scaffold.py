"""Pier / mini-swe-agent scaffold adapter for DeepAgent hardness panel.

VAL-DPANEL-007: certified keep hardness trials execute on the Pier-loadable
pack tree via the mini-swe-agent (or successor Pier agent) scaffold with fixed
scaffold metadata — not an ad-hoc non-Pier harness that cannot load the
exported Harbor tree.

This module:

- builds Pier ``JobConfig``-compatible dicts targeting ``mini-swe-agent``
- records pack path + scaffold/runtime identity on panel result records
- provides an offline dry-run path (no live pier/docker) for unit tests
- optionally shells out to ``pier run -a mini-swe-agent`` when authorized

Oracle Pier cert (VAL-PIER-*) remains separate from this **panel agent** path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from swe_factory.panel.runner import (
    PANEL_SCAFFOLD_AGENT,
    PANEL_SCAFFOLD_NAME,
    PANEL_SCAFFOLD_RUNTIME,
    PANEL_SCAFFOLD_VERSION,
    PanelScaffoldMeta,
    build_panel_scaffold_meta,
)

# Re-export for tests / consumers that import from pier_scaffold directly.
__scaffold_consts__ = (PANEL_SCAFFOLD_AGENT, PANEL_SCAFFOLD_NAME, PANEL_SCAFFOLD_RUNTIME)

DEFAULT_PIER_BIN = Path("/tmp/pier-venv/bin/pier")
DEFAULT_JOBS_ROOT = Path("/tmp/harbor-deepagent-panel-jobs")
PanelScaffoldMode = Literal["dry-run", "pier-invoke"]


class PierScaffoldError(RuntimeError):
    """Unrecoverable Pier panel-scaffold failure."""


@dataclass(frozen=True, slots=True)
class PierPanelJobSpec:
    """Pier job specification for one keep × one model panel trial."""

    pack_path: str
    pack_id: str
    model: str
    agent: str = PANEL_SCAFFOLD_AGENT
    runtime: str = PANEL_SCAFFOLD_RUNTIME
    scaffold: str = PANEL_SCAFFOLD_NAME
    jobs_dir: str = str(DEFAULT_JOBS_ROOT)
    n_attempts: int = 1
    n_concurrent: int = 1
    timeout_multiplier: float = 1.0
    openrouter_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_path": self.pack_path,
            "pack_id": self.pack_id,
            "model": self.model,
            "agent": self.agent,
            "runtime": self.runtime,
            "scaffold": self.scaffold,
            "jobs_dir": self.jobs_dir,
            "n_attempts": self.n_attempts,
            "n_concurrent": self.n_concurrent,
            "timeout_multiplier": self.timeout_multiplier,
            "openrouter_model": self.openrouter_model or self.model,
        }

    def to_job_config(self) -> dict[str, Any]:
        """Minimal Pier JobConfig-shaped dict (docs / dry-run / call prep)."""
        return {
            "jobs_dir": self.jobs_dir,
            "n_attempts": self.n_attempts,
            "n_concurrent_trials": self.n_concurrent,
            "timeout_multiplier": self.timeout_multiplier,
            "agent": {
                "name": self.agent,
                "model_name": self.openrouter_model or self.model,
            },
            "tasks": [
                {
                    "path": self.pack_path,
                    "task_id": self.pack_id,
                }
            ],
            "environment": {
                "type": "docker",
            },
            "metadata": {
                "scaffold": self.scaffold,
                "runtime": self.runtime,
                "panel_model": self.model,
                "scaffold_version": PANEL_SCAFFOLD_VERSION,
            },
        }


@dataclass(frozen=True, slots=True)
class PierPanelInvocation:
    """Record of one Pier/mini-swe panel scaffold invocation (live or dry)."""

    ok: bool
    mode: str
    spec: PierPanelJobSpec
    scaffold_meta: PanelScaffoldMeta
    command: tuple[str, ...] = ()
    exit_code: int | None = None
    job_dir: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    dry_run_config: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    reward: float | int | None = None
    invented_reward: bool = False  # always False; never fabricate

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "spec": self.spec.to_dict(),
            "scaffold_meta": self.scaffold_meta.to_dict(),
            "command": list(self.command),
            "exit_code": self.exit_code,
            "job_dir": self.job_dir,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "dry_run_config": dict(self.dry_run_config),
            "errors": list(self.errors),
            "reward": self.reward,
            "invented_reward": self.invented_reward,
        }


def resolve_pier_bin(explicit: str | Path | None = None) -> Path:
    """Locate pier binary (explicit → env → default /tmp pier-venv → PATH)."""
    if explicit is not None:
        p = Path(explicit)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        raise PierScaffoldError(f"pier binary not executable: {p}")
    env = os.environ.get("PIER_BIN") or os.environ.get("FACTORY_PIER_BIN")
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    if DEFAULT_PIER_BIN.is_file() and os.access(DEFAULT_PIER_BIN, os.X_OK):
        return DEFAULT_PIER_BIN
    which = shutil.which("pier")
    if which:
        return Path(which)
    raise PierScaffoldError("pier binary not found (set PIER_BIN or install via /tmp/pier-venv)")


def build_panel_job_spec(
    *,
    pack_path: str | Path,
    pack_id: str,
    model: str,
    jobs_dir: str | Path = DEFAULT_JOBS_ROOT,
    n_attempts: int = 1,
) -> PierPanelJobSpec:
    """Build a Pier/mini-swe panel job for one pack × model."""
    path = Path(pack_path)
    if not str(path):
        raise PierScaffoldError("pack_path must be non-empty")
    pid = pack_id.strip()
    if not pid:
        raise PierScaffoldError("pack_id must be non-empty")
    mid = model.strip()
    if not mid:
        raise PierScaffoldError("model must be non-empty")
    return PierPanelJobSpec(
        pack_path=str(path.resolve()) if path.exists() else str(path),
        pack_id=pid,
        model=mid,
        agent=PANEL_SCAFFOLD_AGENT,
        runtime=PANEL_SCAFFOLD_RUNTIME,
        scaffold=PANEL_SCAFFOLD_NAME,
        jobs_dir=str(jobs_dir),
        n_attempts=n_attempts,
        openrouter_model=mid,
    )


def dry_run_panel_scaffold(
    *,
    pack_path: str | Path,
    pack_id: str,
    model: str,
    jobs_dir: str | Path = DEFAULT_JOBS_ROOT,
) -> PierPanelInvocation:
    """Offline Pier/mini-swe scaffold preparation (no pier process, no reward invent).

    Validates pack path presence when on disk, builds JobConfig, returns meta
    that panel reports must include (scaffold/runtime/pack id).
    """
    spec = build_panel_job_spec(
        pack_path=pack_path,
        pack_id=pack_id,
        model=model,
        jobs_dir=jobs_dir,
    )
    meta = build_panel_scaffold_meta(pack_path=spec.pack_path, pack_id=spec.pack_id)
    config = spec.to_job_config()
    path = Path(spec.pack_path)
    errors: list[str] = []
    ok = True
    if not path.exists():
        # Not fatal for pure unit scaffolding when pack is virtual.
        errors.append(f"pack path not on disk (dry-run allows): {path}")
        ok = True  # dry-run still constructs scaffold evidence

    return PierPanelInvocation(
        ok=ok,
        mode="dry-run",
        spec=spec,
        scaffold_meta=meta,
        command=(
            "pier",
            "run",
            "-a",
            PANEL_SCAFFOLD_AGENT,
            "--jobs-dir",
            spec.jobs_dir,
            # Real invoke would also pass config path; dry-run records intent.
        ),
        exit_code=0,
        job_dir=None,
        dry_run_config=config,
        errors=tuple(errors),
        reward=None,
        invented_reward=False,
    )


def invoke_pier_mini_swe_panel(
    *,
    pack_path: str | Path,
    pack_id: str,
    model: str,
    jobs_dir: str | Path | None = None,
    pier_bin: str | Path | None = None,
    timeout_s: float = 600.0,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> PierPanelInvocation:
    """Invoke Pier mini-swe-agent on a Harbor pack for hardness panel.

    When ``dry_run=True`` (default offline / tests), no process is started and
    no reward is invented. Live pier is opt-in for authorized panel features.
    """
    jobs = Path(jobs_dir) if jobs_dir is not None else DEFAULT_JOBS_ROOT
    jobs.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return dry_run_panel_scaffold(
            pack_path=pack_path,
            pack_id=pack_id,
            model=model,
            jobs_dir=jobs,
        )

    spec = build_panel_job_spec(
        pack_path=pack_path,
        pack_id=pack_id,
        model=model,
        jobs_dir=jobs,
    )
    meta = build_panel_scaffold_meta(pack_path=spec.pack_path, pack_id=spec.pack_id)
    try:
        bin_path = resolve_pier_bin(pier_bin)
    except PierScaffoldError as exc:
        return PierPanelInvocation(
            ok=False,
            mode="pier-invoke",
            spec=spec,
            scaffold_meta=meta,
            errors=(str(exc),),
            invented_reward=False,
        )

    job_name = f"panel-{pack_id}-{model.replace('/', '_')}-{int(time.time())}"
    config_path = jobs / f"{job_name}.job.json"
    config_path.write_text(
        json.dumps(spec.to_job_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cmd = (
        str(bin_path),
        "run",
        "-a",
        PANEL_SCAFFOLD_AGENT,
        "-c",
        str(config_path),
        "-o",
        str(jobs),
        "--job-name",
        job_name,
        "-n",
        "1",
        "-y",
    )
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    # Model routing for mini-swe / OpenRouter when agent respects these.
    run_env.setdefault("MSWEA_MODEL_NAME", model)
    run_env.setdefault("OPENROUTER_MODEL", model)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=run_env,
        )
    except subprocess.TimeoutExpired as exc:
        return PierPanelInvocation(
            ok=False,
            mode="pier-invoke",
            spec=spec,
            scaffold_meta=meta,
            command=cmd,
            exit_code=None,
            job_dir=str(jobs / job_name),
            stdout_tail=(exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            stderr_tail=(exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            dry_run_config=spec.to_job_config(),
            errors=(f"pier timeout after {timeout_s}s",),
            invented_reward=False,
        )
    except OSError as exc:
        return PierPanelInvocation(
            ok=False,
            mode="pier-invoke",
            spec=spec,
            scaffold_meta=meta,
            command=cmd,
            errors=(f"pier spawn failed: {exc}",),
            invented_reward=False,
        )

    job_dir = jobs / job_name
    # Never invent reward: leave None unless a real reward.json is found and parsed.
    reward: float | int | None = None
    reward_path = None
    if job_dir.is_dir():
        for candidate in job_dir.rglob("reward.json"):
            reward_path = candidate
            break
    errors: list[str] = []
    if reward_path and reward_path.is_file():
        try:
            raw = json.loads(reward_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "reward" in raw:
                val = raw["reward"]
                if isinstance(val, int | float):
                    reward = val
                else:
                    errors.append("reward.json present but reward field not numeric")
            else:
                errors.append("reward.json present but missing reward field")
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"reward.json parse failed: {exc}")

    ok = proc.returncode == 0
    if not ok:
        errors.append(f"pier exit_code={proc.returncode}")

    return PierPanelInvocation(
        ok=ok,
        mode="pier-invoke",
        spec=spec,
        scaffold_meta=meta,
        command=cmd,
        exit_code=proc.returncode,
        job_dir=str(job_dir) if job_dir.exists() else None,
        stdout_tail=(proc.stdout or "")[-2000:],
        stderr_tail=(proc.stderr or "")[-2000:],
        dry_run_config=spec.to_job_config(),
        errors=tuple(errors),
        reward=reward,
        invented_reward=False,
    )


__all__ = [
    "DEFAULT_JOBS_ROOT",
    "DEFAULT_PIER_BIN",
    "PierPanelInvocation",
    "PierPanelJobSpec",
    "PierScaffoldError",
    "build_panel_job_spec",
    "dry_run_panel_scaffold",
    "invoke_pier_mini_swe_panel",
    "resolve_pier_bin",
]
