"""Environment image @recipe models for envbuild stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EnvRecipe:
    """Inputs for building a reproducible env image for (repo, base_commit)."""

    repo_id: str
    base_commit: str
    language: str = "python"
    base_image: str = "python:3.12-slim"
    install_commands: list[str] = field(default_factory=lambda: ["pip install -q pytest"])
    baseline_test_command: str = "python -m pytest -q"
    workspace_dir: str = "/workspace/repo"
    # When set, use an existing local tree (fixture) instead of clone URL.
    local_path: str | None = None
    # Optional git clone URL (owner/name form or https URL) when not local.
    clone_url: str | None = None
    # DeepSWE real-SHA path: refuse placeholder/synthetic base commits.
    require_real_sha: bool = False
    # Image namespace prefix for tagging (sdf-env / deepswe-env / harbor-sdf).
    image_namespace: str = "sdf-env"
    # Document Harbor allow_internet=false runtime contract on metadata.
    allow_internet: bool = False
    # Scrub remotes/reflogs and disable hooks after checkout (agent isolation).
    history_scrub: bool = True
    hooks_off: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnvImage:
    """Persisted green-baseline image metadata (VAL-ENV-001 / VAL-ENV-002)."""

    repo_id: str
    base_commit: str
    language: str
    image_tag: str
    image_digest: str
    base_image: str
    workspace_dir: str
    install_commands: list[str]
    baseline_test_command: str
    baseline_green: bool
    baseline_exit_code: int
    dual_build_verified: bool = False
    dual_build_digests: list[str] = field(default_factory=list)
    baseline_summary: str = ""
    built_at: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    # Runtime contract + isolation bookkeeping (DeepSWE VAL-ENVR-*).
    allow_internet: bool = False
    resolved_head: str = ""
    porcelain_clean: bool = True
    hooks_path: str = "/dev/null"
    history_scrubbed: bool = False
    isolation_clean: bool = True
    dockerfile_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvImage:
        return cls(
            repo_id=str(data["repo_id"]),
            base_commit=str(data["base_commit"]),
            language=str(data["language"]),
            image_tag=str(data["image_tag"]),
            image_digest=str(data["image_digest"]),
            base_image=str(data["base_image"]),
            workspace_dir=str(data["workspace_dir"]),
            install_commands=list(data.get("install_commands") or []),
            baseline_test_command=str(data["baseline_test_command"]),
            baseline_green=bool(data["baseline_green"]),
            baseline_exit_code=int(data["baseline_exit_code"]),
            dual_build_verified=bool(data.get("dual_build_verified", False)),
            dual_build_digests=list(data.get("dual_build_digests") or []),
            baseline_summary=str(data.get("baseline_summary") or ""),
            built_at=str(data.get("built_at") or ""),
            provenance=dict(data.get("provenance") or {}),
            allow_internet=bool(data.get("allow_internet", False)),
            resolved_head=str(data.get("resolved_head") or ""),
            porcelain_clean=bool(data.get("porcelain_clean", True)),
            hooks_path=str(data.get("hooks_path") or "/dev/null"),
            history_scrubbed=bool(data.get("history_scrubbed", False)),
            isolation_clean=bool(data.get("isolation_clean", True)),
            dockerfile_excerpt=str(data.get("dockerfile_excerpt") or ""),
        )

    def environment_meta(self) -> dict[str, str]:
        """TaskRecord-compatible environment metadata."""
        return {"image_digest": self.image_digest}


__all__ = ["EnvImage", "EnvRecipe"]
