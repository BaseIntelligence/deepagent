"""TaskRecord pydantic schema for SWE Dataset Factory export/oracle records.

Required fields match VAL-SKEL-003 / mission data schema:
instance_id, source_track, repo, base_commit, language, problem_statement,
fail_to_pass, pass_to_pass, gold_patch (hidden from agents), environment.image_digest,
license. Optional panel hardness fields and created_at for certified keeps.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceTrack(StrEnum):
    """Provenance label; tracks must never be silently mixed."""

    REAL_PR = "real_pr"
    SYNTHETIC_GROUNDED = "synthetic_grounded"


VALID_SOURCE_TRACKS: frozenset[str] = frozenset(t.value for t in SourceTrack)

Language = Literal["python", "javascript", "typescript", "go", "js", "ts"]


def _nonempty_str(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


class EnvironmentMeta(BaseModel):
    """Docker environment pin for a task (reproducible image digest)."""

    model_config = ConfigDict(extra="forbid")

    image_digest: str = Field(
        min_length=1,
        description="Pinned Docker image digest or immutable reference",
    )

    @field_validator("image_digest")
    @classmethod
    def _digest_nonempty(cls, value: str) -> str:
        return _nonempty_str(value, "image_digest")


class PanelHardness(BaseModel):
    """Optional frontier hardness calibration fields for certified keeps."""

    model_config = ConfigDict(extra="forbid")

    grok_4_5: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="pass@k for x-ai/grok-4.5 panel model",
    )
    kimi_k2_6: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="pass@k for moonshotai/kimi-k2.6 panel model",
    )
    opus_4_8: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="pass@k for anthropic/claude-opus-4.8 panel model",
    )
    pass_at_k: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Frontier aggregate pass@k",
    )
    discrimination: float | None = Field(
        default=None,
        ge=0.0,
        description="IRT-style or score discrimination",
    )


class TaskRecord(BaseModel):
    """Full certified (or candidate) task record.

    Incomplete records are rejected: gold_patch, fail_to_pass, base_commit,
    source_track, environment.image_digest, and license are required and
    non-empty. ``gold_patch`` is part of the internal record for oracle/export
    and is marked hidden from agent-visible workspaces.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    instance_id: str = Field(min_length=1)
    source_track: SourceTrack
    repo: str = Field(min_length=1, description="owner/name or clone identity")
    base_commit: str = Field(min_length=1, description="Pinned git commit SHA")
    language: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    fail_to_pass: list[str] = Field(min_length=1)
    pass_to_pass: list[str] = Field(default_factory=list)
    gold_patch: str = Field(
        min_length=1,
        description="Gold solution patch; hidden from agent mounts",
        json_schema_extra={"hidden": True, "x_hidden_from_agent": True},
    )
    environment: EnvironmentMeta
    license: str = Field(min_length=1)
    requirements: str | None = None
    panel: PanelHardness | None = None
    gate_proof: dict[str, Any] | None = None
    created_at: datetime | None = None

    @field_validator("instance_id", "repo", "problem_statement", "license", "language")
    @classmethod
    def _required_text(cls, value: str, info: Any) -> str:
        return _nonempty_str(value, info.field_name)

    @field_validator("base_commit")
    @classmethod
    def _base_commit_nonempty(cls, value: str) -> str:
        return _nonempty_str(value, "base_commit")

    @field_validator("gold_patch")
    @classmethod
    def _gold_nonempty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("gold_patch must be non-empty")
        return value  # preserve patch formatting (may trailing-newline matter)

    @field_validator("fail_to_pass")
    @classmethod
    def _f2p_nonempty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("fail_to_pass must contain at least one non-empty command")
        return cleaned

    @field_validator("pass_to_pass")
    @classmethod
    def _p2p_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if str(item).strip()]

    @field_validator("source_track", mode="before")
    @classmethod
    def _coerce_source_track(cls, value: object) -> object:
        if isinstance(value, SourceTrack):
            return value
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned not in VALID_SOURCE_TRACKS:
                raise ValueError(
                    f"source_track must be one of {sorted(VALID_SOURCE_TRACKS)}; got {value!r}"
                )
            return cleaned
        raise ValueError(
            f"source_track must be one of {sorted(VALID_SOURCE_TRACKS)}; got {value!r}"
        )


# Back-compat alias used by skeleton CLI tests during transition.
TaskRecordStub = TaskRecord

__all__ = [
    "EnvironmentMeta",
    "PanelHardness",
    "SourceTrack",
    "TaskRecord",
    "TaskRecordStub",
    "VALID_SOURCE_TRACKS",
]
