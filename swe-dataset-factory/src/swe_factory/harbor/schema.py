"""Harbor / DeepSWE schema 1.1-compatible models (task.toml + tests/config.json).

VAL-HARBOR-002: task.toml includes schema_version, artifacts model.patch path,
metadata.language/repository_url/base_commit_hash, verifier.environment_mode
separate (or Pier-compatible), agent/verifier timeouts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION_DEFAULT = "1.1"
MODEL_PATCH_ARTIFACT = "/logs/artifacts/model.patch"
EnvironmentMode = Literal["shared", "separate"]


def _nonempty(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    return cleaned


class HarborAuthor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = ""
    email: str = ""


class HarborTaskIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="org/task-id style name")
    description: str = ""
    authors: list[HarborAuthor] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_ok(cls, value: str) -> str:
        return _nonempty(value, "task.name")


class HarborMetadata(BaseModel):
    """DeepSWE-required metadata fields plus optional display helpers."""

    model_config = ConfigDict(extra="allow")

    language: str = Field(min_length=1)
    repository_url: str = Field(min_length=1)
    base_commit_hash: str = Field(min_length=1)
    task_id: str | None = None
    ext_id: str | None = None
    display_title: str | None = None
    display_description: str | None = None
    original_title: str | None = None
    category: str | None = None
    source_track: str | None = None
    license: str | None = None

    @field_validator("language", "repository_url", "base_commit_hash")
    @classmethod
    def _req(cls, value: str, info: Any) -> str:
        return _nonempty(value, info.field_name)


class HarborTimeoutSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_sec: float = Field(gt=0)


class HarborVerifierEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_timeout_sec: float = 1800.0
    cpus: int = 2
    memory_mb: int = 4096
    storage_mb: int = 10240
    allow_internet: bool = False
    docker_image: str | None = None


class HarborVerifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment_mode: EnvironmentMode = "separate"
    timeout_sec: float = Field(default=1800.0, gt=0)
    environment: HarborVerifierEnvironment = Field(default_factory=HarborVerifierEnvironment)


class HarborAgent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_sec: float = Field(default=5400.0, gt=0)


class HarborEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build_timeout_sec: float = 1800.0
    docker_image: str | None = None
    os: str = "linux"
    cpus: int = 2
    memory_mb: int = 4096
    storage_mb: int = 10240
    gpus: int = 0
    allow_internet: bool = False
    mcp_servers: list[Any] = Field(default_factory=list)


class HarborTaskToml(BaseModel):
    """In-memory representation of a DeepSWE/Harbor task.toml (schema 1.1+)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION_DEFAULT
    artifacts: list[str] = Field(
        default_factory=lambda: [MODEL_PATCH_ARTIFACT],
        min_length=1,
    )
    task: HarborTaskIdentity
    metadata: HarborMetadata
    verifier: HarborVerifier = Field(default_factory=HarborVerifier)
    agent: HarborAgent = Field(default_factory=HarborAgent)
    environment: HarborEnvironment = Field(default_factory=HarborEnvironment)

    @field_validator("schema_version")
    @classmethod
    def _schema_ok(cls, value: str) -> str:
        cleaned = _nonempty(value, "schema_version")
        # Accept 1.1 and forward-compatible 1.x strings
        if not cleaned.startswith("1."):
            raise ValueError(f"schema_version must be 1.x-compatible; got {value!r}")
        return cleaned

    @field_validator("artifacts")
    @classmethod
    def _artifacts_ok(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("artifacts must be non-empty")
        if MODEL_PATCH_ARTIFACT not in cleaned:
            raise ValueError(f"artifacts must include model.patch path {MODEL_PATCH_ARTIFACT!r}")
        return cleaned

    def ensure_required(self) -> None:
        """Fail-closed completeness check for VAL-HARBOR-002 fields."""
        if self.verifier.environment_mode != "separate":
            raise ValueError("verifier.environment_mode must be 'separate' for DeepSWE packs")
        if self.agent.timeout_sec <= 0 or self.verifier.timeout_sec <= 0:
            raise ValueError("agent/verifier timeout_sec must be positive")
        md = self.metadata
        for field in ("language", "repository_url", "base_commit_hash"):
            if not str(getattr(md, field, "")).strip():
                raise ValueError(f"metadata.{field} is required")


class GradeConfig(BaseModel):
    """tests/config.json nested grade block."""

    model_config = ConfigDict(extra="allow")

    format: Literal["junit", "ctrf"] = "junit"
    node_id: str = "name"
    tool_label: str = "pytest"
    reports: list[str] = Field(
        default_factory=lambda: ["/logs/verifier/new.xml", "/logs/verifier/base.xml"]
    )


class TestsConfig(BaseModel):
    """DeepSWE tests/config.json: f2p/p2p node ids + grade recipe."""

    # Prevent pytest from treating this Pydantic model as a test class.
    __test__ = False

    model_config = ConfigDict(extra="forbid")

    base_commit: str = Field(min_length=1)
    f2p_node_ids: list[str] = Field(min_length=1)
    p2p_node_ids: list[str] = Field(default_factory=list)
    grade: GradeConfig = Field(default_factory=GradeConfig)

    @field_validator("base_commit")
    @classmethod
    def _base(cls, value: str) -> str:
        return _nonempty(value, "base_commit")

    @field_validator("f2p_node_ids")
    @classmethod
    def _f2p(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("f2p_node_ids must contain at least one node id")
        return cleaned

    @field_validator("p2p_node_ids")
    @classmethod
    def _p2p(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if str(item).strip()]


class HarborPackSpec(BaseModel):
    """Full inputs required to emit one Harbor pack directory tree."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    instruction_md: str = Field(min_length=1)
    task_toml: HarborTaskToml
    tests_config: TestsConfig
    solution_patch: str = Field(min_length=1)
    test_patch: str = Field(min_length=1)
    environment_dockerfile: str = Field(min_length=1)
    tests_dockerfile: str = Field(min_length=1)
    # Optional overrides; defaults rendered by emitters
    test_sh: str | None = None
    solve_sh: str | None = None
    grader_py: str | None = None
    pre_artifacts_sh: str | None = None

    @field_validator("task_id", "instruction_md", "solution_patch", "test_patch")
    @classmethod
    def _text(cls, value: str, info: Any) -> str:
        field = info.field_name
        if field in {"solution_patch", "test_patch"}:
            if not value.strip():
                raise ValueError(f"{field} must be non-empty")
            return value
        return _nonempty(value, field)

    def ensure_consistent(self) -> None:
        self.task_toml.ensure_required()
        md = self.task_toml.metadata
        if md.base_commit_hash != self.tests_config.base_commit:
            raise ValueError("metadata.base_commit_hash must match tests_config.base_commit")


def validate_pack_spec(spec: HarborPackSpec) -> HarborPackSpec:
    """Validate and normalize a pack spec (raises ValueError on failure)."""
    # Re-validate through pydantic + ensure_required
    cleaned = HarborPackSpec.model_validate(spec.model_dump())
    cleaned.task_toml.ensure_required()
    if cleaned.task_toml.metadata.base_commit_hash != cleaned.tests_config.base_commit:
        raise ValueError("metadata.base_commit_hash must match tests_config.base_commit")
    # Ensure task_id present on metadata for inspectability
    if not cleaned.task_toml.metadata.task_id:
        cleaned = cleaned.model_copy(
            update={
                "task_toml": cleaned.task_toml.model_copy(
                    update={
                        "metadata": cleaned.task_toml.metadata.model_copy(
                            update={"task_id": cleaned.task_id}
                        )
                    }
                )
            }
        )
    if not cleaned.solution_patch.strip() or not cleaned.test_patch.strip():
        raise ValueError("solution_patch and test_patch must be non-empty")
    return cleaned


def render_task_toml(task: HarborTaskToml) -> str:
    """Serialize HarborTaskToml to schema-1.1-compatible TOML text."""
    task.ensure_required()
    lines: list[str] = []
    lines.append(f'schema_version = "{_esc(task.schema_version)}"')
    arts = ", ".join(f'"{_esc(a)}"' for a in task.artifacts)
    lines.append(f"artifacts = [{arts}]")
    lines.append("[task]")
    lines.append(f'name = "{_esc(task.task.name)}"')
    lines.append(f'description = "{_esc(task.task.description)}"')
    if task.task.authors:
        authors_repr = ", ".join(
            f'{{ name = "{_esc(a.name)}", email = "{_esc(a.email)}" }}' for a in task.task.authors
        )
        lines.append(f"authors = [{authors_repr}]")
    else:
        lines.append("authors = []")
    if task.task.keywords:
        kws = ", ".join(f'"{_esc(k)}"' for k in task.task.keywords)
        lines.append(f"keywords = [{kws}]")
    else:
        lines.append("keywords = []")

    lines.append("[metadata]")
    md = task.metadata
    # Stable required first
    for key in (
        "ext_id",
        "task_id",
        "display_title",
        "display_description",
        "original_title",
        "category",
        "language",
        "repository_url",
        "base_commit_hash",
        "source_track",
        "license",
    ):
        val = getattr(md, key, None)
        if val is None or val == "":
            continue
        lines.append(f'{key} = "{_esc(str(val))}"')
    # Any extra metadata fields
    extras = md.model_extra or {}
    for key, val in sorted(extras.items()):
        if val is None:
            continue
        if isinstance(val, bool):
            lines.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, int | float):
            lines.append(f"{key} = {val}")
        else:
            lines.append(f'{key} = "{_esc(str(val))}"')

    lines.append("[verifier]")
    lines.append(f'environment_mode = "{task.verifier.environment_mode}"')
    lines.append(f"timeout_sec = {float(task.verifier.timeout_sec)}")
    lines.append("")
    lines.append("[verifier.env]")
    lines.append("[verifier.environment]")
    ve = task.verifier.environment
    lines.append(f"build_timeout_sec = {float(ve.build_timeout_sec)}")
    lines.append(f"cpus = {int(ve.cpus)}")
    lines.append(f"memory_mb = {int(ve.memory_mb)}")
    lines.append(f"storage_mb = {int(ve.storage_mb)}")
    lines.append(f"allow_internet = {'true' if ve.allow_internet else 'false'}")
    if ve.docker_image:
        lines.append(f'docker_image = "{_esc(ve.docker_image)}"')

    lines.append("")
    lines.append("[agent]")
    lines.append(f"timeout_sec = {float(task.agent.timeout_sec)}")

    lines.append("[environment]")
    env = task.environment
    lines.append(f"build_timeout_sec = {float(env.build_timeout_sec)}")
    if env.docker_image:
        lines.append(f'docker_image = "{_esc(env.docker_image)}"')
    lines.append(f'os = "{_esc(env.os)}"')
    lines.append(f"cpus = {int(env.cpus)}")
    lines.append(f"memory_mb = {int(env.memory_mb)}")
    lines.append(f"storage_mb = {int(env.storage_mb)}")
    lines.append(f"gpus = {int(env.gpus)}")
    lines.append(f"allow_internet = {'true' if env.allow_internet else 'false'}")
    lines.append("mcp_servers = []")
    lines.append("")
    lines.append("[environment.env]")
    lines.append("[solution.env]")
    lines.append("")
    return "\n".join(lines)


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "MODEL_PATCH_ARTIFACT",
    "SCHEMA_VERSION_DEFAULT",
    "GradeConfig",
    "HarborAgent",
    "HarborEnvironment",
    "HarborMetadata",
    "HarborPackSpec",
    "HarborTaskIdentity",
    "HarborTaskToml",
    "HarborVerifier",
    "HarborVerifierEnvironment",
    "TestsConfig",
    "render_task_toml",
    "validate_pack_spec",
]
