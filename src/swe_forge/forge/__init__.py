"""SWE-Forge: configurable-teacher synthetic SWE benchmark generator.

The forge package builds synthetic, 100%-verifiable software-engineering tasks.
Its LLM access goes exclusively through an env-driven LiteLLM layer; it never
imports the repository's bespoke LLM clients or response cache, and no provider
hostname or brand string is hardcoded anywhere in this package.
"""

from swe_forge.forge.config import ForgeSettings
from swe_forge.forge.models import (
    GENERATOR_NAMES,
    ORACLE_VERDICTS,
    SUPPORTED_LANGUAGES,
    BaselineNotGreenError,
    Candidate,
    CandidateTarget,
    EnvImage,
    GeneratedSpec,
    InstanceGrant,
    ModelError,
    OracleReport,
    OracleTestFile,
    Provenance,
    RepoSpec,
    require_green_baseline,
)
from swe_forge.forge.oracle.pipeline import (
    GATE_ORDER,
    ExportRefusedError,
    OraclePipelineError,
    ensure_oracle_exportable,
    is_oracle_exportable,
    run_oracle_pipeline,
)
from swe_forge.forge.sources import (
    SourceError,
    SourceRegistry,
    UnknownRepoError,
    build_source_registry,
)

__all__ = [
    "GATE_ORDER",
    "GENERATOR_NAMES",
    "ORACLE_VERDICTS",
    "SUPPORTED_LANGUAGES",
    "BaselineNotGreenError",
    "Candidate",
    "CandidateTarget",
    "EnvImage",
    "ExportRefusedError",
    "ForgeSettings",
    "GeneratedSpec",
    "InstanceGrant",
    "ModelError",
    "OraclePipelineError",
    "OracleReport",
    "OracleTestFile",
    "Provenance",
    "RepoSpec",
    "SourceError",
    "SourceRegistry",
    "UnknownRepoError",
    "build_source_registry",
    "ensure_oracle_exportable",
    "is_oracle_exportable",
    "require_green_baseline",
    "run_oracle_pipeline",
]
