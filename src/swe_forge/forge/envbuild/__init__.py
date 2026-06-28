"""Stage 1: env-first Docker image build with a green-baseline gate.

Builds exactly one Docker image per repo via the language adapter (correct base,
pinned-SHA checkout, baseline install with test deps, the repo's own baseline
suite required GREEN), persists an :class:`~swe_forge.forge.models.EnvImage`
with the proven baseline command and green proof, and enforces that a green
baseline is a hard precondition for every downstream stage.
"""

from __future__ import annotations

from swe_forge.forge.envbuild.builder import (
    DockerCLI,
    EnvBuildError,
    EnvBuilder,
    EnvBuildResult,
    ExecOutcome,
)

__all__ = [
    "DockerCLI",
    "EnvBuildError",
    "EnvBuildResult",
    "EnvBuilder",
    "ExecOutcome",
]
