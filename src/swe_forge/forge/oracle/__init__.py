"""Stage 3 oracle hardening: gates that make a task "100% verifiable".

Each gate runs in a throwaway :class:`~swe_forge.execution.sandbox.DockerSandbox`
on the candidate's :class:`~swe_forge.forge.models.EnvImage` and contributes
evidence to an :class:`~swe_forge.forge.models.OracleReport`. This package starts
with the *establish* gate (:mod:`swe_forge.forge.oracle.establish`), which owns
the reusable Docker FAIL->PASS (F2P) / PASS->PASS (P2P) execution recipe consumed
by the later gates and by calibration.
"""

from swe_forge.forge.oracle.establish import (
    DockerOracleRecipe,
    EstablishError,
    EstablishOutcome,
    HiddenTest,
    HiddenTestFile,
    HiddenTestSynthesizer,
    NullSynthesizer,
    RecipeProtocol,
    SandboxProtocol,
    SynthesisContext,
    TestRun,
    TreeState,
    build_establish_report,
    establish_oracle,
    run_establish_gate,
)

__all__ = [
    "DockerOracleRecipe",
    "EstablishError",
    "EstablishOutcome",
    "HiddenTest",
    "HiddenTestFile",
    "HiddenTestSynthesizer",
    "NullSynthesizer",
    "RecipeProtocol",
    "SandboxProtocol",
    "SynthesisContext",
    "TestRun",
    "TreeState",
    "build_establish_report",
    "establish_oracle",
    "run_establish_gate",
]
