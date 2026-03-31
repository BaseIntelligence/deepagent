"""Orchestrator module for managing the test generation pipeline.

This module provides the orchestration layer that coordinates the
full lifecycle of test generation, validation, Docker building,
verification, repair, scoring, and publishing.

Pipeline stages (in order):
1. Generate Tests - Create test specifications via agentic loop
2. Validate Tests - Verify test quality and syntax
3. Build Docker - Containerize the test environment
4. Verify - Run tests before/after patch
5. Repair - Fix failing tests if needed
6. Score - Calculate quality metrics
7. Publish - Export to dataset

Usage:
    from swe_forge.orchestrator import OrchestratorTask, TaskState
    from swe_forge.orchestrator.models import GenerateTestsResult
"""

from .dataset_orchestrator import DatasetOrchestrator
from .full_pipeline_orchestrator import FullPipelineOrchestrator
from .github_source_orchestrator import (
    GitHubRepo,
    GitHubSourceOrchestrator,
    PreFilterConfig,
)
from .master_orchestrator import MasterOrchestrator
from .models import (
    BuildDockerResult,
    GenerateTestsResult,
    OrchestratorStats,
    OrchestratorTask,
    PublishResult,
    RepairResult,
    ScoreResult,
    TaskState,
    TestFile,
    ValidateTestsResult,
    VerifyResult,
)
from .tools import (
    build_docker,
    generate_tests,
    publish_task,
    reject_task,
    repair_test,
    score_task,
    validate_tests,
    verify_fail_to_pass,
)

__all__ = [
    "DatasetOrchestrator",
    "FullPipelineOrchestrator",
    "MasterOrchestrator",
    "GitHubRepo",
    "GitHubSourceOrchestrator",
    "PreFilterConfig",
    "BuildDockerResult",
    "GenerateTestsResult",
    "OrchestratorStats",
    "OrchestratorTask",
    "PublishResult",
    "RepairResult",
    "ScoreResult",
    "TaskState",
    "TestFile",
    "ValidateTestsResult",
    "VerifyResult",
    "build_docker",
    "generate_tests",
    "publish_task",
    "reject_task",
    "repair_test",
    "score_task",
    "validate_tests",
    "verify_fail_to_pass",
]
