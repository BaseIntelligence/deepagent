"""Docker test harness for verifying patches with before/after test execution."""

from .harness import DockerTestHarness, TestRunResult
from .verification import verify_patch_fixes_issue, VerificationResult
from .image_builder import (
    BuildResult,
    build_task_image,
    build_images_for_tasks,
    generate_dockerfile,
    task_to_dict,
)

__all__ = [
    "DockerTestHarness",
    "TestRunResult",
    "verify_patch_fixes_issue",
    "VerificationResult",
    "BuildResult",
    "build_task_image",
    "build_images_for_tasks",
    "generate_dockerfile",
    "task_to_dict",
]
