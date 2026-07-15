"""Envbuild: pin base_commit, build Docker image, green baseline, digest metadata.

Always names containers under ``sdf-*`` / ``deepswe-*`` / ``harbor-sdf-*`` and
removes them on exit. Off-limits containers are never touched.

DeepSWE real-SHA path (VAL-ENVR-001..008):
- real 40-char base SHA pin + HEAD match
- runtime offline ``allow_internet=false`` recipe docs
- dual-build optional
- porcelain + hooks off
- history scrub / isolation
- multi-lang recipes (python/go/javascript/typescript/rust)
- disk gate + owned image prune
"""

from __future__ import annotations

from swe_factory.envbuild.agent_recipe import (
    ALLOW_INTERNET_FALSE,
    SUPPORTED_RECIPE_LANGUAGES,
    AgentRecipeContract,
    RealPrDockerfileError,
    agent_dockerfile_bakes_held_out_tests,
    agent_recipe_isolates_held_out_tests,
    assert_real_pr_agent_dockerfile,
    base_image_for_language,
    default_agent_contract,
    default_baseline_test_command,
    default_install_commands,
    dockerfile_has_clone_at_sha,
    looks_motor_fixture_copy,
    normalize_recipe_language,
    render_agent_dockerfile,
    render_real_pr_agent_dockerfile,
    render_task_toml_env_snippet,
)
from swe_factory.envbuild.builder import (
    BASELINE_FAILED,
    CHECKOUT_FAILED,
    DISK_FAILED,
    ISOLATION_FAILED,
    SHA_FAILED,
    DockerCLI,
    EnvBuilder,
    EnvBuildError,
    EnvBuildResult,
    ExecOutcome,
    dual_build,
    prune_owned_env_images,
    remove_leftover_sdf_containers,
)
from swe_factory.envbuild.fixture import (
    language_recipe_table,
    recipe_for_language,
    recipe_from_clone,
    recipe_from_go_fixture,
    recipe_from_green_fixture,
    recipe_from_offline_broken_fixture,
    recipe_from_rust_defaults,
    recipe_from_ts_fixture,
)
from swe_factory.envbuild.hygiene import (
    CONCURRENCY_HINT,
    DEFAULT_MIN_FREE_DISK_BYTES,
    MAX_CONCURRENT_ENVBUILD_JOBS,
    HygieneError,
    check_disk_for_envbuild,
    is_off_limits_name,
    is_owned_container_name,
    is_owned_image_ref,
    prune_owned_images,
    require_disk_for_envbuild,
)
from swe_factory.envbuild.models import EnvImage, EnvRecipe
from swe_factory.envbuild.parallel import (
    DEFAULT_ENVBUILD_WORKERS,
    HARD_MAX_ENVBUILD_WORKERS,
    ParallelEnvJob,
    ParallelEnvReport,
    ParallelEnvResult,
    clamp_envbuild_workers,
    parallel_envbuild,
    parallel_envbuild_recipes,
)
from swe_factory.envbuild.sha import (
    BaseCommitError,
    is_full_sha,
    isolation_scan,
    looks_synthetic_sha,
    require_full_sha,
)

__all__ = [
    "ALLOW_INTERNET_FALSE",
    "BASELINE_FAILED",
    "CHECKOUT_FAILED",
    "CONCURRENCY_HINT",
    "DEFAULT_ENVBUILD_WORKERS",
    "DEFAULT_MIN_FREE_DISK_BYTES",
    "DISK_FAILED",
    "DockerCLI",
    "EnvBuildError",
    "EnvBuildResult",
    "EnvBuilder",
    "EnvImage",
    "EnvRecipe",
    "ExecOutcome",
    "HARD_MAX_ENVBUILD_WORKERS",
    "HygieneError",
    "ISOLATION_FAILED",
    "MAX_CONCURRENT_ENVBUILD_JOBS",
    "AgentRecipeContract",
    "BaseCommitError",
    "ParallelEnvJob",
    "ParallelEnvReport",
    "ParallelEnvResult",
    "RealPrDockerfileError",
    "SHA_FAILED",
    "SUPPORTED_RECIPE_LANGUAGES",
    "agent_dockerfile_bakes_held_out_tests",
    "agent_recipe_isolates_held_out_tests",
    "assert_real_pr_agent_dockerfile",
    "base_image_for_language",
    "check_disk_for_envbuild",
    "clamp_envbuild_workers",
    "default_agent_contract",
    "default_baseline_test_command",
    "default_install_commands",
    "dockerfile_has_clone_at_sha",
    "dual_build",
    "is_full_sha",
    "is_off_limits_name",
    "is_owned_container_name",
    "is_owned_image_ref",
    "isolation_scan",
    "language_recipe_table",
    "looks_motor_fixture_copy",
    "looks_synthetic_sha",
    "normalize_recipe_language",
    "parallel_envbuild",
    "parallel_envbuild_recipes",
    "prune_owned_env_images",
    "prune_owned_images",
    "recipe_for_language",
    "recipe_from_clone",
    "recipe_from_go_fixture",
    "recipe_from_green_fixture",
    "recipe_from_offline_broken_fixture",
    "recipe_from_rust_defaults",
    "recipe_from_ts_fixture",
    "remove_leftover_sdf_containers",
    "render_agent_dockerfile",
    "render_real_pr_agent_dockerfile",
    "render_task_toml_env_snippet",
    "require_disk_for_envbuild",
    "require_full_sha",
]
