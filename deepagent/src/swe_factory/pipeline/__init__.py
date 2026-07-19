"""End-to-end pipeline orchestration (micro-keep / ship paths)."""

from swe_factory.pipeline.archive_hybrid import (
    ArchiveHybridError,
    ArchiveHybridResult,
    archive_hybrid_deepagent,
)
from swe_factory.pipeline.micro_keep import (
    MicroKeepError,
    MicroKeepResult,
    run_micro_keep,
)
from swe_factory.pipeline.repo_diversity import (
    DEFAULT_MAX_PACKS_PER_REPO,
    apply_max_packs_per_repo,
    normalize_upstream_repo,
    select_diverse_pack_ids,
)
from swe_factory.pipeline.ship_deepagent import (
    ShipDeepAgentResult,
    run_ship_deepagent,
)
from swe_factory.pipeline.ship_harbor import (
    ShipHarborResult,
    run_ship_harbor,
)
from swe_factory.pipeline.ship_real_pr import (
    HybridProductPromoteRejected,
    ShipRealPrError,
    run_ship_deepagent_real_pr,
)
from swe_factory.pipeline.ship_v1 import (
    ShipV1Result,
    default_harvest_plan,
    run_ship_v1,
)

__all__ = [
    "ArchiveHybridError",
    "ArchiveHybridResult",
    "DEFAULT_MAX_PACKS_PER_REPO",
    "HybridProductPromoteRejected",
    "MicroKeepError",
    "MicroKeepResult",
    "ShipDeepAgentResult",
    "ShipHarborResult",
    "ShipRealPrError",
    "ShipV1Result",
    "apply_max_packs_per_repo",
    "archive_hybrid_deepagent",
    "default_harvest_plan",
    "normalize_upstream_repo",
    "run_micro_keep",
    "run_ship_deepagent",
    "run_ship_deepagent_real_pr",
    "run_ship_harbor",
    "run_ship_v1",
    "select_diverse_pack_ids",
]
