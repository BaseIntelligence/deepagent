"""Result aggregation for SWE-bench harness runs.

Aggregates multiple HarnessResult instances into summary statistics
for reporting and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swe_forge.swe.harness import HarnessResult


@dataclass
class AggregateStats:
    """Aggregated statistics from multiple harness runs.

    Attributes:
        total_tasks: Total number of tasks evaluated.
        resolved: Number of resolved tasks.
        unresolved: Number of unresolved tasks.
        errors: Number of error statuses (agent_error + test_error + setup_error).
        sanity_fails: Number of sanity check failures.
        resolved_rate: Percentage of tasks resolved (0.0 to 1.0).
        avg_duration_seconds: Average duration across all tasks.
        total_fail_to_pass_tests: Total fail_to_pass tests across all results.
        total_pass_to_pass_tests: Total pass_to_pass tests across all results.
    """

    total_tasks: int
    resolved: int
    unresolved: int
    errors: int
    sanity_fails: int
    resolved_rate: float
    avg_duration_seconds: float
    total_fail_to_pass_tests: int
    total_pass_to_pass_tests: int


def aggregate_results(results: list["HarnessResult"]) -> AggregateStats:
    """Aggregate multiple harness results into summary statistics.

    Args:
        results: List of HarnessResult instances to aggregate.

    Returns:
        AggregateStats with calculated statistics.

    Example:
        >>> from swe_forge.swe.harness import HarnessResult, HarnessStatus
        >>> results = [
        ...     HarnessResult(task_id="1", status=HarnessStatus.RESOLVED, resolved=True),
        ...     HarnessResult(task_id="2", status=HarnessStatus.UNRESOLVED, resolved=False),
        ... ]
        >>> stats = aggregate_results(results)
        >>> stats.total_tasks
        2
        >>> stats.resolved
        1
    """
    from swe_forge.swe.harness import HarnessStatus

    total_tasks = len(results)

    if total_tasks == 0:
        return AggregateStats(
            total_tasks=0,
            resolved=0,
            unresolved=0,
            errors=0,
            sanity_fails=0,
            resolved_rate=0.0,
            avg_duration_seconds=0.0,
            total_fail_to_pass_tests=0,
            total_pass_to_pass_tests=0,
        )

    resolved_count = 0
    unresolved_count = 0
    errors_count = 0
    sanity_fails_count = 0
    total_duration = 0.0
    total_fail_to_pass_tests = 0
    total_pass_to_pass_tests = 0

    error_statuses = {
        HarnessStatus.AGENT_ERROR,
        HarnessStatus.TEST_ERROR,
        HarnessStatus.SETUP_ERROR,
    }

    for result in results:
        # Count by status
        if result.status == HarnessStatus.RESOLVED:
            resolved_count += 1
        elif result.status == HarnessStatus.UNRESOLVED:
            unresolved_count += 1
        elif result.status in error_statuses:
            errors_count += 1
        elif result.status == HarnessStatus.SANITY_FAIL:
            sanity_fails_count += 1

        # Accumulate duration
        total_duration += result.duration_seconds

        # Accumulate test counts
        total_fail_to_pass_tests += len(result.fail_to_pass_results)
        total_pass_to_pass_tests += len(result.pass_to_pass_results)

    resolved_rate = resolved_count / total_tasks if total_tasks > 0 else 0.0
    avg_duration = total_duration / total_tasks if total_tasks > 0 else 0.0

    return AggregateStats(
        total_tasks=total_tasks,
        resolved=resolved_count,
        unresolved=unresolved_count,
        errors=errors_count,
        sanity_fails=sanity_fails_count,
        resolved_rate=resolved_rate,
        avg_duration_seconds=avg_duration,
        total_fail_to_pass_tests=total_fail_to_pass_tests,
        total_pass_to_pass_tests=total_pass_to_pass_tests,
    )


def to_json_dict(stats: AggregateStats) -> dict:
    """Convert AggregateStats to a JSON-serializable dictionary.

    Args:
        stats: AggregateStats instance to convert.

    Returns:
        Dictionary with all stats fields, suitable for JSON serialization.

    Example:
        >>> from swe_forge.swe.result_aggregation import AggregateStats
        >>> stats = AggregateStats(
        ...     total_tasks=10,
        ...     resolved=7,
        ...     unresolved=2,
        ...     errors=1,
        ...     sanity_fails=0,
        ...     resolved_rate=0.7,
        ...     avg_duration_seconds=45.5,
        ...     total_fail_to_pass_tests=20,
        ...     total_pass_to_pass_tests=15,
        ... )
        >>> d = to_json_dict(stats)
        >>> d["total_tasks"]
        10
    """
    return asdict(stats)
