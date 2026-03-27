"""Tests for result_aggregation module."""

from __future__ import annotations


from swe_forge.swe.harness import HarnessResult, HarnessStatus
from swe_forge.swe.result_aggregation import (
    AggregateStats,
    aggregate_results,
    to_json_dict,
)


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_empty_results(self) -> None:
        """Test aggregating empty list returns zero stats."""
        stats = aggregate_results([])

        assert stats.total_tasks == 0
        assert stats.resolved == 0
        assert stats.unresolved == 0
        assert stats.errors == 0
        assert stats.sanity_fails == 0
        assert stats.resolved_rate == 0.0
        assert stats.avg_duration_seconds == 0.0
        assert stats.total_fail_to_pass_tests == 0
        assert stats.total_pass_to_pass_tests == 0

    def test_all_resolved(self) -> None:
        """Test aggregating when all tasks are resolved."""
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=10.0,
                fail_to_pass_results=[("test1", True), ("test2", True)],
                pass_to_pass_results=[("test3", True)],
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=20.0,
                fail_to_pass_results=[("test1", True)],
                pass_to_pass_results=[("test2", True), ("test3", True)],
            ),
            HarnessResult(
                task_id="task-3",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=30.0,
                fail_to_pass_results=[],
                pass_to_pass_results=[],
            ),
        ]

        stats = aggregate_results(results)

        assert stats.total_tasks == 3
        assert stats.resolved == 3
        assert stats.unresolved == 0
        assert stats.errors == 0
        assert stats.sanity_fails == 0
        assert stats.resolved_rate == 1.0
        assert stats.avg_duration_seconds == 20.0  # (10 + 20 + 30) / 3
        assert stats.total_fail_to_pass_tests == 3  # 2 + 1 + 0
        assert stats.total_pass_to_pass_tests == 3  # 1 + 2 + 0

    def test_all_failed(self) -> None:
        """Test aggregating when all tasks are unresolved."""
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=5.0,
                fail_to_pass_results=[("test1", False)],
                pass_to_pass_results=[],
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=15.0,
                fail_to_pass_results=[("test1", False), ("test2", True)],
                pass_to_pass_results=[("test3", True)],
            ),
        ]

        stats = aggregate_results(results)

        assert stats.total_tasks == 2
        assert stats.resolved == 0
        assert stats.unresolved == 2
        assert stats.errors == 0
        assert stats.sanity_fails == 0
        assert stats.resolved_rate == 0.0
        assert stats.avg_duration_seconds == 10.0  # (5 + 15) / 2
        assert stats.total_fail_to_pass_tests == 3  # 1 + 2
        assert stats.total_pass_to_pass_tests == 1  # 0 + 1

    def test_mixed_statuses(self) -> None:
        """Test aggregating with mixed statuses."""
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=10.0,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=20.0,
            ),
            HarnessResult(
                task_id="task-3",
                status=HarnessStatus.AGENT_ERROR,
                resolved=False,
                duration_seconds=30.0,
                error_message="Agent timeout",
            ),
            HarnessResult(
                task_id="task-4",
                status=HarnessStatus.TEST_ERROR,
                resolved=False,
                duration_seconds=40.0,
                error_message="Test crashed",
            ),
            HarnessResult(
                task_id="task-5",
                status=HarnessStatus.SETUP_ERROR,
                resolved=False,
                duration_seconds=50.0,
                error_message="Container failed",
            ),
            HarnessResult(
                task_id="task-6",
                status=HarnessStatus.SANITY_FAIL,
                resolved=False,
                duration_seconds=60.0,
                error_message="Sanity check failed",
            ),
        ]

        stats = aggregate_results(results)

        assert stats.total_tasks == 6
        assert stats.resolved == 1
        assert stats.unresolved == 1
        assert stats.errors == 3  # agent_error + test_error + setup_error
        assert stats.sanity_fails == 1
        # Resolved rate: 1/6 ≈ 0.1667
        assert abs(stats.resolved_rate - 1 / 6) < 0.001
        # Average duration: (10 + 20 + 30 + 40 + 50 + 60) / 6 = 35.0
        assert stats.avg_duration_seconds == 35.0

    def test_resolved_rate_calculation(self) -> None:
        """Test that resolved rate is calculated correctly."""
        # 2 resolved out of 4 = 0.5
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=0.0,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=0.0,
            ),
            HarnessResult(
                task_id="task-3",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=0.0,
            ),
            HarnessResult(
                task_id="task-4",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=0.0,
            ),
        ]

        stats = aggregate_results(results)

        assert stats.resolved_rate == 0.5

    def test_test_counts_accumulation(self) -> None:
        """Test that test counts are properly accumulated."""
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=0.0,
                fail_to_pass_results=[
                    ("f2p-1", True),
                    ("f2p-2", True),
                    ("f2p-3", True),
                ],
                pass_to_pass_results=[
                    ("p2p-1", True),
                ],
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=0.0,
                fail_to_pass_results=[
                    ("f2p-4", False),
                ],
                pass_to_pass_results=[
                    ("p2p-2", True),
                    ("p2p-3", True),
                    ("p2p-4", True),
                ],
            ),
        ]

        stats = aggregate_results(results)

        # Count of results, not passed tests
        assert stats.total_fail_to_pass_tests == 4  # 3 + 1
        assert stats.total_pass_to_pass_tests == 4  # 1 + 3


class TestAggregateStats:
    """Tests for AggregateStats dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test that AggregateStats has all expected fields."""
        stats = AggregateStats(
            total_tasks=100,
            resolved=75,
            unresolved=20,
            errors=4,
            sanity_fails=1,
            resolved_rate=0.75,
            avg_duration_seconds=45.5,
            total_fail_to_pass_tests=200,
            total_pass_to_pass_tests=150,
        )

        assert stats.total_tasks == 100
        assert stats.resolved == 75
        assert stats.unresolved == 20
        assert stats.errors == 4
        assert stats.sanity_fails == 1
        assert stats.resolved_rate == 0.75
        assert stats.avg_duration_seconds == 45.5
        assert stats.total_fail_to_pass_tests == 200
        assert stats.total_pass_to_pass_tests == 150


class TestToJsonDict:
    """Tests for to_json_dict function."""

    def test_json_dict_contains_all_fields(self) -> None:
        """Test that JSON dict contains all stats fields."""
        stats = AggregateStats(
            total_tasks=10,
            resolved=7,
            unresolved=2,
            errors=1,
            sanity_fails=0,
            resolved_rate=0.7,
            avg_duration_seconds=45.5,
            total_fail_to_pass_tests=20,
            total_pass_to_pass_tests=15,
        )

        result = to_json_dict(stats)

        assert result["total_tasks"] == 10
        assert result["resolved"] == 7
        assert result["unresolved"] == 2
        assert result["errors"] == 1
        assert result["sanity_fails"] == 0
        assert result["resolved_rate"] == 0.7
        assert result["avg_duration_seconds"] == 45.5
        assert result["total_fail_to_pass_tests"] == 20
        assert result["total_pass_to_pass_tests"] == 15

    def test_json_dict_serializable(self) -> None:
        """Test that result can be JSON serialized."""
        import json

        stats = AggregateStats(
            total_tasks=5,
            resolved=3,
            unresolved=1,
            errors=1,
            sanity_fails=0,
            resolved_rate=0.6,
            avg_duration_seconds=12.5,
            total_fail_to_pass_tests=10,
            total_pass_to_pass_tests=5,
        )

        result = to_json_dict(stats)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Round-trip should work
        parsed = json.loads(json_str)
        assert parsed["total_tasks"] == 5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_result(self) -> None:
        """Test aggregating a single result."""
        result = HarnessResult(
            task_id="single-task",
            status=HarnessStatus.RESOLVED,
            resolved=True,
            duration_seconds=42.5,
            fail_to_pass_results=[("test", True)],
            pass_to_pass_results=[("test2", True)],
        )

        stats = aggregate_results([result])

        assert stats.total_tasks == 1
        assert stats.resolved == 1
        assert stats.unresolved == 0
        assert stats.resolved_rate == 1.0
        assert stats.avg_duration_seconds == 42.5
        assert stats.total_fail_to_pass_tests == 1
        assert stats.total_pass_to_pass_tests == 1

    def test_zero_duration_results(self) -> None:
        """Test aggregating results with zero duration."""
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=0.0,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=0.0,
            ),
        ]

        stats = aggregate_results(results)

        assert stats.avg_duration_seconds == 0.0

    def test_decimal_resolved_rate(self) -> None:
        """Test resolved rate with decimal precision."""
        # 1 resolved out of 3 = 0.333...
        results = [
            HarnessResult(
                task_id="task-1",
                status=HarnessStatus.RESOLVED,
                resolved=True,
                duration_seconds=0.0,
            ),
            HarnessResult(
                task_id="task-2",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=0.0,
            ),
            HarnessResult(
                task_id="task-3",
                status=HarnessStatus.UNRESOLVED,
                resolved=False,
                duration_seconds=0.0,
            ),
        ]

        stats = aggregate_results(results)

        # Check approximate value
        assert abs(stats.resolved_rate - (1 / 3)) < 0.0001
