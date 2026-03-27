"""Harness report generation.

Writes harness evaluation results to JSON files and prints human-readable summaries.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swe_forge.swe.harness import HarnessResult


class HarnessReport:
    """Generates reports for harness evaluation results.

    Provides both machine-readable JSON output and human-readable console summaries.

    Example:
        report = HarnessReport()
        report.write_report(results, "output/report.json")
        summary = report.print_summary(results)
        print(summary)
    """

    def write_report(
        self,
        results: list[HarnessResult],
        output_path: str,
    ) -> None:
        """Write harness results to a JSON file.

        Args:
            results: List of HarnessResult objects.
            output_path: Path to write the JSON report.

        The JSON structure is:
        {
            "generated_at": "ISO_TIMESTAMP",
            "total_tasks": N,
            "resolved": N,
            "results": [...]
        }
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_tasks": len(results),
            "resolved": sum(1 for r in results if r.resolved),
            "results": [self._result_to_dict(r) for r in results],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    def print_summary(self, results: list[HarnessResult]) -> str:
        """Generate a human-readable summary of harness results.

        Args:
            results: List of HarnessResult objects.

        Returns:
            Human-readable summary string.

        Format:
            Harness Results: 10 tasks, 6 resolved (60.0%)
            RESOLVED: 6 | UNRESOLVED: 3 | ERROR: 1
            Avg duration: 45.2s
        """
        from swe_forge.swe.harness import HarnessStatus

        total = len(results)
        if total == 0:
            return "Harness Results: 0 tasks"

        resolved = sum(1 for r in results if r.status == HarnessStatus.RESOLVED)
        unresolved = sum(1 for r in results if r.status == HarnessStatus.UNRESOLVED)
        errors = sum(
            1
            for r in results
            if r.status
            in (
                HarnessStatus.AGENT_ERROR,
                HarnessStatus.TEST_ERROR,
                HarnessStatus.SETUP_ERROR,
                HarnessStatus.SANITY_FAIL,
            )
        )

        resolution_pct = (resolved / total) * 100 if total > 0 else 0.0
        durations = [r.duration_seconds for r in results if r.duration_seconds > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        lines = [
            f"Harness Results: {total} tasks, {resolved} resolved ({resolution_pct:.1f}%)",
            f"RESOLVED: {resolved} | UNRESOLVED: {unresolved} | ERROR: {errors}",
            f"Avg duration: {avg_duration:.1f}s",
        ]

        return "\n".join(lines)

    def _result_to_dict(self, result: HarnessResult) -> dict:
        """Convert HarnessResult to a JSON-serializable dict.

        Args:
            result: HarnessResult to convert.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        data = {
            "task_id": result.task_id,
            "status": result.status.value,
            "resolved": result.resolved,
            "patch_applied": result.patch_applied,
            "fail_to_pass_results": [
                {"command": cmd, "passed": passed}
                for cmd, passed in result.fail_to_pass_results
            ],
            "pass_to_pass_results": [
                {"command": cmd, "passed": passed}
                for cmd, passed in result.pass_to_pass_results
            ],
            "duration_seconds": result.duration_seconds,
        }

        if result.error_message:
            data["error_message"] = result.error_message

        if result.container_id:
            data["container_id"] = result.container_id

        return data
