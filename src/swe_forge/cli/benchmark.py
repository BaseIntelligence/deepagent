"""Benchmark command for SWE task evaluation.

Usage:
    swe-forge benchmark --model gpt-4 --tasks 10 --difficulty easy
    swe-forge benchmark --model gpt-4 --tasks 5 --output results/ --report
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from swe_forge.export.jsonl import export_jsonl
from swe_forge.swe.github_api import GitHubClient
from swe_forge.swe.harness import HarnessConfig, HarnessStatus
from swe_forge.swe.models import SweTask
from swe_forge.swe.pipeline import SwePipeline, SwePipelineConfig

logger = logging.getLogger(__name__)

app = typer.Typer(name="benchmark", help="Benchmark LLM models on SWE tasks")

console = Console()


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    tasks_completed: int = 0
    resolved_count: int = 0
    total_count: int = 0
    results: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def validate_difficulty(difficulty: str) -> bool:
    """Validate that difficulty is one of: easy, medium, hard."""
    valid_difficulties = {"easy", "medium", "hard"}
    return difficulty.lower() in valid_difficulties


@app.command()
def benchmark(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model to benchmark",
        ),
    ] = "gpt-4",
    tasks: Annotated[
        int,
        typer.Option(
            "--tasks",
            "-t",
            help="Number of tasks to run",
            min=1,
        ),
    ] = 10,
    difficulty: Annotated[
        Optional[str],
        typer.Option(
            "--difficulty",
            "-d",
            help="Task difficulty filter (easy/medium/hard)",
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Results output directory",
        ),
    ] = "results/",
    report: Annotated[
        bool,
        typer.Option(
            "--report",
            "-r",
            help="Generate HTML report",
        ),
    ] = False,
    parallel: Annotated[
        int,
        typer.Option(
            "--parallel",
            "-p",
            help="Number of parallel tasks",
            min=1,
            max=32,
        ),
    ] = 1,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            help="Per-task timeout in seconds",
            min=60,
        ),
    ] = 600,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
) -> None:
    """Benchmark an LLM model on SWE tasks.

    Mines tasks from GitHub, runs the harness evaluation, and generates results.

    Examples:
        swe-forge benchmark --model gpt-4 --tasks 10 --difficulty easy
        swe-forge benchmark --model gpt-4 --tasks 5 --output results/ --report
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if difficulty and not validate_difficulty(difficulty):
        console.print("[red]Error: Difficulty must be one of: easy, medium, hard[/red]")
        raise typer.Exit(code=1)

    if not model.strip():
        console.print("[red]Error: Model name cannot be empty[/red]")
        raise typer.Exit(code=1)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]SWE-Forge Benchmark Configuration[/bold blue]")
    console.print(f"  Model: {model}")
    console.print(f"  Tasks: {tasks}")
    console.print(f"  Difficulty: {difficulty or 'All'}")
    console.print(f"  Output: {output}")
    console.print(f"  Report: {'Yes' if report else 'No'}")
    console.print(f"  Parallel: {parallel}")
    console.print(f"  Timeout: {timeout}s")
    console.print()

    try:
        result = asyncio.run(
            _run_benchmark(
                model=model,
                num_tasks=tasks,
                difficulty_filter=difficulty.lower() if difficulty else None,
                output_path=output_path,
                parallel=parallel,
                timeout=timeout,
                verbose=verbose,
            )
        )

        results_file = output_path / "benchmark_results.json"
        _save_results(result, results_file)

        if report:
            _generate_report(result, output_path)

        _print_summary(result.results)

        console.print(
            f"\n[green]Benchmark complete: {result.resolved_count}/{result.total_count} resolved[/green]"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


async def _run_benchmark(
    model: str,
    num_tasks: int,
    difficulty_filter: Optional[str],
    output_path: Path,
    parallel: int,
    timeout: int,
    verbose: bool,
) -> BenchmarkResult:
    """Run the benchmark: mine tasks, run harness, aggregate results."""
    github_token = os.environ.get("GITHUB_TOKEN", "")

    console.print("[cyan]Mining tasks...[/cyan]")

    mined_tasks = await _mine_tasks(
        github_token=github_token,
        num_tasks=num_tasks,
        difficulty_filter=difficulty_filter,
        verbose=verbose,
    )

    if not mined_tasks:
        console.print("[yellow]No tasks mined[/yellow]")
        return BenchmarkResult()

    console.print(f"[green]Mined {len(mined_tasks)} tasks[/green]")

    tasks_path = output_path / "tasks.jsonl"
    export_jsonl(mined_tasks, tasks_path)

    console.print("[cyan]Running harness evaluation...[/cyan]")

    results = await _run_harness(
        tasks=mined_tasks,
        model=model,
        parallel=parallel,
        timeout=timeout,
    )

    resolved_count = sum(1 for r in results if r.get("resolved", False))

    return BenchmarkResult(
        tasks_completed=len(results),
        resolved_count=resolved_count,
        total_count=len(results),
        results=results,
        metrics={
            "model": model,
            "difficulty_filter": difficulty_filter,
            "resolution_rate": resolved_count / len(results) if results else 0,
        },
    )


async def _mine_tasks(
    github_token: str,
    num_tasks: int,
    difficulty_filter: Optional[str],
    verbose: bool,
) -> list[SweTask]:
    """Mine tasks using the SWE pipeline."""
    async with GitHubClient(token=github_token) as gh_client:
        config = SwePipelineConfig(
            max_tasks=num_tasks,
            once=True,
            difficulty_filter=difficulty_filter,
        )

        tasks: list[SweTask] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Mining tasks...", total=num_tasks)

            async with SwePipeline(gh_client, config=config) as pipeline:
                from swe_forge.swe.pipeline import SwePipelineEventType

                async for event in pipeline.run_with_progress():
                    if event.event_type == SwePipelineEventType.TASK_EXTRACTED:
                        task = event.data.get("task")
                        if task and isinstance(task, SweTask):
                            tasks.append(task)
                            progress.update(
                                task_id, advance=1, description=f"Mined {len(tasks)} tasks"
                            )

                        if len(tasks) >= num_tasks:
                            break

        return tasks[:num_tasks]


async def _run_harness(
    tasks: list[SweTask],
    model: str,
    parallel: int,
    timeout: int,
) -> list[dict]:
    """Run harness evaluation on all tasks."""
    config = HarnessConfig(
        agent_timeout_seconds=float(timeout),
        agent_script=None,
    )

    results: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Running harness...", total=len(tasks))

        for i, task in enumerate(tasks):
            progress.update(
                task_id, description=f"Task {i + 1}/{len(tasks)}: {task.id[:30]}"
            )

            result_dict = await _evaluate_task(task, config, model)
            results.append(result_dict)
            progress.advance(task_id)

    return results


async def _evaluate_task(
    task: SweTask,
    config: HarnessConfig,
    model: str,
) -> dict:
    """Evaluate a single task.

    Note: This is a stub implementation. Full implementation would:
    1. Create a Docker container/sandbox
    2. Clone the repo at base commit
    3. Run the model as an agent
    4. Verify with tests
    """
    try:
        result_dict = {
            "task_id": task.id,
            "status": HarnessStatus.SETUP_ERROR.value,
            "resolved": False,
            "duration_seconds": 0.0,
            "error_message": "Sandbox not implemented in CLI stub",
            "model": model,
            "repo": task.repo,
            "fail_to_pass_results": [],
            "pass_to_pass_results": [],
        }
    except Exception as e:
        result_dict = {
            "task_id": task.id,
            "status": HarnessStatus.AGENT_ERROR.value,
            "resolved": False,
            "duration_seconds": 0.0,
            "error_message": str(e),
            "model": model,
            "repo": task.repo,
            "fail_to_pass_results": [],
            "pass_to_pass_results": [],
        }

    return result_dict


def _save_results(result: BenchmarkResult, output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    data = {
        "tasks_completed": result.tasks_completed,
        "resolved_count": result.resolved_count,
        "total_count": result.total_count,
        "resolution_rate": (
            result.resolved_count / result.total_count if result.total_count > 0 else 0
        ),
        "metrics": result.metrics,
        "results": result.results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Results saved to {output_path}[/green]")


def _generate_report(result: BenchmarkResult, output_path: Path) -> None:
    """Generate HTML report from benchmark results."""
    report_path = output_path / "benchmark_report.html"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SWE-Forge Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ margin: 10px 0; }}
        .resolved {{ color: green; }}
        .unresolved {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>SWE-Forge Benchmark Report</h1>
    <div class="metric">Model: {result.metrics.get("model", "unknown")}</div>
    <div class="metric">Tasks Completed: {result.tasks_completed}</div>
    <div class="metric">Resolved: {result.resolved_count}/{result.total_count}</div>
    <div class="metric">Resolution Rate: {result.metrics.get("resolution_rate", 0):.1%}</div>

    <h2>Results</h2>
    <table>
        <tr>
            <th>Task ID</th>
            <th>Status</th>
            <th>Resolved</th>
        </tr>
"""

    for r in result.results:
        status = r.get("status", "unknown")
        resolved = r.get("resolved", False)
        status_class = "resolved" if resolved else "unresolved"
        html_content += f"""
        <tr>
            <td>{r.get("task_id", "unknown")}</td>
            <td class="{status_class}">{status}</td>
            <td class="{status_class}">{resolved}</td>
        </tr>
"""

    html_content += """
    </table>
</body>
</html>
"""

    with open(report_path, "w") as f:
        f.write(html_content)

    console.print(f"[green]HTML report generated at {report_path}[/green]")


def _print_summary(results: list[dict]) -> None:
    """Print summary table of benchmark results."""
    status_counts: dict[str, int] = {}
    for r in results:
        status = r.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    resolved_count = status_counts.get(HarnessStatus.RESOLVED.value, 0)
    total_count = len(results)

    table = Table(title="Benchmark Results Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for status, count in sorted(status_counts.items()):
        pct = (count / total_count * 100) if total_count > 0 else 0
        table.add_row(status, str(count), f"{pct:.1f}%")

    console.print(table)
    console.print()

    if total_count > 0:
        console.print(
            f"[bold]Resolution rate: {resolved_count}/{total_count} "
            f"({resolved_count / total_count * 100:.1f}%)[/bold]"
        )


if __name__ == "__main__":
    app()
