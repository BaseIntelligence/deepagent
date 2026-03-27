"""CLI harness command for running SWE evaluation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

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

from swe_forge.swe.harness import HarnessConfig, HarnessRunner, HarnessStatus
from swe_forge.swe.models import SweTask

console = Console()

app = typer.Typer(name="harness", help="Run SWE evaluation harness")


@app.command()
def harness(
    input: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input JSONL file with tasks",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    agent_script: Annotated[
        str | None,
        typer.Option(
            "--agent-script",
            "-a",
            help="Script to run as agent",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            "-t",
            help="Per-task timeout in seconds",
            min=1,
        ),
    ] = 600,
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
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Results output file (JSON format)",
        ),
    ] = Path("results.json"),
) -> None:
    """Run SWE evaluation harness on tasks.

    Reads tasks from a JSONL file, runs the agent script on each task,
    and outputs evaluation results.

    Example:
        swe-forge harness --input tasks.jsonl --agent-script ./agent.sh --timeout 300
    """
    tasks = _load_tasks(input)

    if not tasks:
        console.print("[yellow]No tasks to process[/yellow]")
        return

    console.print(f"[blue]Loaded {len(tasks)} tasks from {input}[/blue]")

    config = HarnessConfig(
        agent_timeout_seconds=float(timeout),
        agent_script=agent_script,
    )

    results = asyncio.run(_run_harness(tasks, config, parallel))
    _output_results(results, output)
    _print_summary(results)


async def _run_harness(
    tasks: list[SweTask],
    config: HarnessConfig,
    parallel: int,
) -> list[dict]:
    """Run harness on all tasks with progress display."""
    results: list[dict] = []
    runner = HarnessRunner(config=config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_progress = progress.add_task(
            "[cyan]Running harness...",
            total=len(tasks),
        )

        semaphore = asyncio.Semaphore(parallel)

        async def process_task(task: SweTask) -> dict:
            async with semaphore:
                result = await _process_single_task(runner, task, config)
                progress.advance(task_progress)
                return result

        coroutines = [process_task(task) for task in tasks]
        results = await asyncio.gather(*coroutines)

    return results


async def _process_single_task(
    runner: HarnessRunner,
    task: SweTask,
    config: HarnessConfig,
) -> dict:
    """Process a single task and return result dict.

    Note: This is a simplified version. A full implementation would:
    1. Create a Docker container/sandbox
    2. Clone the repo at base commit
    3. Run the harness
    4. Clean up the container
    """
    try:
        result_dict = {
            "task_id": task.id,
            "status": HarnessStatus.SETUP_ERROR.value,
            "resolved": False,
            "duration_seconds": 0.0,
            "error_message": "Sandbox not implemented in CLI stub",
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
            "fail_to_pass_results": [],
            "pass_to_pass_results": [],
        }

    return result_dict


def _load_tasks(input_path: Path) -> list[SweTask]:
    """Load tasks from JSONL file."""
    tasks = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks.append(SweTask(**data))
    return tasks


def _output_results(results: list[dict], output_path: Path) -> None:
    """Output results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Results written to {output_path}[/green]")


def _print_summary(results: list[dict]) -> None:
    """Print summary table of results."""
    status_counts: dict[str, int] = {}
    for r in results:
        status = r.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    resolved_count = status_counts.get(HarnessStatus.RESOLVED.value, 0)
    total_count = len(results)

    table = Table(title="Harness Results Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for status, count in sorted(status_counts.items()):
        pct = (count / total_count * 100) if total_count > 0 else 0
        table.add_row(status, str(count), f"{pct:.1f}%")

    console.print(table)
    console.print()
    if total_count > 0:
        rate_pct = resolved_count / total_count * 100
        console.print(
            f"[bold]Resolution rate: {resolved_count}/{total_count} "
            f"({rate_pct:.1f}%)[/bold]"
        )
    else:
        console.print("[bold]Resolution rate: 0/0 (0.0%)[/bold]")
