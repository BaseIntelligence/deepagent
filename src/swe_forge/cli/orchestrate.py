"""CLI command for orchestrating the full pipeline.

Usage:
    swe-forge orchestrate --tasks-dir ./tasks --parallel 5
    swe-forge orchestrate --tasks-dir ./tasks --min-score 0.6 --output report.json
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from swe_forge.orchestrator import MasterOrchestrator, OrchestratorTask, TaskState

logger = logging.getLogger(__name__)

app = typer.Typer(name="orchestrate", help="Orchestrate the full SWE-Forge pipeline")

console = Console()


def load_tasks_from_directory(tasks_dir: Path) -> list[OrchestratorTask]:
    """Load tasks from workspace.yaml files in the tasks directory.

    Args:
        tasks_dir: Directory containing task folders with workspace.yaml files.

    Returns:
        List of OrchestratorTask objects ready for processing.
    """
    tasks: list[OrchestratorTask] = []

    if not tasks_dir.exists():
        return tasks

    for task_folder in tasks_dir.iterdir():
        if not task_folder.is_dir():
            continue

        workspace_path = task_folder / "workspace.yaml"
        if not workspace_path.exists():
            continue

        try:
            with open(workspace_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                continue

            # Read patch file if exists
            patch_path = task_folder / "patch.diff"
            patch_content = ""
            if patch_path.exists():
                with open(patch_path, "r") as f:
                    patch_content = f.read()

            task = OrchestratorTask(
                task_id=data.get("task_id", task_folder.name),
                repo_url=data.get("repo", {}).get("url", ""),
                base_commit=data.get("repo", {}).get("base_commit", ""),
                patch=patch_content,
                language=data.get("language", "unknown"),
                tests=data.get("tests", {}),
                metadata=data,
            )
            tasks.append(task)

        except Exception as e:
            logger.warning(f"Failed to load task from {task_folder}: {e}")
            continue

    return tasks


@app.command("run")
def orchestrate(
    tasks_dir: Annotated[
        Path,
        typer.Option(
            "--tasks-dir",
            "-t",
            help="Directory containing task folders with workspace.yaml files",
        ),
    ] = Path("./tasks"),
    parallel: Annotated[
        int,
        typer.Option(
            "--parallel",
            "-p",
            help="Number of parallel workers",
            min=1,
        ),
    ] = 5,
    min_score: Annotated[
        float,
        typer.Option(
            "--min-score",
            "-s",
            help="Minimum score threshold for publishing",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5,
    max_repair: Annotated[
        int,
        typer.Option(
            "--max-repair",
            "-r",
            help="Maximum repair attempts per task",
            min=0,
        ),
    ] = 5,
    output_report: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Path to save the orchestration report JSON",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model for test generation and repair",
        ),
    ] = "openai/gpt-4o",
    docker_username: Annotated[
        Optional[str],
        typer.Option(
            "--docker-username",
            "-D",
            help="Docker Hub username for image naming",
        ),
    ] = None,
    push_images: Annotated[
        bool,
        typer.Option(
            "--push-images",
            help="Push Docker images to registry",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
) -> None:
    """Run the orchestration pipeline on all tasks.

    Pipeline: Generate Tests → Validate → Build Docker → Verify → Repair → Score → Publish

    Examples:
        swe-forge orchestrate --tasks-dir ./tasks --parallel 5
        swe-forge orchestrate --tasks-dir ./tasks --min-score 0.6 --output report.json
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Check API key
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        console.print(
            "[red]Error: OPENROUTER_API_KEY environment variable not set[/red]"
        )
        raise typer.Exit(code=1)

    # Load tasks from directory
    if not tasks_dir.exists():
        console.print(f"[red]Error: Tasks directory not found: {tasks_dir}[/red]")
        raise typer.Exit(code=1)

    tasks = load_tasks_from_directory(tasks_dir)

    if not tasks:
        console.print(f"[yellow]No tasks found in {tasks_dir}[/yellow]")
        raise typer.Exit(code=0)

    # Display configuration
    console.print("[bold blue]SWE-Forge Orchestration Configuration[/bold blue]")
    console.print(f"  Tasks Directory: {tasks_dir}")
    console.print(f"  Tasks Found: {len(tasks)}")
    console.print(f"  Parallel Workers: {parallel}")
    console.print(f"  Min Score Threshold: {min_score}")
    console.print(f"  Max Repair Attempts: {max_repair}")
    console.print(f"  Model: {model}")
    if docker_username:
        console.print(f"  Docker Username: {docker_username}")
        console.print(f"  Push Images: {'Yes' if push_images else 'No'}")
    if output_report:
        console.print(f"  Output Report: {output_report}")
    console.print()

    # Run orchestration
    try:
        stats = asyncio.run(
            _run_orchestration(
                tasks=tasks,
                openrouter_key=openrouter_key,
                parallel=parallel,
                min_score=min_score,
                max_repair=max_repair,
                model=model,
                docker_username=docker_username,
                push_images=push_images,
                verbose=verbose,
            )
        )

        # Print summary
        console.print("\n[bold]Orchestration Summary:[/bold]")
        console.print(f"  Total Tasks: {stats.total_tasks}")
        console.print(f"  Completed: {stats.state_counts.get(TaskState.COMPLETED, 0)}")
        console.print(f"  Rejected: {stats.state_counts.get(TaskState.REJECTED, 0)}")
        console.print(f"  Failed: {stats.state_counts.get(TaskState.FAILED, 0)}")
        console.print(f"  Pass Rate: {stats.pass_rate():.1%}")
        console.print(f"  Avg Time/Task: {stats.average_time_per_task():.1f}s")

        # Save report if requested
        if output_report:
            report = {
                "total_tasks": stats.total_tasks,
                "state_counts": {k.value: v for k, v in stats.state_counts.items()},
                "pass_rate": stats.pass_rate(),
                "average_time_per_task": stats.average_time_per_task(),
                "timing": stats.timing,
            }
            with open(output_report, "w") as f:
                json.dump(report, f, indent=2)
            console.print(f"\n[green]Report saved to {output_report}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Orchestration interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Orchestration error: {e}")
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


async def _run_orchestration(
    tasks: list[OrchestratorTask],
    openrouter_key: str,
    parallel: int,
    min_score: float,
    max_repair: int,
    model: str,
    docker_username: Optional[str],
    push_images: bool,
    verbose: bool,
) -> "OrchestratorStats":
    """Run the orchestration pipeline asynchronously."""
    from swe_forge.llm.openrouter import OpenRouterClient

    # Create LLM client
    llm_client = OpenRouterClient(api_key=openrouter_key, default_model=model)

    # Create orchestrator
    orchestrator = MasterOrchestrator(
        parallel=parallel,
        llm_client=llm_client,
        min_score_threshold=min_score,
        max_repair_attempts=max_repair,
        model=model,
        docker_username=docker_username or "swe-forge",
        push_images=push_images,
    )

    # Run with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress_task = progress.add_task("Processing tasks...", total=len(tasks))

        # Run orchestration
        stats = await orchestrator.run_all(tasks)

        progress.update(
            progress_task,
            completed=len(tasks),
            description=f"Completed: {stats.state_counts.get(TaskState.COMPLETED, 0)}/{len(tasks)}",
        )

    return stats


if __name__ == "__main__":
    app()
