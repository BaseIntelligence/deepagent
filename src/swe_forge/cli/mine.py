"""Mine command for SWE task extraction.

Usage:
    swe-forge mine --repo owner/repo --limit 5 --output ./tasks.jsonl
    swe-forge mine --difficulty easy --model gpt-4 --once
    swe-forge mine --continuous
"""

import asyncio
import json
import logging
import os
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

from swe_forge.export.jsonl import export_jsonl
from swe_forge.llm.openrouter import OpenRouterClient
from swe_forge.swe.github_api import GitHubClient
from swe_forge.swe.gharchive import GhArchiveClient
from swe_forge.swe.models import SweTask
from swe_forge.swe.pipeline import (
    DifficultyTargets,
    SwePipeline,
    SwePipelineConfig,
    SwePipelineEventType,
)
from swe_forge.swe.test_generator import TestGenerator
from swe_forge.swe.concurrency import set_docker_containers_limit

logger = logging.getLogger(__name__)

app = typer.Typer(name="mine", help="Mine SWE tasks from GitHub repositories")

console = Console()


def validate_repo_format(repo: str) -> bool:
    """Validate repository format is owner/repo."""
    if not repo:
        return False
    parts = repo.split("/")
    return len(parts) == 2 and all(p.strip() for p in parts)


@app.command()
def mine(
    repo: Annotated[
        Optional[str],
        typer.Option(
            "--repo",
            "-r",
            help="Target repository in owner/repo format",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-l",
            help="Maximum number of tasks to mine",
            min=1,
        ),
    ] = 10,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for JSONL results",
        ),
    ] = "./tasks.jsonl",
    difficulty: Annotated[
        Optional[str],
        typer.Option(
            "--difficulty",
            "-d",
            help="Filter by difficulty level (easy/medium/hard)",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model for classification",
        ),
    ] = "moonshotai/kimi-k2.5",
    once: Annotated[
        bool,
        typer.Option(
            "--once",
            help="Run once then exit",
        ),
    ] = True,
    continuous: Annotated[
        bool,
        typer.Option(
            "--continuous",
            help="Keep running continuously",
        ),
    ] = False,
    max_candidates: Annotated[
        int,
        typer.Option(
            "--max-candidates",
            help="Maximum PR candidates to process",
        ),
    ] = 50,
    min_stars: Annotated[
        int,
        typer.Option(
            "--min-stars",
            help="Minimum repository stars required",
        ),
    ] = 100,
    language: Annotated[
        Optional[str],
        typer.Option(
            "--language",
            help="Filter by programming language",
        ),
    ] = None,
    filter_json: Annotated[
        str,
        typer.Option(
            "--filter",
            "-f",
            help='JSON filter for max tasks per difficulty. Default: {"easy": 10, "medium": 10, "hard": 10}',
        ),
    ] = '{"easy": 10, "medium": 10, "hard": 10}',
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
    parallel: Annotated[
        int,
        typer.Option(
            "--parallel",
            "-p",
            help="Maximum concurrent Docker containers",
            min=1,
        ),
    ] = 8,
    output_folder: Annotated[
        Path | None,
        typer.Option(
            "--output-folder",
            "-O",
            help="Output folder for workspace format export",
        ),
    ] = None,
    docker_username: Annotated[
        str | None,
        typer.Option(
            "--docker-username",
            "-D",
            help="Docker Hub username for image names (user -> user/swe-forge-tasks:task-id)",
        ),
    ] = None,
    build_docker: Annotated[
        bool,
        typer.Option(
            "--build-docker",
            "-B",
            help="Build Docker images with repo + deps pre-installed for faster evaluation",
        ),
    ] = False,
    docker_push: Annotated[
        bool,
        typer.Option(
            "--docker-push",
            help="Push built Docker images to registry",
        ),
    ] = False,
) -> None:
    """Mine SWE tasks from GitHub repositories.

    Extracts potential SWE-bench tasks from merged PRs using the pipeline.

    Examples:
        swe-forge mine --repo owner/repo --limit 5 --output ./tasks.jsonl
        swe-forge mine --difficulty easy --model gpt-4 --once
        swe-forge mine --continuous --limit 100
        swe-forge mine --limit 5 --build-docker --docker-username myuser
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure Docker container parallelism
    set_docker_containers_limit(parallel)

    # Validate repo format if provided
    if repo and not validate_repo_format(repo):
        console.print("[red]Error: Repository must be in 'owner/repo' format[/red]")
        raise typer.Exit(code=1)

    # Validate difficulty
    valid_difficulties = {"easy", "medium", "hard"}
    if difficulty and difficulty.lower() not in valid_difficulties:
        console.print(
            f"[red]Error: Difficulty must be one of: {', '.join(valid_difficulties)}[/red]"
        )
        raise typer.Exit(code=1)

    # Handle once/continuous conflict
    if continuous:
        once = False

    # Validate build_docker requirements
    if build_docker and not docker_username:
        console.print(
            "[red]Error: --docker-username is required when using --build-docker[/red]"
        )
        raise typer.Exit(code=1)

    # Validate output directory
    output_path = Path(output)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build pipeline config
    languages = (
        [language.lower()]
        if language
        else ["python", "javascript", "typescript", "rust", "go"]
    )
    difficulty_filter = difficulty.lower() if difficulty else None

    try:
        filter_config = json.loads(filter_json)
    except json.JSONDecodeError:
        console.print("[red]Error: Invalid JSON in --filter option[/red]")
        raise typer.Exit(code=1)

    difficulty_targets = DifficultyTargets(targets=filter_config)

    config = SwePipelineConfig(
        max_candidates=max_candidates,
        max_tasks=limit,
        once=once,
        min_stars=min_stars,
        languages=languages,
        difficulty_filter=difficulty_filter,
        difficulty_targets=difficulty_targets,
    )

    # Get GitHub token from environment
    github_token = os.environ.get("GITHUB_TOKEN", "")

    # Display configuration
    console.print("[bold blue]SWE-Forge Mine Configuration[/bold blue]")
    console.print(f"  Repository: {repo or 'All repositories (from GH Archive)'}")
    console.print(f"  Limit: {limit} tasks")
    console.print(f"  Output: {output}")
    console.print(f"  Difficulty: {difficulty or 'All'}")
    console.print(f"  Model: {model}")
    console.print(f"  Mode: {'Continuous' if continuous else 'Once'}")
    console.print(f"  Min Stars: {min_stars}")
    console.print(f"  Language: {language or 'python'}")
    console.print(f"  Filter: {filter_json}")
    console.print(f"  Parallel: {parallel} containers")
    if build_docker:
        console.print(f"  Build Docker: Yes (username: {docker_username})")
        console.print(f"  Push Images: {'Yes' if docker_push else 'No'}")
    console.print()

    # Run the pipeline
    try:
        result = asyncio.run(_run_pipeline(github_token, config, repo, verbose, model))

        if result.tasks:
            # Export results
            export_jsonl(result.tasks, output_path)
            console.print(
                f"\n[green]Exported {len(result.tasks)} tasks to {output}[/green]"
            )

            if output_folder:
                from swe_forge.export.workspace import export_tasks_to_workspace

                export_tasks_to_workspace(
                    result.tasks, output_folder, docker_username=docker_username
                )
                console.print(f"[green]Workspace export: {output_folder}[/green]")

            # Build Docker images if requested
            if build_docker and docker_username:
                console.print("\n[bold]Building Docker images...[/bold]")
                build_results = asyncio.run(
                    _build_docker_images(
                        result.tasks,
                        docker_username,
                        push=docker_push,
                        parallel=min(parallel, 4),  # Limit parallel builds
                        verbose=verbose,
                    )
                )
                successful = sum(1 for r in build_results if r.success)
                console.print(
                    f"[green]Built {successful}/{len(build_results)} images[/green]"
                )

                # Update workspace exports with pre-built image info
                if output_folder:
                    for task, build_result in zip(result.tasks, build_results):
                        if build_result.success and build_result.image_name:
                            _update_workspace_docker(
                                output_folder, task.id, build_result.image_name
                            )

            # Print summary
            if result.benchmark_metrics:
                metrics = result.benchmark_metrics
                console.print("\n[bold]Pipeline Summary:[/bold]")
                console.print(f"  Total candidates: {metrics.total_prefiltered}")
                console.print(f"  Enriched: {metrics.enriched_count}")
                console.print(f"  Filter passed: {metrics.filter_passed}")
                console.print(f"  Tasks extracted: {metrics.extraction_succeeded}")
                console.print(f"  Quality passed: {metrics.quality_passed}")
        else:
            console.print("\n[yellow]No tasks extracted[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Mining interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


async def _build_docker_images(
    tasks: list[SweTask],
    docker_username: str,
    *,
    push: bool = False,
    parallel: int = 2,
    verbose: bool = False,
):
    """Build Docker images for tasks with pre-installed dependencies."""
    from swe_forge.docker_test.image_builder import (
        build_images_for_tasks,
        task_to_dict,
    )
    from swe_forge.execution.docker_client import DockerClient

    task_dicts = [task_to_dict(t) for t in tasks]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress_task = progress.add_task("Building images...", total=len(tasks))

        async with DockerClient() as docker_client:
            results = await build_images_for_tasks(
                docker_client,
                task_dicts,
                docker_username,
                push=push,
                parallel=parallel,
            )

            for i, result in enumerate(results):
                status = "✅" if result.success else "❌"
                task_id = tasks[i].id
                if verbose or not result.success:
                    console.print(
                        f"  {status} {task_id}: {result.image_name or result.error}"
                    )
                progress.update(progress_task, advance=1)

    return results


def _update_workspace_docker(
    output_folder: Path, task_id: str, image_name: str
) -> None:
    """Update workspace.yaml with pre-built Docker image info."""
    import yaml

    workspace_path = output_folder / task_id / "workspace.yaml"
    if not workspace_path.exists():
        return

    with open(workspace_path, "r") as f:
        data = yaml.safe_load(f)

    # Update docker section to indicate pre-built image
    data["docker"] = {
        "image": image_name,
        "build": False,  # Already built
        "prebuilt": True,
    }

    with open(workspace_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


async def _run_pipeline(
    token: str,
    config: SwePipelineConfig,
    repo_filter: Optional[str],
    verbose: bool,
    model: str = "openai/gpt-5.4",
):
    """Run the SWE pipeline with progress tracking."""
    from dataclasses import dataclass, field

    @dataclass
    class PipelineResult:
        tasks: list = field(default_factory=list)
        benchmark_metrics: object = None

    # Create LLM client and TestGenerator for test generation
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    llm_client = None
    test_generator = None
    if openrouter_key:
        llm_client = OpenRouterClient(api_key=openrouter_key, default_model=model)
        test_generator = TestGenerator(llm=llm_client, model=model)
        config.test_generator = test_generator

    async with GitHubClient(token=token) as gh_client:
        gh_archive_client = GhArchiveClient(token=token) if not repo_filter else None

        tasks: list[SweTask] = []
        metrics = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Mining tasks...", total=config.max_tasks)

            async with SwePipeline(
                gh_client, gh_archive_client=gh_archive_client, config=config
            ) as pipeline:
                async for event in pipeline.run_with_progress():
                    if event.event_type == SwePipelineEventType.TASK_EXTRACTED:
                        task = event.data.get("task")
                        if task and isinstance(task, SweTask):
                            tasks.append(task)
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"Mined {len(tasks)} tasks",
                            )

                    elif event.event_type == SwePipelineEventType.PIPELINE_COMPLETED:
                        metrics = event.data.get("metrics")
                        progress.update(task_id, completed=len(tasks))

        return PipelineResult(tasks=tasks, benchmark_metrics=metrics)


@app.command("complete")
def mine_complete(
    repo: Annotated[
        str,
        typer.Option(
            "--repo",
            "-r",
            help="Target repository in owner/repo format",
        ),
    ],
    pr: Annotated[
        int,
        typer.Option(
            "--pr",
            "-p",
            help="Pull request number to mine",
        ),
    ],
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Output file path for JSONL results",
        ),
    ] = "./tasks.jsonl",
    llm_model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="LLM model for test generation",
        ),
    ] = "openai/gpt-5.4",
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose logging",
        ),
    ] = False,
) -> None:
    """Complete A-Z mining with Docker verification.

    Runs the full pipeline:
    1. Fetch PR from GitHub
    2. Detect language
    3. Discover commands from CI/CD
    4. Generate tests via LLM
    5. Verify tests fail before patch
    6. Apply patch
    7. Verify tests pass after patch
    8. Export validated task

    Only exports if ALL verification checks pass.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        console.print("[red]Error: GITHUB_TOKEN environment variable not set[/red]")
        raise typer.Exit(code=1)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    console.print("[bold blue]Complete Mining Pipeline[/bold blue]")
    console.print(f"  Repository: {repo}")
    console.print(f"  PR: #{pr}")
    console.print(f"  Model: {llm_model}")
    console.print(f"  Output: {output}")
    console.print()

    try:
        result = asyncio.run(
            _run_complete_mining(
                repo, pr, output, llm_model, github_token, openrouter_key
            )
        )

        if result:
            console.print(f"\n[green]✅ Task validated: {result.task.id}[/green]")
            console.print(
                f"   Tests before: {'FAILED' if not result.before_tests_passed else 'PASSED'}"
            )
            console.print(
                f"   Tests after: {'PASSED' if result.after_tests_passed else 'FAILED'}"
            )
            console.print(f"   Exported to: {output}")
        else:
            console.print("\n[red]❌ Task failed verification[/red]")
            raise typer.Exit(code=1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Mining interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(code=1)


async def _run_complete_mining(
    repo: str,
    pr_number: int,
    output: str,
    model: str,
    github_token: str,
    openrouter_key: str,
):
    """Run the complete mining pipeline."""
    from swe_forge.pipeline import CompleteMiningPipeline

    llm_client = None
    if openrouter_key:
        llm_client = OpenRouterClient(api_key=openrouter_key, default_model=model)

    async with GitHubClient(token=github_token) as gh:
        pipeline = CompleteMiningPipeline(
            gh_client=gh,
            llm_client=llm_client,
            model=model,
        )

        result = await pipeline.mine_pr(repo, pr_number)

        if result:
            from pathlib import Path
            from swe_forge.export.jsonl import export_jsonl

            export_jsonl([result.task], Path(output), append=True)

        return result


if __name__ == "__main__":
    app()
