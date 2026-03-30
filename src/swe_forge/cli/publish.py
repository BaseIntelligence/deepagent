"""CLI command for publishing tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

console = Console()


def publish(
    tasks_dir: Annotated[
        Path,
        typer.Option("--tasks-dir", "-t", help="Path to tasks directory"),
    ] = Path("./tasks"),
    docker_user: Annotated[
        Optional[str],
        typer.Option("--docker-user", "-d", help="Docker Hub username"),
    ] = None,
    hf_dataset: Annotated[
        str,
        typer.Option("--hf-dataset", "-h", help="HuggingFace dataset repo"),
    ] = "cortexlm/swe-forge",
    push: Annotated[
        bool,
        typer.Option("--push", help="Actually push (default: dry-run)"),
    ] = False,
    limit: Annotated[
        Optional[int],
        typer.Option("--limit", "-l", help="Max tasks to process"),
    ] = None,
    parallel: Annotated[
        int,
        typer.Option("--parallel", "-p", help="Parallel Docker builds"),
    ] = 4,
    docker_only: Annotated[
        bool,
        typer.Option("--docker-only", help="Only build/push Docker"),
    ] = False,
    hf_only: Annotated[
        bool,
        typer.Option("--hf-only", help="Only upload to HuggingFace"),
    ] = False,
) -> None:
    """Publish tasks to Docker Hub and HuggingFace.

    Examples:
        # Dry run
        swe-forge publish --docker-user myuser

        # Build and push Docker images
        swe-forge publish --docker-user myuser --push

        # Upload to HuggingFace
        swe-forge publish --hf-dataset cortexlm/swe-forge --push --hf-only

        # Full pipeline
        swe-forge publish --docker-user myuser --hf-dataset cortexlm/swe-forge --push
    """
    if not tasks_dir.exists():
        console.print(f"[red]Error: Tasks directory not found: {tasks_dir}[/red]")
        raise typer.Exit(code=1)

    if not docker_user and not docker_only and not hf_only:
        hf_only = True

    if docker_user and not hf_only:
        from ..publish.docker_builder import build_docker_images

        console.print("\n[bold]Building Docker images...[/bold]")

        results = asyncio.run(
            build_docker_images(
                tasks_dir=tasks_dir,
                docker_user=docker_user,
                push=push,
                parallel=parallel,
                limit=limit,
            )
        )

        success = sum(1 for r in results if r.success)
        console.print(f"  [green]Built {success}/{len(results)} images[/green]")

        if push:
            pushed = sum(1 for r in results if r.success and r.push_url)
            console.print(f"  [green]Pushed {pushed} images to Docker Hub[/green]")

        for r in results:
            if not r.success:
                console.print(f"  [red]Failed: {r.task_id} - {r.error}[/red]")

    if hf_only or (not docker_only):
        from ..publish.parquet_converter import convert_tasks_to_parquet
        from ..publish.hf_uploader import upload_dataset

        console.print("\n[bold]Converting to Parquet...[/bold]")

        try:
            parquet_path = convert_tasks_to_parquet(tasks_dir)
            console.print(f"  [green]Created {parquet_path}[/green]")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            raise typer.Exit(code=1)

        if push:
            console.print("\n[bold]Uploading to HuggingFace...[/bold]")
            try:
                url = upload_dataset(parquet_path, hf_dataset)
                console.print(f"  [green]Uploaded to {url}[/green]")
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                raise typer.Exit(code=1)
        else:
            console.print("  [yellow]Dry-run: Use --push to actually upload[/yellow]")

    console.print("\n[bold green]Done![/bold green]")
