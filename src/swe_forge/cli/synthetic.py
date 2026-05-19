"""Synthetic task generation commands."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from swe_forge.export.jsonl import export_jsonl
from swe_forge.export.workspace import export_task_to_workspace
from swe_forge.synthetic.pipeline import create_feature_deletion_task

app = typer.Typer(name="synthetic", help="Generate synthetic SWE benchmark tasks")
console = Console()


def _current_commit(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@app.command()
def generate(
    repo_path: Annotated[
        Path,
        typer.Option(
            "--repo-path",
            help="Local checkout used to generate the synthetic task",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    repo: Annotated[
        str,
        typer.Option("--repo", help="GitHub repository in owner/repo format"),
    ],
    source_file: Annotated[
        Path,
        typer.Option("--source-file", help="Python file containing the target symbol"),
    ],
    symbol: Annotated[
        str,
        typer.Option("--symbol", help="Function or method body to delete"),
    ],
    fail_to_pass: Annotated[
        list[str],
        typer.Option(
            "--fail-to-pass",
            help="Test command that must fail after deletion and pass after oracle patch",
        ),
    ],
    output_folder: Annotated[
        Path,
        typer.Option("--output-folder", "-o", help="Workspace output directory"),
    ] = Path("./synthetic_tasks"),
    output_jsonl: Annotated[
        Path | None,
        typer.Option("--output-jsonl", help="Optional JSONL output path"),
    ] = None,
    base_commit: Annotated[
        str | None,
        typer.Option("--base-commit", help="Base commit; defaults to repo_path HEAD"),
    ] = None,
    pass_to_pass: Annotated[
        list[str] | None,
        typer.Option(
            "--pass-to-pass", help="Regression test command that must keep passing"
        ),
    ] = None,
    install_command: Annotated[
        list[str] | None,
        typer.Option(
            "--install-command", help="Install/setup command to run before tests"
        ),
    ] = None,
    task_id: Annotated[
        str | None,
        typer.Option("--task-id", help="Override generated task id"),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite existing task directory"),
    ] = False,
) -> None:
    """Create one Cursor-style synthetic feature-deletion task."""
    commit = base_commit or _current_commit(repo_path)
    task = create_feature_deletion_task(
        repo_root=repo_path,
        repo=repo,
        base_commit=commit,
        source_file=source_file,
        symbol=symbol,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass or [],
        install_commands=install_command or [],
        task_id=task_id,
    )

    task_dir = export_task_to_workspace(
        task,
        output_folder,
        overwrite=overwrite,
    )
    if output_jsonl:
        export_jsonl([task], output_jsonl)

    if task_dir is None:
        raise typer.Exit(code=1)
    console.print(f"[green]Generated synthetic task:[/green] {task_dir}")
