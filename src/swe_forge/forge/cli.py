"""`swe-forge forge` CLI group.

Skeleton for the forge pipeline. The teacher/panel/pipeline subcommands are
stubbed here and filled in by later milestones. This module never imports the
repository's bespoke LLM clients or response cache.
"""

from __future__ import annotations

import typer
from rich.console import Console

from swe_forge.forge.config import ForgeSettings

app = typer.Typer(
    name="forge",
    help="Synthetic SWE benchmark generator (env-driven LiteLLM teacher/panel).",
    no_args_is_help=True,
)
console = Console()

_STUB_COMMANDS = ("llm-check", "panel-info", "panel-validate", "panel-rollouts")


@app.command()
def info() -> None:
    """Show forge LLM configuration status (never prints secrets)."""
    settings = ForgeSettings()
    console.print("[bold]forge configuration[/bold]")
    console.print(f"  teacher provider : {settings.teacher_llm_provider or '(unset)'}")
    console.print(f"  teacher model    : {settings.teacher_llm_model or '(unset)'}")
    console.print(f"  teacher base url : {settings.teacher_llm_base_url or '(unset)'}")
    console.print(
        f"  teacher api key  : {'set' if settings.teacher_llm_api_key else 'unset'}"
    )
    console.print(
        f"  panel override   : {'set' if settings.panel_llm_base_url or settings.panel_llm_api_key else 'inherits teacher'}"
    )


def _register_stub(name: str) -> None:
    @app.command(
        name=name, help=f"[stub] `forge {name}` is implemented in a later milestone."
    )
    def _stub() -> None:
        console.print(f"[yellow]`forge {name}` is not implemented yet (stub).[/yellow]")
        raise typer.Exit(code=1)


for _name in _STUB_COMMANDS:
    _register_stub(_name)
