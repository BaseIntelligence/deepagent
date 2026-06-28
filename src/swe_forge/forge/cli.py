"""`swe-forge forge` CLI group.

Hosts the forge pipeline subcommands. The teacher health-check (`llm-check`)
exercises the env-driven LiteLLM teacher client; panel subcommands are stubbed
here and filled in by later foundation features. This module never imports the
repository's bespoke LLM clients or response cache, and never prints secrets.
"""

from __future__ import annotations

import asyncio
import json
import os

import typer
from rich.console import Console

from swe_forge.forge.config import ForgeSettings
from swe_forge.forge.panel import (
    PANEL_BASE_URL_VAR,
    PANEL_API_KEY_VAR,
    TEACHER_API_KEY_VAR,
    TEACHER_BASE_URL_VAR,
    VALID_TIERS,
    PanelModel,
    build_panel_from_env,
    resolve_panel_endpoint,
    run_rollouts,
    select_default_model,
    validate_model,
    validate_models,
)
from swe_forge.forge.teacher import (
    AgenticResult,
    LLMResult,
    MissingCredentialsError,
    ModelRoutingError,
    NormalizedToolCall,
    TeacherClient,
    resolve_routing,
)

app = typer.Typer(
    name="forge",
    help="Synthetic SWE benchmark generator (env-driven LiteLLM teacher/panel).",
    no_args_is_help=True,
)
console = Console()

# Default model ids per check mode (model ids only; no provider host/brand).
_OPENAI_CHECK_MODEL = "openai/gpt-4o-mini"
_ANTHROPIC_FALLBACK_MODEL = "anthropic/claude-3-5-sonnet"


def _redact(text: str, secret: str) -> str:
    """Defensively strip a secret from any string before it is emitted."""
    if secret and secret in text:
        return text.replace(secret, "***redacted***")
    return text


def _fail(message: str, secret: str = "") -> None:
    console.print(f"[red]error:[/red] {_redact(message, secret)}")
    raise typer.Exit(code=1)


def _emit(payload: object, *, json_out: bool) -> None:
    if json_out:
        typer.echo(json.dumps(payload))
    else:
        console.print(payload)


def _weather_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def _resolve_model(mode: str, model: str | None, configured: str) -> str:
    if model:
        return model
    if mode == "openai":
        return configured if configured.startswith("openai/") else _OPENAI_CHECK_MODEL
    return (
        configured if configured.startswith("anthropic/") else _ANTHROPIC_FALLBACK_MODEL
    )


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


@app.command(name="llm-check")
def llm_check(
    mode: str = typer.Option(
        "anthropic",
        help="Routing mode: 'anthropic' (host-only base) or 'openai' (<base>/v1).",
    ),
    prompt: str = typer.Option(
        "Reply with the single word: pong", help="Prompt for the text/tool/json call."
    ),
    model: str | None = typer.Option(
        None, help="Explicit provider-prefixed model id (overrides --mode default)."
    ),
    base_url: str | None = typer.Option(
        None, help="Override base URL (defaults to TEACHER_LLM_BASE_URL)."
    ),
    json_schema: str | None = typer.Option(
        None, help="JSON schema string -> request structured json output."
    ),
    tool_demo: str | None = typer.Option(
        None, help="Run a tool-calling demo (currently: 'weather')."
    ),
    agentic_demo: bool = typer.Option(
        False, "--agentic-demo", help="Run a 2-step agentic tool exchange demo."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Resolve routing only; make no network call."
    ),
    repeat: int = typer.Option(
        1, help="Repeat the text call N times to show calls are independent."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
    raw: bool = typer.Option(
        False, "--raw", help="Emit only the raw model text (for piping)."
    ),
    max_tokens: int = typer.Option(512, help="Max tokens per call."),
    timeout: float = typer.Option(120.0, help="Per-call timeout in seconds."),
    num_retries: int = typer.Option(3, help="Bounded retries for transient errors."),
) -> None:
    """Exercise the LiteLLM teacher client (text / json / tools / agentic / dry-run)."""
    # Resolve credentials from the process environment only (not a re-read of
    # .env) so that explicitly unsetting a var (e.g. `env -u TEACHER_LLM_API_KEY`)
    # is authoritative and the fail-fast path can name the missing variable.
    env_base_url = os.environ.get("TEACHER_LLM_BASE_URL", "")
    env_api_key = os.environ.get("TEACHER_LLM_API_KEY", "")
    env_model = os.environ.get("TEACHER_LLM_MODEL", "")

    effective_base_url = base_url if base_url is not None else env_base_url
    effective_model = _resolve_model(mode, model, env_model)
    secret = env_api_key

    # Dry-run: pure routing normalization, no credentials required, no call.
    if dry_run:
        try:
            routing = resolve_routing(effective_model, effective_base_url)
        except ModelRoutingError as exc:
            _fail(str(exc), secret)
        payload = {"mode": mode, **routing.to_dict(), "dry_run": True}
        _emit(payload, json_out=json_out)
        return

    client = TeacherClient(
        base_url=effective_base_url,
        api_key=env_api_key,
        model=effective_model,
        max_tokens=max_tokens,
        num_retries=num_retries,
        timeout=timeout,
    )

    try:
        if json_schema is not None:
            _run_json_schema(client, prompt, json_schema, raw=raw, json_out=json_out)
        elif tool_demo is not None:
            _run_tool_demo(client, prompt, tool_demo, json_out=json_out)
        elif agentic_demo:
            _run_agentic_demo(client, json_out=json_out)
        elif repeat > 1:
            _run_repeat(client, prompt, repeat, json_out=json_out)
        else:
            _run_text(client, prompt, raw=raw, json_out=json_out)
    except MissingCredentialsError as exc:
        _fail(str(exc), secret)
    except ModelRoutingError as exc:
        _fail(str(exc), secret)
    except typer.Exit:
        raise
    except Exception as exc:  # surface a clean, secret-free message
        _fail(f"LLM call failed: {type(exc).__name__}: {exc}", secret)


def _run_text(client: TeacherClient, prompt: str, *, raw: bool, json_out: bool) -> None:
    result: LLMResult = asyncio.run(client.complete_text(prompt))
    if raw:
        typer.echo(result.text)
        return
    payload = {**result.to_dict(), **client.routing.to_dict()}
    if not json_out:
        console.print(
            f"text={result.text!r} total_tokens={result.usage.total_tokens} "
            f"cost={result.cost}"
        )
        return
    _emit(payload, json_out=True)


def _run_json_schema(
    client: TeacherClient, prompt: str, schema_str: str, *, raw: bool, json_out: bool
) -> None:
    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError as exc:
        _fail(f"--json-schema is not valid JSON: {exc}")
    result = asyncio.run(client.complete_json(prompt, schema))
    if raw:
        typer.echo(result.text)
        return
    _emit({**result.to_dict(), **client.routing.to_dict()}, json_out=True)


def _run_tool_demo(
    client: TeacherClient, prompt: str, tool_demo: str, *, json_out: bool
) -> None:
    if tool_demo != "weather":
        _fail(f"unknown --tool-demo {tool_demo!r}; supported: 'weather'")
    result = asyncio.run(client.complete_with_tools(prompt, [_weather_tool()]))
    payload = {
        "tool_calls": [tc.to_dict() for tc in result.tool_calls],
        "usage": result.usage.to_dict(),
        "cost": result.cost,
        "finish_reason": result.finish_reason,
    }
    _emit(payload, json_out=True)


def _run_agentic_demo(client: TeacherClient, *, json_out: bool) -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "What's the weather in Paris? Use the get_weather tool, then tell me "
                "the temperature and conditions."
            ),
        }
    ]

    def executor(call: NormalizedToolCall) -> str:
        city = str(call.arguments.get("city", "Paris"))
        return f"It is 25 degrees Celsius and sunny in {city}."

    result: AgenticResult = asyncio.run(
        client.agentic_turn(messages, [_weather_tool()], executor)
    )
    _emit(result.to_dict(), json_out=True)


def _run_repeat(
    client: TeacherClient, prompt: str, repeat: int, *, json_out: bool
) -> None:
    async def run_all() -> list[dict[str, object]]:
        results = []
        for _ in range(repeat):
            result = await client.complete_text(prompt)
            results.append(result.to_dict())
        return results

    payload = asyncio.run(run_all())
    _emit(payload, json_out=True)


def _require_panel_creds(base_url: str, api_key: str) -> None:
    """Fail fast (naming the missing var) before any live panel call."""
    if not base_url:
        _fail(
            f"no panel base URL configured; set {PANEL_BASE_URL_VAR} "
            f"(or inherit it from {TEACHER_BASE_URL_VAR})"
        )
    if not api_key:
        _fail(
            f"no panel API key configured; set {PANEL_API_KEY_VAR} "
            f"(or inherit it from {TEACHER_API_KEY_VAR})",
            api_key,
        )


@app.command(name="panel-info")
def panel_info(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List the calibration panel (tiers, model ids, resolved endpoint)."""
    base_url, api_key = resolve_panel_endpoint()
    panel = build_panel_from_env()
    if json_out:
        # The api_key value is included so endpoint inheritance/override can be
        # verified; the human view below never prints it.
        typer.echo(json.dumps([m.to_dict(include_api_key=True) for m in panel]))
        return
    override = bool(
        os.environ.get(PANEL_BASE_URL_VAR) or os.environ.get(PANEL_API_KEY_VAR)
    )
    console.print("[bold]forge panel[/bold]")
    console.print(f"  endpoint   : {base_url or '(unset)'}")
    console.print(f"  api key    : {'set' if api_key else 'unset'}")
    console.print(
        f"  source     : {'PANEL_LLM_* override' if override else 'inherits teacher'}"
    )
    for model in panel:
        console.print(
            f"  - {model.id} ({model.tier}) -> {model.model_string} "
            f"@ {model.routing.api_base}"
        )


@app.command(name="panel-validate")
def panel_validate(
    models: str = typer.Option(
        ...,
        "--models",
        help="Comma-separated provider-prefixed model ids to probe.",
    ),
    prompt: str = typer.Option("ping", help="Probe prompt for the validation call."),
    max_tokens: int = typer.Option(8, help="Max tokens per validation probe."),
    timeout: float = typer.Option(60.0, help="Per-probe timeout in seconds."),
    num_retries: int = typer.Option(1, help="Bounded retries per probe."),
    concurrency: int = typer.Option(4, help="Max concurrent probes."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Validate model ids with one live probe each (no bulk rollouts)."""
    base_url, api_key = resolve_panel_endpoint()
    _require_panel_creds(base_url, api_key)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        _fail("no model ids supplied; pass --models a,b,c")

    try:
        results = asyncio.run(
            validate_models(
                model_list,
                base_url=base_url,
                api_key=api_key,
                prompt=prompt,
                concurrency=concurrency,
                max_tokens=max_tokens,
                num_retries=num_retries,
                timeout=timeout,
            )
        )
    except MissingCredentialsError as exc:
        _fail(str(exc), api_key)

    if json_out:
        typer.echo(json.dumps([r.to_dict() for r in results]))
    else:
        for r in results:
            status = "[green]valid[/green]" if r.valid else "[red]invalid[/red]"
            suffix = "" if r.valid else f" ({r.error})"
            console.print(f"  {status} {r.model}{suffix}")

    if not all(r.valid for r in results):
        raise typer.Exit(code=1)


@app.command(name="panel-rollouts")
def panel_rollouts(
    k: int = typer.Option(3, "--k", help="Number of independent rollouts to issue."),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Provider-prefixed model id (default: a frontier panel model).",
    ),
    tier: str = typer.Option(
        "frontier", help="Tier to pick a default model from when --model is unset."
    ),
    prompt: str = typer.Option(
        "Reply with a random 6-digit number and nothing else.",
        "--prompt",
        help="The task prompt issued on every rollout.",
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", help="Max concurrent in-flight rollouts."
    ),
    max_tokens: int = typer.Option(64, help="Max tokens per rollout."),
    timeout: float = typer.Option(120.0, help="Per-rollout timeout in seconds."),
    num_retries: int = typer.Option(2, help="Bounded retries per rollout."),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate the model id with one probe before the k-burst.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Issue k independent, uncached rollouts of a task on a panel model."""
    base_url, api_key = resolve_panel_endpoint()
    _require_panel_creds(base_url, api_key)

    if model is not None:
        if tier not in VALID_TIERS:
            _fail(f"--tier must be one of {VALID_TIERS}, got {tier!r}")
        try:
            target = PanelModel(
                id=model,
                model_string=model,
                tier=tier,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as exc:
            _fail(str(exc), api_key)
    else:
        target = select_default_model(build_panel_from_env(), tier=tier)

    if validate:
        try:
            check = asyncio.run(
                validate_model(
                    target.model_string,
                    base_url=base_url,
                    api_key=api_key,
                    timeout=timeout,
                )
            )
        except MissingCredentialsError as exc:
            _fail(str(exc), api_key)
        if not check.valid:
            _fail(
                f"model {target.model_string!r} failed validation: {check.error}",
                api_key,
            )

    try:
        results = asyncio.run(
            run_rollouts(
                prompt,
                target,
                k,
                concurrency=concurrency,
                max_tokens=max_tokens,
                num_retries=num_retries,
                timeout=timeout,
            )
        )
    except MissingCredentialsError as exc:
        _fail(str(exc), api_key)
    except ModelRoutingError as exc:
        _fail(str(exc), api_key)

    payload = [r.to_dict() for r in results]
    if json_out:
        typer.echo(json.dumps(payload))
        return
    console.print(
        f"[bold]{len(results)} rollouts[/bold] on {target.model_string} "
        f"({target.tier}) @ {target.routing.api_base}"
    )
    for r in results:
        if r.error:
            console.print(f"  #{r.index} [red]error[/red]: {r.error}")
        else:
            console.print(
                f"  #{r.index} total_tokens={r.usage.total_tokens} cost={r.cost} "
                f"text={r.text!r}"
            )
