"""`swe-forge forge` CLI group.

Hosts the forge pipeline subcommands. The teacher health-check (`llm-check`)
exercises the env-driven LiteLLM teacher client; panel subcommands are stubbed
here and filled in by later foundation features. This module never imports the
repository's bespoke LLM clients or response cache, and never prints secrets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console

from swe_forge.forge.adapters import (
    LanguageAdapter,
    NoAdapterFoundError,
    ParseError,
    build_default_registry,
)
from swe_forge.forge.config import ForgeSettings
from swe_forge.forge.envbuild import EnvBuilder
from swe_forge.forge.generators import (
    GenerationError,
    GenerationRequest,
    build_default_generator_registry,
)
from swe_forge.forge.models import (
    BaselineNotGreenError,
    EnvImage,
    require_green_baseline,
)
from swe_forge.forge.panel import (
    PANEL_BASE_URL_VAR,
    PANEL_API_KEY_VAR,
    TEACHER_API_KEY_VAR,
    TEACHER_BASE_URL_VAR,
    VALID_TIERS,
    PanelError,
    PanelModel,
    build_panel_from_env,
    resolve_panel_endpoint,
    run_rollouts,
    select_default_model,
    validate_model,
    validate_models,
)
from swe_forge.forge.secrets import key_fingerprint
from swe_forge.forge.sources import (
    UnknownRepoError,
    build_source_registry,
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

# Generic fallback model ids per check mode (model ids only; no provider host or
# brand). Used ONLY when the env-configured TEACHER_LLM_MODEL does not fit the
# requested --mode; otherwise the default is derived from the env so it targets a
# model the configured endpoint actually hosts.
_OPENAI_FALLBACK_MODEL = "openai/gpt-4o-mini"
_ANTHROPIC_FALLBACK_MODEL = "anthropic/claude-3-5-sonnet"


def _redact(text: str, secret: str) -> str:
    """Defensively strip a secret from any string before it is emitted."""
    if secret and secret in text:
        return text.replace(secret, "***redacted***")
    return text


def _fail(message: str, secret: str = "") -> NoReturn:
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


def _env_model_for_mode(env_model: str, env_provider: str) -> tuple[str, str]:
    """Derive ``(provider, full_model_id)`` from the configured env, or ``("", "")``.

    A provider-prefixed ``TEACHER_LLM_MODEL`` (``anthropic/<id>``) yields its own
    provider; an unprefixed id is paired with ``TEACHER_LLM_PROVIDER`` so the
    derived model still routes correctly. Returns empty strings when neither a
    model nor a usable provider is configured.
    """
    cleaned = (env_model or "").strip()
    if not cleaned:
        return "", ""
    if "/" in cleaned:
        provider = cleaned.partition("/")[0].strip().lower()
        return provider, cleaned
    provider = (env_provider or "").strip().lower()
    if provider:
        return provider, f"{provider}/{cleaned}"
    return "", ""


def _resolve_model(
    mode: str, model: str | None, env_model: str, env_provider: str = ""
) -> str:
    """Pick the model id for a check.

    An explicit ``--model`` always wins. Otherwise the default is derived from
    ``TEACHER_LLM_MODEL``/``TEACHER_LLM_PROVIDER`` when that model fits the
    requested ``--mode`` (so it targets a model the endpoint actually hosts); a
    generic fallback id is used only when the env model does not fit the mode.
    """
    if model:
        return model
    provider, full = _env_model_for_mode(env_model, env_provider)
    if full and provider == mode:
        return full
    return _OPENAI_FALLBACK_MODEL if mode == "openai" else _ANTHROPIC_FALLBACK_MODEL


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
        f"  teacher key fp   : {key_fingerprint(settings.teacher_llm_api_key) or '(unset)'}"
    )
    console.print(
        f"  panel override   : {'set' if settings.panel_llm_base_url or settings.panel_llm_api_key else 'inherits teacher'}"
    )


def _split_csv(value: str | None) -> list[str]:
    """Split a comma-separated CLI option into a clean list (empty when unset)."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _describe_adapter(
    adapter: LanguageAdapter, *, selection: list[str], paths: list[str]
) -> dict[str, object]:
    """Build the JSON-serializable metadata record for one adapter."""
    record: dict[str, object] = {
        "language": adapter.name,
        "base_image": adapter.base_image(),
        "test_command": adapter.test_command(),
        "mutation_tool": adapter.mutation_tool,
        "mutation_tools": list(adapter.mutation_tools),
    }
    if selection:
        record["test_command_selection"] = adapter.test_command(selection)
    if paths:
        record["classification"] = {path: adapter.is_test_file(path) for path in paths}
    return record


@app.command()
def detect(
    repo_path: str = typer.Argument(..., help="Path to the repository to inspect."),
    select: str | None = typer.Option(
        None, "--select", help="Comma-separated test selection for test_command."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Detect a repo's language adapter (mutually exclusive) and show build metadata."""
    path = Path(repo_path)
    if not path.exists():
        _fail(f"path does not exist: {repo_path}")

    registry = build_default_registry()
    table = {adapter.name: adapter.detect(path) for adapter in registry}
    matches = [name for name, matched in table.items() if matched]

    if len(matches) != 1:
        if not matches:
            reason = f"unsupported language: no adapter matched the repository at {repo_path}"
        else:
            reason = (
                f"ambiguous language: multiple adapters matched ({', '.join(matches)})"
            )
        payload: dict[str, object] = {
            "repo": str(path),
            "matched": False,
            "reason": reason,
            "detection": table,
        }
        if json_out:
            typer.echo(json.dumps(payload))
        else:
            console.print(f"[red]{reason}[/red]")
            console.print(f"detection: {table}")
        raise typer.Exit(code=1)

    adapter = registry.get(matches[0])
    selection = _split_csv(select)
    result: dict[str, object] = {
        "repo": str(path),
        "matched": True,
        "language": adapter.name,
        "base_image": adapter.base_image(),
        "install_commands": adapter.install_commands(path),
        "test_command": adapter.test_command(),
        "mutation_tool": adapter.mutation_tool,
        "detection": table,
    }
    if selection:
        result["test_command_selection"] = adapter.test_command(selection)

    if json_out:
        typer.echo(json.dumps(result))
        return
    console.print(f"[bold]detected language:[/bold] {adapter.name}")
    console.print(f"  base image       : {result['base_image']}")
    console.print(f"  install commands : {result['install_commands']}")
    console.print(f"  test command     : {result['test_command']}")
    if selection:
        console.print(f"  test (selection) : {result['test_command_selection']}")
    console.print(f"  mutation tool    : {result['mutation_tool']}")
    console.print(f"  detection table  : {table}")


@app.command(name="adapter-info")
def adapter_info(
    language: str | None = typer.Option(
        None, "--language", help="Restrict to one adapter (python|javascript|go)."
    ),
    select: str | None = typer.Option(
        None, "--select", help="Comma-separated selection for test_command."
    ),
    classify: str | None = typer.Option(
        None, "--classify", help="Comma-separated paths to classify via is_test_file."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Show static adapter metadata (base image, test command, mutation tool)."""
    registry = build_default_registry()
    selection = _split_csv(select)
    paths = _split_csv(classify)

    if language is not None:
        try:
            adapter = registry.get(language)
        except NoAdapterFoundError:
            _fail(
                f"no adapter for language {language!r}; "
                f"known: {', '.join(registry.names())}"
            )
        record = _describe_adapter(adapter, selection=selection, paths=paths)
        if json_out:
            typer.echo(json.dumps(record))
        else:
            console.print(record)
        return

    records = [
        _describe_adapter(adapter, selection=selection, paths=paths)
        for adapter in registry
    ]
    if json_out:
        typer.echo(json.dumps(records))
    else:
        for record in records:
            console.print(record)


_EXTENSION_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".mts": "javascript",
    ".cts": "javascript",
    ".go": "go",
}


@app.command(name="parse-symbols")
def parse_symbols_cmd(
    file_path: str = typer.Argument(..., help="Path to the source file to parse."),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Force an adapter (python|javascript|go); default: infer by extension.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Parse a source file's function/method symbols via its language adapter.

    Returns each symbol's name, kind, file, and 1-based inclusive line span. A
    file with no declarations yields an empty list; malformed source exits
    non-zero with a clean parse-error reason (never a traceback).
    """
    path = Path(file_path)
    if not path.is_file():
        _fail(f"file does not exist: {file_path}")

    registry = build_default_registry()
    lang = language or _EXTENSION_LANGUAGE.get(path.suffix.lower())
    if lang is None:
        _fail(
            f"cannot infer language from extension {path.suffix!r}; "
            f"pass --language ({', '.join(registry.names())})"
        )
    try:
        adapter = registry.get(lang)
    except NoAdapterFoundError:
        _fail(f"no adapter for language {lang!r}; known: {', '.join(registry.names())}")

    try:
        symbols = adapter.parse_symbols(path)
    except ParseError as exc:
        payload: dict[str, object] = {
            "file": str(path),
            "language": adapter.name,
            "parse_error": True,
            "reason": str(exc),
        }
        if json_out:
            typer.echo(json.dumps(payload))
        else:
            console.print(f"[red]parse error:[/red] {exc}")
        raise typer.Exit(code=1)

    records = [
        {
            "name": s.name,
            "kind": s.kind,
            "file": s.file,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "signature": s.signature,
        }
        for s in symbols
    ]
    result = {"file": str(path), "language": adapter.name, "symbols": records}
    if json_out:
        typer.echo(json.dumps(result))
        return
    console.print(f"[bold]{adapter.name}[/bold] symbols in {path} ({len(records)}):")
    for s in symbols:
        console.print(
            f"  {s.kind:8} {s.name}  L{s.start_line}-{s.end_line}"
            + (f"  {s.signature}" if s.signature else "")
        )


@app.command(name="sources-list")
def sources_list(
    language: str | None = typer.Option(
        None, "--language", help="Restrict to one language (python|javascript|go)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """List the curated Stage-0 seed repos and their contamination metadata."""
    registry = build_source_registry()
    if language is not None:
        specs = registry.by_language(language)
        if not specs:
            _fail(
                f"no curated repo for language {language!r}; "
                f"known: {', '.join(registry.languages())}"
            )
    else:
        specs = list(registry)

    records = [spec.to_dict() for spec in specs]
    if json_out:
        typer.echo(json.dumps(records))
        return

    console.print(
        f"[bold]source registry[/bold] ({len(records)} repos; "
        f"languages: {', '.join(registry.languages())})"
    )
    for spec in specs:
        console.print(
            f"  - {spec.repo_id} [{spec.language}] {spec.license} "
            f"@ {spec.commit[:12]} ({spec.commit_date}) "
            f"cap={spec.instance_cap} used={spec.used} remaining={spec.remaining}"
        )


@app.command(name="sources-acquire")
def sources_acquire(
    repo: str = typer.Option(..., "--repo", help="Repo id to acquire instances from."),
    count: int = typer.Option(
        1, "--count", help="Number of instance requests to issue (one process)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Request task instances from a repo, enforcing its per-repo cap.

    Issues ``--count`` requests against the repo within a single process so the
    cap is observable: once ``instance_cap`` is reached every further request is
    rejected with a clear reason and usage never exceeds the cap.
    """
    if count < 1:
        _fail(f"--count must be >= 1, got {count}")

    registry = build_source_registry()
    try:
        spec = registry.get(repo)
    except UnknownRepoError as exc:
        _fail(str(exc))

    attempts: list[dict[str, object]] = []
    accepted = 0
    rejected = 0
    for request_index in range(1, count + 1):
        grant = registry.acquire(repo)
        record = grant.to_dict()
        record["request"] = request_index
        attempts.append(record)
        if grant.accepted:
            accepted += 1
        else:
            rejected += 1

    payload: dict[str, object] = {
        "repo_id": spec.repo_id,
        "cap": spec.instance_cap,
        "requested": count,
        "accepted": accepted,
        "rejected": rejected,
        "used": spec.used,
        "remaining": spec.remaining,
        "attempts": attempts,
    }
    if json_out:
        typer.echo(json.dumps(payload))
        return

    console.print(
        f"[bold]{spec.repo_id}[/bold] cap={spec.instance_cap} "
        f"requested={count} accepted={accepted} rejected={rejected} "
        f"used={spec.used} remaining={spec.remaining}"
    )
    for record in attempts:
        if record["accepted"]:
            console.print(
                f"  #{record['request']} [green]accepted[/green] "
                f"instance={record['instance_index']} remaining={record['remaining']}"
            )
        else:
            console.print(
                f"  #{record['request']} [red]rejected[/red]: {record['reason']}"
            )


@app.command(name="build-env")
def build_env(
    repo: str | None = typer.Option(
        None, "--repo", help="Seed repo id from the Stage-0 source registry."
    ),
    path: str | None = typer.Option(
        None, "--path", help="Local repo checkout to build instead of a registry repo."
    ),
    repo_id: str | None = typer.Option(
        None,
        "--repo-id",
        help="Repo id to record for a --path build (default: dir name).",
    ),
    commit: str | None = typer.Option(
        None,
        "--commit",
        help="Commit to record for a --path build (default: git HEAD).",
    ),
    emit: str | None = typer.Option(
        None, "--emit", help="Write the EnvImage JSON here on a green build."
    ),
    advance: bool = typer.Option(
        False,
        "--advance",
        help="Gate the downstream stage on a green baseline (blocks a non-green repo).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Env-first build: one green-baseline Docker image per repo.

    Detects the repo's language, builds ONE image off the language-correct base,
    installs the repo's deps (incl. test deps), runs the repo's baseline suite and
    requires it GREEN, then persists an EnvImage with the proven baseline command
    and re-runs it in a fresh container to prove reproducibility. A RED baseline
    (or an install/build failure, reported distinctly) is rejected with cleanup
    and emits no green EnvImage. With --advance, a non-green repo is blocked from
    advancing past Stage 1.
    """
    if repo and path:
        _fail("pass exactly one of --repo or --path, not both")
    if not repo and not path:
        _fail("provide --repo <id> (registry) or --path <local repo>")

    builder = EnvBuilder()
    if repo:
        registry = build_source_registry()
        try:
            spec = registry.get(repo)
        except UnknownRepoError as exc:
            _fail(str(exc))
        result = builder.build(spec)
    else:
        assert path is not None
        target = Path(path)
        if not target.is_dir():
            _fail(f"path does not exist or is not a directory: {path}")
        result = builder.build_from_path(
            target,
            repo_id=repo_id or target.resolve().name,
            commit=commit or "",
        )

    if result.success and emit and result.env_image is not None:
        Path(emit).write_text(
            json.dumps(result.env_image.to_dict(), indent=2), encoding="utf-8"
        )

    advance_status: dict[str, object] | None = None
    if advance:
        try:
            require_green_baseline(result.env_image)
        except BaselineNotGreenError as exc:
            advance_status = {"advanced": False, "reason": str(exc)}
        else:
            advance_status = {
                "advanced": True,
                "reason": "stage-1 precondition satisfied: green baseline present",
            }

    payload = result.to_dict()
    if advance_status is not None:
        payload["advance"] = advance_status

    if json_out:
        typer.echo(json.dumps(payload))
    else:
        if result.success:
            console.print(
                f"[green]baseline green[/green] {result.repo_id} "
                f"[{result.language}] -> image {result.image_tag}"
            )
            if result.env_image is not None:
                console.print(
                    f"  baseline cmd : {result.env_image.baseline_test_command}"
                )
                console.print(f"  proof        : {result.env_image.baseline_summary}")
        else:
            console.print(
                f"[red]rejected[/red] {result.repo_id} [{result.language}] "
                f"at stage {result.stage} ({result.failure_kind}): {result.reason}"
            )
        if advance_status is not None:
            colour = "green" if advance_status["advanced"] else "red"
            console.print(
                f"  advance      : [{colour}]{advance_status['reason']}[/{colour}]"
            )

    # Exit non-zero when the build was rejected OR when --advance was blocked, so
    # callers/CI can gate on a green baseline.
    if not result.success or (
        advance_status is not None and not advance_status["advanced"]
    ):
        raise typer.Exit(code=1)


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
    env_provider = os.environ.get("TEACHER_LLM_PROVIDER", "")

    effective_base_url = base_url if base_url is not None else env_base_url
    effective_model = _resolve_model(mode, model, env_model, env_provider)
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
        # Emit base_url + a non-reversible key fingerprint per model so endpoint
        # inheritance/override stays verifiable WITHOUT ever printing the raw key.
        typer.echo(json.dumps([m.to_dict() for m in panel]))
        return
    override = bool(
        os.environ.get(PANEL_BASE_URL_VAR) or os.environ.get(PANEL_API_KEY_VAR)
    )
    console.print("[bold]forge panel[/bold]")
    console.print(f"  endpoint   : {base_url or '(unset)'}")
    console.print(f"  api key    : {'set' if api_key else 'unset'}")
    console.print(f"  key fp     : {key_fingerprint(api_key) or '(unset)'}")
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
    if k < 0:
        # Validate the arg up front so a negative k surfaces a clean, secret-free
        # message instead of a PanelError traceback escaping from run_rollouts.
        _fail(f"--k must be non-negative (>= 0), got {k}")

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
    except PanelError as exc:
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


@app.command(name="generate")
def generate(
    path: str = typer.Option(..., "--path", help="Local repo checkout to mutate."),
    out: str = typer.Option(
        ..., "--out", help="Output dir for candidate.json + mutation/oracle patches."
    ),
    generator: str = typer.Option(
        "ast_mutation", "--generator", help="Bug generator to run."
    ),
    file: str | None = typer.Option(
        None, "--file", help="Repo-relative target file (default: auto-discover)."
    ),
    symbol: str | None = typer.Option(
        None, "--symbol", help="Target symbol name (default: auto-select)."
    ),
    op: str | None = typer.Option(
        None,
        "--op",
        help="Operator: operator_swap | off_by_one | branch_removal (default: any).",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for deterministic selection."),
    language: str | None = typer.Option(
        None, "--language", help="Force an adapter (default: detect from the repo)."
    ),
    env: str | None = typer.Option(
        None, "--env", help="EnvImage JSON to enforce the green-baseline precondition."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Generate a bug Candidate (forward mutation + inverse gold oracle).

    Writes ``candidate.json`` plus ``mutation.patch`` and ``oracle.patch`` to the
    output dir on success. The generator self-validates the round-trip (mutation
    then oracle restores every touched file byte-for-byte) before emitting; an
    invalid generation fails non-zero and writes NO candidate artifact. With the
    same repo+target+seed, deterministic generators reproduce identical patches.
    """
    repo_root = Path(path)
    if not repo_root.is_dir():
        _fail(f"path does not exist or is not a directory: {path}")

    registry = build_default_registry()
    if language is not None:
        try:
            adapter = registry.get(language)
        except NoAdapterFoundError:
            _fail(
                f"no adapter for language {language!r}; "
                f"known: {', '.join(registry.names())}"
            )
    else:
        try:
            adapter = registry.detect(repo_root)
        except NoAdapterFoundError as exc:
            _fail(str(exc))

    gen_registry = build_default_generator_registry()
    try:
        bug_generator = gen_registry.get(generator)
    except KeyError as exc:
        _fail(str(exc))

    env_image: EnvImage | None = None
    if env is not None:
        env_path = Path(env)
        if not env_path.is_file():
            _fail(f"env image file not found: {env}")
        env_image = EnvImage.from_dict(json.loads(env_path.read_text(encoding="utf-8")))

    request = GenerationRequest(
        repo_root=repo_root,
        seed=seed,
        file=file,
        symbol=symbol,
        op=op,
        env_image=env_image,
    )
    try:
        candidate = bug_generator.generate(request, adapter)
    except (GenerationError, BaselineNotGreenError) as exc:
        # Fail cleanly and emit NO candidate artifact.
        _fail(f"{generator}: {exc}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mutation_path = out_dir / "mutation.patch"
    oracle_path = out_dir / "oracle.patch"
    candidate_path = out_dir / "candidate.json"
    mutation_path.write_text(candidate.mutation_patch, encoding="utf-8")
    oracle_path.write_text(candidate.oracle_patch, encoding="utf-8")
    candidate_path.write_text(
        json.dumps(candidate.to_dict(), indent=2), encoding="utf-8"
    )

    mutation_sha = hashlib.sha256(candidate.mutation_patch.encode("utf-8")).hexdigest()
    oracle_sha = hashlib.sha256(candidate.oracle_patch.encode("utf-8")).hexdigest()
    payload: dict[str, object] = {
        "generator": generator,
        "language": candidate.language,
        "seed": seed,
        "out": str(out_dir),
        "candidate": candidate.to_dict(),
        "patch_sha256": {
            "mutation.patch": mutation_sha,
            "oracle.patch": oracle_sha,
        },
    }
    if json_out:
        typer.echo(json.dumps(payload))
        return
    console.print(
        f"[green]candidate[/green] {candidate.generator} [{candidate.language}] "
        f"-> {out_dir}"
    )
    console.print(f"  target   : {candidate.target.to_dict()}")
    console.print(f"  operator : {candidate.provenance.details.get('operator')}")
    console.print(f"  seed     : {seed}")
    console.print(
        f"  patches  : mutation.patch={mutation_sha[:12]} oracle.patch={oracle_sha[:12]}"
    )
