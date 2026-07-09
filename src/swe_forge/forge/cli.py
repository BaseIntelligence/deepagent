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
import shutil
import tempfile
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
from swe_forge.forge.calibrate.filter import (
    DEFAULT_BAND_HIGH,
    DEFAULT_DISCRIMINATION_THRESHOLD,
    BandFilterConfig,
    BandFilterError,
    apply_band_filter,
)
from swe_forge.forge.calibrate.irt import (
    DEFAULT_TIER_ABILITIES,
    IrtError,
    build_calibration_report,
    pass_at_k,
)
from swe_forge.forge.calibrate.pipeline import (
    CalibrationOutcome,
    run_calibration,
)
from swe_forge.forge.calibrate.runner import CalibrationRunnerError
from swe_forge.forge.calibrate.solver import (
    DEFAULT_MAX_TOKENS as SOLVER_DEFAULT_MAX_TOKENS,
)
from swe_forge.forge.calibrate.solver import (
    DEFAULT_MAX_TURNS,
    AgenticSolver,
    SolverError,
    run_solver_rollout,
)
from swe_forge.forge.config import ForgeSettings
from swe_forge.forge.envbuild import EnvBuilder
from swe_forge.forge.export import (
    ExportRequest,
    export_batch,
)
from swe_forge.forge.gold_eval import (
    DEFAULT_DETERMINISM_RUNS,
    GoldEvalError,
    GoldEvalReport,
    run_gold_eval,
)
from swe_forge.forge.gold_eval import (
    DEFAULT_TIMEOUT as GOLD_EVAL_TIMEOUT,
)
from swe_forge.forge.report import (
    DEFAULT_FRONTIER_THRESHOLD,
    BenchmarkReport,
    GoldSummary,
    ReportError,
    build_benchmark_report,
    write_report,
)
from swe_forge.forge.pilot import (
    DEFAULT_GENERATORS_BY_LANGUAGE,
    PilotError,
    build_pilot_plans,
    default_pilot_config,
    run_pilot,
)
from swe_forge.forge.generators import (
    GenerationError,
    GenerationRequest,
    build_default_generator_registry,
    run_menu_selfcheck,
    verify_candidate_roundtrip,
)
from swe_forge.forge.models import (
    SUPPORTED_LANGUAGES,
    BaselineNotGreenError,
    CalibrationReport,
    Candidate,
    EnvImage,
    GeneratedSpec,
    ModelError,
    ModelSolveRecord,
    OracleReport,
    require_green_baseline,
)
from swe_forge.forge.oracle import (
    DEFAULT_FLAKINESS_RUNS,
    DEFAULT_KILL_THRESHOLD,
    DEFAULT_MAX_STRENGTHEN_ROUNDS,
    DEFAULT_MAX_SYNTHESIS_ROUNDS,
    DEFAULT_NUM_ALTERNATIVES,
    DEFAULT_NUM_VARIANTS,
    AltCorrectError,
    DifferentialError,
    EstablishError,
    FlakinessError,
    HiddenTest,
    HiddenTestFile,
    LeakError,
    MutationError,
    OraclePipelineError,
    run_alt_correct_gate,
    run_differential_gate,
    run_establish_gate,
    run_flakiness_gate,
    run_leak_gate,
    run_mutation_gate,
    run_oracle_pipeline,
)
from swe_forge.forge.oracle.alt_correct_synth import TeacherAltCorrectGenerator
from swe_forge.forge.oracle.differential_synth import (
    DifferentialKillSynthesizer,
    TeacherVariantGenerator,
)
from swe_forge.forge.oracle.mutation_synth import MutationKillSynthesizer
from swe_forge.forge.oracle.test_synth import AgenticTestSynthesizer
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
from swe_forge.forge.spec import (
    F2PTrace,
    SpecError,
    TemplateSpecAuthor,
    generate_spec,
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


def _generate_params(pr_number: int | None, repo: str | None) -> dict[str, object]:
    """Build the generator ``params`` map from the generate CLI options."""
    params: dict[str, object] = {}
    if pr_number is not None:
        params["pr_number"] = pr_number
    if repo is not None:
        params["repo"] = repo
    return params


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
    pr_number: int | None = typer.Option(
        None, "--pr-number", help="pr_mirror: the merged PR number to invert."
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="pr_mirror: 'owner/name' (default: detect from the origin remote).",
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
        params=_generate_params(pr_number, repo),
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
    details = candidate.provenance.details
    operation = (
        details.get("operator") or details.get("operation") or candidate.generator
    )
    console.print(f"  operation: {operation}")
    console.print(f"  seed     : {seed}")
    console.print(
        f"  patches  : mutation.patch={mutation_sha[:12]} oracle.patch={oracle_sha[:12]}"
    )


def _write_menu_candidates(out_dir: Path, report) -> None:
    """Write per-cell Candidate JSON + mutation/oracle patches for a passing menu."""
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for cell in report.cells:
        if cell.candidate is None:
            continue
        cell_dir = candidates_dir / f"{cell.generator}__{cell.language}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "candidate.json").write_text(
            json.dumps(cell.candidate.to_dict(), indent=2), encoding="utf-8"
        )
        (cell_dir / "mutation.patch").write_text(
            cell.candidate.mutation_patch, encoding="utf-8"
        )
        (cell_dir / "oracle.patch").write_text(
            cell.candidate.oracle_patch, encoding="utf-8"
        )


@app.command(name="gen-menu")
def gen_menu(
    out: str = typer.Option(
        ..., "--out", help="Output dir for coverage.json + per-cell candidates."
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for deterministic selection."),
    no_go: bool = typer.Option(
        False, "--no-go", help="Skip Go cells even when the toolchain is available."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Run the cross-generator menu self-validation and record the coverage matrix.

    Exercises all six generators behind the uniform interface on built-in
    fixtures, independently re-verifies each Candidate's forward+inverse
    round-trip (byte-for-byte restore) and that the forward mutation is
    behavior-changing, checks the Candidate schema is complete, and writes a
    ``coverage[generator][language]`` matrix (``coverage.json``) plus per-cell
    Candidate artifacts. If any generator fails its self-validation the run
    aborts non-zero with an attributable reason and writes NO artifact.
    """
    from swe_forge.forge.generators.menu import build_menu_cell_specs

    specs = build_menu_cell_specs(include_go=False) if no_go else None
    workdir = Path(tempfile.mkdtemp(prefix="forge-menu-"))
    try:
        report = run_menu_selfcheck(workdir, seed=seed, specs=specs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if not report.ok:
        # Fail cleanly and emit NO artifact (no coverage.json, no candidates).
        reasons = "; ".join(report.reasons) or "menu self-validation failed"
        _fail(f"gen-menu: {reasons}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = out_dir / "coverage.json"
    coverage_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    _write_menu_candidates(out_dir, report)

    if json_out:
        typer.echo(json.dumps(report.to_dict()))
        return
    console.print(
        f"[green]menu self-check passed[/green] {len(report.cells)} cells "
        f"-> {coverage_path}"
    )
    for name in report.coverage:
        langs = report.coverage[name]
        cells = ", ".join(
            f"{lang}{'*' if entry.get('llm_backed') else ''}"
            f"={'ok' if entry.get('ok') else 'FAIL'}"
            for lang, entry in sorted(langs.items())
        )
        console.print(f"  {name:16s}: {cells}")
    console.print("  (* = llm-backed, offline-stubbed, non-Python exempt)")


def _load_candidate(candidate_file: str) -> Candidate:
    """Load and validate a ``candidate.json`` from disk, or fail cleanly."""
    path = Path(candidate_file)
    if not path.is_file():
        _fail(f"candidate file not found: {candidate_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Candidate.from_dict(data)
    except (json.JSONDecodeError, KeyError, ModelError) as exc:
        _fail(f"invalid candidate.json: {exc}")


def _load_trace(trace_file: str) -> F2PTrace:
    """Load an F2P failure trace JSON from disk, or fail cleanly."""
    path = Path(trace_file)
    if not path.is_file():
        _fail(f"trace file not found: {trace_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"invalid trace JSON: {exc}")
    if not isinstance(data, dict):
        _fail("trace JSON must be an object with a 'tests'/'fail_to_pass' list")
    return F2PTrace.from_dict(data)


@app.command(name="spec")
def spec(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    path: str = typer.Option(
        ..., "--path", help="Pristine repo checkout (the gold target source)."
    ),
    trace: str = typer.Option(
        ...,
        "--trace",
        help="F2P failure-trace JSON ({'tests':[{name,message,expected,observed}]}).",
    ),
    out: str = typer.Option(..., "--out", help="Output dir for spec.json."),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Use the deterministic, trace-derived template author (no LLM).",
    ),
    language: str | None = typer.Option(
        None, "--language", help="Force an adapter (default: the candidate's language)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Build a GeneratedSpec by test-conditioned backtranslation of the F2P trace.

    Loads a valid Candidate and confirms its forward+inverse round-trip against
    the pristine checkout (a spec is only emitted alongside a valid Candidate),
    derives the interface block from the real gold API, drafts the problem
    statement + requirements from the F2P failure trace (never the diff), grounds
    each requirement in a named F2P test, and runs a leak scan over all three
    fields. On success it writes ``spec.json``; any failure exits non-zero and
    writes NO artifact.
    """
    candidate_obj = _load_candidate(candidate)
    f2p_trace = _load_trace(trace)

    repo_root = Path(path)
    if not repo_root.is_dir():
        _fail(f"path does not exist or is not a directory: {path}")

    registry = build_default_registry()
    adapter_name = language or candidate_obj.language
    try:
        adapter = registry.get(adapter_name)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {adapter_name!r}; "
            f"known: {', '.join(registry.names())}"
        )

    # A spec is only ever produced for a Candidate that passes its
    # forward+inverse self-validation (VAL-GEN-017).
    rt = verify_candidate_roundtrip(repo_root, candidate_obj)
    if not rt.ok:
        _fail(f"candidate does not round-trip; emitting no spec: {rt.reason}")

    spec_author = TemplateSpecAuthor() if offline else None
    try:
        generated = generate_spec(
            candidate_obj, f2p_trace, repo_root, adapter, author=spec_author
        )
    except (SpecError, MissingCredentialsError, ModelRoutingError) as exc:
        # Fail cleanly and emit NO spec artifact.
        _fail(f"spec: {exc}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "spec.json"
    spec_path.write_text(json.dumps(generated.to_dict(), indent=2), encoding="utf-8")

    payload: dict[str, object] = {
        "out": str(out_dir),
        "spec": generated.to_dict(),
        "candidate_round_trip_ok": rt.ok,
    }
    if json_out:
        typer.echo(json.dumps(payload))
        return
    console.print(
        f"[green]spec[/green] {candidate_obj.generator} [{candidate_obj.language}] "
        f"-> {spec_path}"
    )
    console.print(f"  requirements: {len(generated.requirements)}")
    details = generated.provenance.details
    console.print(f"  authoring   : {details.get('authoring_mode')}")
    f2p_tests = details.get("f2p_tests", [])
    names = ", ".join(str(t) for t in f2p_tests) if isinstance(f2p_tests, list) else ""
    console.print(f"  f2p tests   : {names}")


def _load_env_image(env_file: str) -> EnvImage:
    """Load and validate an ``EnvImage`` (Stage 1 build) from disk, or fail."""
    path = Path(env_file)
    if not path.is_file():
        _fail(f"env image file not found: {env_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return EnvImage.from_dict(data)
    except (json.JSONDecodeError, KeyError, ModelError) as exc:
        _fail(f"invalid env image JSON: {exc}")


def _load_provided_tests(tests_file: str | None) -> list[HiddenTest]:
    """Load caller-declared (intended) hidden F2P tests from disk, or ``[]``."""
    if not tests_file:
        return []
    path = Path(tests_file)
    if not path.is_file():
        _fail(f"tests file not found: {tests_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"invalid tests JSON: {exc}")
    raw = data.get("tests", []) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        _fail("tests JSON must be a list (or {'tests': [...]}) of hidden tests")
    out: list[HiddenTest] = []
    for item in raw:
        if not isinstance(item, dict) or not str(item.get("test_id", "")).strip():
            _fail("each hidden test needs a non-empty 'test_id'")
        files = tuple(
            HiddenTestFile(path=str(f["path"]), content=str(f.get("content", "")))
            for f in item.get("files", [])
            if isinstance(f, dict) and str(f.get("path", "")).strip()
        )
        out.append(
            HiddenTest(
                test_id=str(item["test_id"]),
                files=files,
                origin=str(item.get("origin", "provided")),
            )
        )
    return out


@app.command(name="oracle-establish")
def oracle_establish(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    tests: str | None = typer.Option(
        None,
        "--tests",
        help="Optional caller-declared hidden tests JSON ({'tests':[{test_id,files,origin}]}).",
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the OracleReport JSON here (file or dir)."
    ),
    synthesize: bool = typer.Option(
        True,
        "--synthesize/--no-synthesize",
        help="Agentically synthesize hidden tests when none are provided/discriminate.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never call the LLM (no synthesis); rely on provided tests only.",
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Establish gate: confirm broken FAILS F2P, gold PASSES F2P, suite green as P2P.

    Runs in a throwaway DockerSandbox on the candidate's EnvImage. Confirms the
    P2P/regression suite is green on the gold and broken trees, then establishes
    discriminating hidden F2P tests (synthesized via the teacher when the fault is
    uncovered; the teacher proposes and Docker execution disposes). Writes an
    OracleReport and exits non-zero on a reject verdict (no F2P transition, gold
    fails to fix an F2P, or P2P not green on broken).
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    provided = _load_provided_tests(tests)

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    synthesizer = AgenticTestSynthesizer() if (synthesize and not offline) else None

    try:
        report = asyncio.run(
            run_establish_gate(
                candidate_obj,
                env_image,
                provided_tests=provided,
                synthesizer=synthesizer,
                adapter=adapter,
                command_timeout=timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"oracle establish: {exc}")
    except (EstablishError, DockerError) as exc:
        _fail(f"oracle establish: {exc}")

    if out:
        out_path = Path(out)
        if out_path.suffix == ".json":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            report_path = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            report_path = out_path / "oracle_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_out:
        typer.echo(json.dumps({"report": report.to_dict()}))
    else:
        colour = "green" if report.is_pass else "red"
        console.print(
            f"[{colour}]establish {report.verdict}[/{colour}] "
            f"{report.generator} [{report.language}]"
        )
        console.print(f"  fail_to_pass : {report.fail_to_pass}")
        console.print(f"  pass_to_pass : {report.pass_to_pass}")
        console.print(f"  test_files   : {[tf.path for tf in report.test_files]}")
        if report.reasons:
            console.print(f"  reasons      : {report.reasons}")

    if not report.is_pass:
        raise typer.Exit(code=1)


def _load_oracle_report(report_file: str) -> OracleReport:
    """Load and validate a prior-gate ``OracleReport`` from disk, or fail cleanly."""
    path = Path(report_file)
    if not path.is_file():
        _fail(f"oracle report file not found: {report_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return OracleReport.from_dict(data)
    except (json.JSONDecodeError, KeyError, ModelError) as exc:
        _fail(f"invalid oracle report JSON: {exc}")


def _write_oracle_report(out: str, report: OracleReport) -> None:
    """Write an ``OracleReport`` JSON to a file path or a directory."""
    out_path = Path(out)
    if out_path.suffix == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = out_path
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "oracle_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


@app.command(name="oracle-flakiness")
def oracle_flakiness(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the establish-gate OracleReport JSON (its F2P/P2P set).",
    ),
    runs: int = typer.Option(
        3, "--runs", help="Validation repeats in fresh containers (clamped up to 3)."
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the updated OracleReport JSON here (file or dir)."
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Flakiness gate: re-run the established F2P+P2P validation >=3x for determinism.

    Each repeat runs in a FRESH throwaway DockerSandbox on the candidate's
    EnvImage. A deterministic suite yields identical per-test verdicts and passes
    with ``flakiness_runs`` recorded (>=3). A non-deterministic F2P test is dropped
    from ``fail_to_pass``/``test_files``; if dropping removes the last F2P, or the
    P2P/regression suite itself is non-deterministic, the candidate is rejected
    with an attributable flakiness reason (non-zero exit).
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    establish_report = _load_oracle_report(report)

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    try:
        result = asyncio.run(
            run_flakiness_gate(
                candidate_obj,
                env_image,
                establish_report,
                runs=runs,
                adapter=adapter,
                command_timeout=timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"oracle flakiness: {exc}")
    except (FlakinessError, EstablishError, DockerError) as exc:
        _fail(f"oracle flakiness: {exc}")

    if out:
        _write_oracle_report(out, result)

    if json_out:
        typer.echo(json.dumps({"report": result.to_dict()}))
    else:
        colour = "green" if result.is_pass else "red"
        console.print(
            f"[{colour}]flakiness {result.verdict}[/{colour}] "
            f"{result.generator} [{result.language}]"
        )
        console.print(f"  flakiness_runs : {result.flakiness_runs}")
        console.print(f"  fail_to_pass   : {result.fail_to_pass}")
        flak = result.details.get("flakiness", {})
        dropped = flak.get("dropped", []) if isinstance(flak, dict) else []
        console.print(f"  dropped (flaky): {dropped}")
        if result.reasons:
            console.print(f"  reasons        : {result.reasons}")

    if not result.is_pass:
        raise typer.Exit(code=1)


@app.command(name="oracle-mutation")
def oracle_mutation(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the prior-gate OracleReport JSON (flakiness; its F2P/P2P set).",
    ),
    threshold: float = typer.Option(
        DEFAULT_KILL_THRESHOLD,
        "--threshold",
        help="Minimum mutant kill ratio required to pass (0 < t <= 1).",
    ),
    max_rounds: int = typer.Option(
        DEFAULT_MAX_SYNTHESIS_ROUNDS,
        "--max-rounds",
        help="Bounded auto-synthesis rounds before rejecting an inadequate suite.",
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the updated OracleReport JSON here (file or dir)."
    ),
    synthesize: bool = typer.Option(
        True,
        "--synthesize/--no-synthesize",
        help="Auto-synthesize survivor-killing tests when below threshold.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never call the LLM (no synthesis); measure adequacy of the suite as-is.",
    ),
    timeout: float = typer.Option(
        1200.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Mutation-adequacy gate: mutate the gold; require kill ratio >= threshold.

    Runs the adapter's mutation tool (cosmic-ray | Stryker | go-mutesting) against
    the gold target file(s) in throwaway DockerSandboxes with the established
    hidden tests present, recording ``mutants_total``/``mutants_killed``. If the
    kill ratio is below ``--threshold`` it auto-synthesizes survivor-killing tests
    (teacher proposes; each is confirmed to reduce survivors by re-measuring) until
    the threshold is met. Rejects (non-zero exit) when the bounded loop cannot
    reach the threshold, citing the surviving mutants.
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    prior_report = _load_oracle_report(report)

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    synthesizer = MutationKillSynthesizer() if (synthesize and not offline) else None

    try:
        result = asyncio.run(
            run_mutation_gate(
                candidate_obj,
                env_image,
                prior_report,
                synthesizer=synthesizer,
                threshold=threshold,
                max_rounds=max_rounds,
                adapter=adapter,
                command_timeout=timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"oracle mutation: {exc}")
    except (MutationError, FlakinessError, EstablishError, DockerError) as exc:
        _fail(f"oracle mutation: {exc}")

    if out:
        _write_oracle_report(out, result)

    if json_out:
        typer.echo(json.dumps({"report": result.to_dict()}))
    else:
        colour = "green" if result.is_pass else "red"
        console.print(
            f"[{colour}]mutation {result.verdict}[/{colour}] "
            f"{result.generator} [{result.language}]"
        )
        mut = result.details.get("mutation", {})
        tool = mut.get("tool", "") if isinstance(mut, dict) else ""
        console.print(
            f"  mutants        : {result.mutants_killed}/{result.mutants_total} "
            f"killed (tool={tool})"
        )
        console.print(f"  test_files     : {[tf.path for tf in result.test_files]}")
        if result.reasons:
            console.print(f"  reasons        : {result.reasons}")

    if not result.is_pass:
        raise typer.Exit(code=1)


@app.command(name="oracle-differential")
def oracle_differential(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the prior-gate OracleReport JSON (mutation; its F2P/P2P set).",
    ),
    num_variants: int = typer.Option(
        DEFAULT_NUM_VARIANTS,
        "--num-variants",
        help="Plausible-but-wrong variants the teacher proposes per candidate.",
    ),
    max_rounds: int = typer.Option(
        DEFAULT_MAX_STRENGTHEN_ROUNDS,
        "--max-rounds",
        help="Bounded test-strengthening rounds before rejecting a survivor.",
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the updated OracleReport JSON here (file or dir)."
    ),
    synthesize: bool = typer.Option(
        True,
        "--synthesize/--no-synthesize",
        help="Auto-strengthen with separating tests when a wrong variant survives.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never call the LLM (no variants/strengthening); gold-only sanity run.",
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Differential gate: only gold passes; every plausible-wrong variant fails.

    Generates plausible-but-wrong variants (PatchDiff-style) and runs the
    established F2P+P2P suite against each in throwaway DockerSandboxes. Gold must
    pass while every behaviorally-divergent variant fails >=1 test
    (``differential_pass=true``). A wrong variant that survives drives bounded
    test-strengthening (teacher proposes a separating test; execution confirms it
    fails the variant and passes gold) before the gate passes; an unresolvable
    survivor rejects (``differential_pass=false``, non-zero exit).
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    prior_report = _load_oracle_report(report)

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    variant_generator = TeacherVariantGenerator() if not offline else None
    synthesizer = (
        DifferentialKillSynthesizer() if (synthesize and not offline) else None
    )

    try:
        result = asyncio.run(
            run_differential_gate(
                candidate_obj,
                env_image,
                prior_report,
                variant_generator=variant_generator,
                synthesizer=synthesizer,
                num_variants=num_variants,
                max_rounds=max_rounds,
                adapter=adapter,
                command_timeout=timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"oracle differential: {exc}")
    except (
        DifferentialError,
        MutationError,
        FlakinessError,
        EstablishError,
        DockerError,
    ) as exc:
        _fail(f"oracle differential: {exc}")

    if out:
        _write_oracle_report(out, result)

    if json_out:
        typer.echo(json.dumps({"report": result.to_dict()}))
    else:
        colour = "green" if result.is_pass else "red"
        console.print(
            f"[{colour}]differential {result.verdict}[/{colour}] "
            f"{result.generator} [{result.language}]"
        )
        diff = result.details.get("differential", {})
        total = diff.get("variants_total", 0) if isinstance(diff, dict) else 0
        killed = (
            diff.get("final", {}).get("variants_killed", 0)
            if isinstance(diff, dict) and isinstance(diff.get("final"), dict)
            else 0
        )
        console.print(
            f"  variants       : {killed}/{total} separated from gold "
            f"(differential_pass={result.differential_pass})"
        )
        console.print(f"  test_files     : {[tf.path for tf in result.test_files]}")
        if result.reasons:
            console.print(f"  reasons        : {result.reasons}")

    if not result.is_pass:
        raise typer.Exit(code=1)


def _load_spec(spec_file: str | None) -> GeneratedSpec | None:
    """Load a ``spec.json`` (Stage 2 GeneratedSpec) from disk, or ``None``."""
    if not spec_file:
        return None
    path = Path(spec_file)
    if not path.is_file():
        _fail(f"spec file not found: {spec_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return GeneratedSpec.from_dict(data)
    except (json.JSONDecodeError, KeyError, ModelError) as exc:
        _fail(f"invalid spec.json: {exc}")


@app.command(name="oracle-alt-correct")
def oracle_alt_correct(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the prior-gate OracleReport JSON (differential; its F2P/P2P set).",
    ),
    spec: str | None = typer.Option(
        None,
        "--spec",
        help="Optional spec.json; its interface_block pins the public signatures.",
    ),
    num_alternatives: int = typer.Option(
        DEFAULT_NUM_ALTERNATIVES,
        "--num-alternatives",
        help="Genuinely-correct alternatives the teacher proposes per candidate.",
    ),
    relax: bool = typer.Option(
        False,
        "--relax/--no-relax",
        help="On over-fit, drop the offending hidden test(s) when safe (else reject).",
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the updated OracleReport JSON here (file or dir)."
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never call the LLM (no alternatives); gold-only sanity run.",
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Alt-correct gate: the suite must ACCEPT a genuinely-correct alternative.

    The teacher writes 1-2 genuinely-correct alternative implementations (same
    public Interface, different internals/symbol names) and the established
    F2P+P2P suite is run against each in throwaway DockerSandboxes. Each must pass
    (``alt_correct_accepted=true``); the Interface pinning prevents naming
    false-negatives. If a correct alternative is FAILED the suite is over-fit: the
    default rejects (``alt_correct_accepted=false``, non-zero exit) citing
    over-fit; ``--relax`` drops the offending hidden test(s) when that is safe
    (recording the relax action) else rejects.
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    prior_report = _load_oracle_report(report)
    spec_obj = _load_spec(spec)

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    alt_generator = TeacherAltCorrectGenerator() if not offline else None

    try:
        result = asyncio.run(
            run_alt_correct_gate(
                candidate_obj,
                env_image,
                prior_report,
                spec=spec_obj,
                alt_generator=alt_generator,
                num_alternatives=num_alternatives,
                relax=relax,
                adapter=adapter,
                command_timeout=timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"oracle alt-correct: {exc}")
    except (
        AltCorrectError,
        DifferentialError,
        MutationError,
        FlakinessError,
        EstablishError,
        DockerError,
    ) as exc:
        _fail(f"oracle alt-correct: {exc}")

    if out:
        _write_oracle_report(out, result)

    if json_out:
        typer.echo(json.dumps({"report": result.to_dict()}))
    else:
        colour = "green" if result.is_pass else "red"
        console.print(
            f"[{colour}]alt-correct {result.verdict}[/{colour}] "
            f"{result.generator} [{result.language}]"
        )
        alt = result.details.get("alt_correct", {})
        total = alt.get("alternatives_total", 0) if isinstance(alt, dict) else 0
        relax_info = alt.get("relax", {}) if isinstance(alt, dict) else {}
        relaxed = (
            bool(relax_info.get("succeeded")) if isinstance(relax_info, dict) else False
        )
        console.print(
            f"  alternatives   : {total} proposed "
            f"(alt_correct_accepted={result.alt_correct_accepted}, relaxed={relaxed})"
        )
        console.print(f"  test_files     : {[tf.path for tf in result.test_files]}")
        if result.reasons:
            console.print(f"  reasons        : {result.reasons}")

    if not result.is_pass:
        raise typer.Exit(code=1)


@app.command(name="oracle-leak")
def oracle_leak(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the prior-gate OracleReport JSON (alt-correct; its F2P/test set).",
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the updated OracleReport JSON here (file or dir)."
    ),
    no_sanitize: bool = typer.Option(
        False,
        "--no-sanitize",
        help="Detect only; never strip a removable leak (a finding then rejects).",
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Leak gate: the agent-facing tree must carry NO oracle/solution content.

    Materializes the broken (mutation-applied) tree from the candidate's EnvImage
    in a throwaway DockerSandbox, strips build/cache artifacts, then runs the leak
    auditor over it (oracle-snippet + forbidden-artifact scan) and confirms no
    hidden F2P/P2P test path/body is present. A clean tree passes
    (``leak_audit=clean``). A detected leak is stripped when it is a safely-
    removable standalone artifact (``leak_audit=sanitized: ...``, exit 0); an
    unremovable embedded leak rejects (``leak_audit=leak: ...``, non-zero exit)
    citing the marker.
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    prior_report = _load_oracle_report(report)

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    try:
        result = asyncio.run(
            run_leak_gate(
                candidate_obj,
                env_image,
                prior_report,
                adapter=adapter,
                sanitize=not no_sanitize,
                command_timeout=timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"oracle leak: {exc}")
    except (
        LeakError,
        AltCorrectError,
        DifferentialError,
        MutationError,
        FlakinessError,
        EstablishError,
        DockerError,
    ) as exc:
        _fail(f"oracle leak: {exc}")

    if out:
        _write_oracle_report(out, result)

    if json_out:
        typer.echo(json.dumps({"report": result.to_dict()}))
    else:
        colour = "green" if result.is_pass else "red"
        console.print(
            f"[{colour}]leak {result.verdict}[/{colour}] "
            f"{result.generator} [{result.language}]"
        )
        console.print(f"  leak_audit     : {result.leak_audit}")
        leak = result.details.get("leak", {})
        if isinstance(leak, dict):
            removed = leak.get("removed", [])
            normalized = leak.get("normalized", [])
            console.print(f"  sanitized      : {removed}")
            console.print(f"  normalized     : {normalized}")
        if result.reasons:
            console.print(f"  reasons        : {result.reasons}")

    if not result.is_pass:
        raise typer.Exit(code=1)


@app.command(name="oracle")
def oracle(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    spec: str | None = typer.Option(
        None,
        "--spec",
        help="Optional spec.json; its interface_block pins the alt-correct signatures.",
    ),
    tests: str | None = typer.Option(
        None,
        "--tests",
        help="Optional caller-declared hidden tests JSON ({'tests':[{test_id,files,origin}]}).",
    ),
    threshold: float = typer.Option(
        DEFAULT_KILL_THRESHOLD,
        "--threshold",
        help="Minimum mutant kill ratio required by the mutation gate (0 < t <= 1).",
    ),
    flakiness_runs: int = typer.Option(
        DEFAULT_FLAKINESS_RUNS,
        "--flakiness-runs",
        help="Determinism repeats in fresh containers (clamped up to 3).",
    ),
    num_variants: int = typer.Option(
        DEFAULT_NUM_VARIANTS,
        "--num-variants",
        help="Plausible-but-wrong variants the differential gate proposes.",
    ),
    num_alternatives: int = typer.Option(
        DEFAULT_NUM_ALTERNATIVES,
        "--num-alternatives",
        help="Genuinely-correct alternatives the alt-correct gate proposes.",
    ),
    max_mutation_rounds: int = typer.Option(
        DEFAULT_MAX_SYNTHESIS_ROUNDS,
        "--max-mutation-rounds",
        help="Bounded mutation auto-synthesis rounds before rejecting.",
    ),
    max_differential_rounds: int = typer.Option(
        DEFAULT_MAX_STRENGTHEN_ROUNDS,
        "--max-differential-rounds",
        help="Bounded differential test-strengthening rounds before rejecting.",
    ),
    relax: bool = typer.Option(
        False,
        "--relax/--no-relax",
        help="On alt-correct over-fit, drop the offending test when safe (else reject).",
    ),
    synthesize: bool = typer.Option(
        True,
        "--synthesize/--no-synthesize",
        help="Agentically synthesize tests/variants/alternatives via the teacher.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Never call the LLM (no synthesis/variants/alternatives).",
    ),
    no_sanitize: bool = typer.Option(
        False,
        "--no-sanitize",
        help="Leak gate detects only; never strip a removable leak (a finding rejects).",
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    mutation_timeout: float = typer.Option(
        1200.0, "--mutation-timeout", help="Mutation-gate Docker timeout (seconds)."
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the final OracleReport JSON here (file or dir)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Oracle pipeline: run every gate in order -> one OracleReport.

    Orchestrates establish -> flakiness -> mutation -> differential -> alt-correct
    -> leak in throwaway DockerSandboxes on the candidate's EnvImage. ``verdict``
    is ``pass`` only when EVERY gate passes with consistent fields (empty reasons,
    F2P non-empty, flakiness_runs>=3, kill ratio>=threshold, differential_pass,
    alt_correct_accepted, clean leak_audit); a single gate failure stops the
    pipeline and yields ``reject`` with attributable reasons citing the earliest
    failed gate (later gate fields are never credited). A reject exits non-zero so
    a rejected candidate is never advanced to export.
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    spec_obj = _load_spec(spec)
    provided = _load_provided_tests(tests)

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    use_llm = synthesize and not offline
    establish_synth = AgenticTestSynthesizer() if use_llm else None
    mutation_synth = MutationKillSynthesizer() if use_llm else None
    variant_gen = TeacherVariantGenerator() if not offline else None
    differential_synth = DifferentialKillSynthesizer() if use_llm else None
    alt_gen = TeacherAltCorrectGenerator() if not offline else None

    try:
        report = asyncio.run(
            run_oracle_pipeline(
                candidate_obj,
                env_image,
                provided_tests=provided,
                establish_synthesizer=establish_synth,
                mutation_synthesizer=mutation_synth,
                variant_generator=variant_gen,
                differential_synthesizer=differential_synth,
                alt_generator=alt_gen,
                spec=spec_obj,
                flakiness_runs=flakiness_runs,
                kill_threshold=threshold,
                max_mutation_rounds=max_mutation_rounds,
                num_variants=num_variants,
                max_differential_rounds=max_differential_rounds,
                num_alternatives=num_alternatives,
                relax_alt_correct=relax,
                sanitize=not no_sanitize,
                adapter=adapter,
                command_timeout=timeout,
                mutation_timeout=mutation_timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"oracle pipeline: {exc}")
    except (
        OraclePipelineError,
        LeakError,
        AltCorrectError,
        DifferentialError,
        MutationError,
        FlakinessError,
        EstablishError,
        DockerError,
    ) as exc:
        _fail(f"oracle pipeline: {exc}")

    if out:
        _write_oracle_report(out, report)

    if json_out:
        typer.echo(json.dumps({"report": report.to_dict()}))
    else:
        colour = "green" if report.is_pass else "red"
        console.print(
            f"[{colour}]oracle {report.verdict}[/{colour}] "
            f"{report.generator} [{report.language}]"
        )
        pipeline = report.details.get("pipeline", {})
        gates_run = pipeline.get("gates_run", []) if isinstance(pipeline, dict) else []
        failed = pipeline.get("failed_gate") if isinstance(pipeline, dict) else None
        console.print(f"  gates_run      : {gates_run}")
        console.print(f"  fail_to_pass   : {report.fail_to_pass}")
        console.print(f"  flakiness_runs : {report.flakiness_runs}")
        console.print(
            f"  mutants        : {report.mutants_killed}/{report.mutants_total} killed"
        )
        console.print(f"  differential   : {report.differential_pass}")
        console.print(f"  alt_correct    : {report.alt_correct_accepted}")
        console.print(f"  leak_audit     : {report.leak_audit}")
        if failed:
            console.print(f"  failed_gate    : {failed}")
        if report.reasons:
            console.print(f"  reasons        : {report.reasons}")

    if not report.is_pass:
        raise typer.Exit(code=1)


@app.command(name="solver-rollout")
def solver_rollout(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the oracle-pass OracleReport JSON (its F2P/P2P/test_files set).",
    ),
    spec: str = typer.Option(
        ...,
        "--spec",
        help="Path to the GeneratedSpec JSON (statement+requirements+interface ONLY).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Provider-prefixed solver model id (default: a frontier panel model).",
    ),
    tier: str = typer.Option(
        "frontier", help="Tier to pick a default model from when --model is unset."
    ),
    max_turns: int = typer.Option(
        DEFAULT_MAX_TURNS, "--max-turns", help="Bounded solver tool-loop budget."
    ),
    max_tokens: int = typer.Option(
        SOLVER_DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens per solver call."
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    emit_patch: str | None = typer.Option(
        None, "--emit-patch", help="Write the submitted unified diff to this file."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Run ONE agentic solver rollout on a broken task and score it (shared path).

    Sets the broken tree (mutation applied) in a throwaway DockerSandbox, hands the
    solver model ONLY the GeneratedSpec surface (statement+requirements+interface),
    runs the bash/read/write/finish tool loop, captures the finish-submitted patch,
    and scores it via the SAME Docker FAIL->PASS recipe the oracle gates use (a solve
    requires the FULL hidden test_files[] set to pass AND P2P/regression green).
    Submit-gated: a non-finish / empty / non-applying patch records solve=false
    without crashing. Exits non-zero only when the rollout could not run at all.
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    oracle_report = _load_oracle_report(report)
    spec_obj = _load_spec(spec)
    if spec_obj is None:
        _fail("a valid --spec (GeneratedSpec JSON) is required for a solver rollout")

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    base_url, api_key = resolve_panel_endpoint()
    _require_panel_creds(base_url, api_key)

    if model is not None:
        if tier not in VALID_TIERS:
            _fail(f"--tier must be one of {VALID_TIERS}, got {tier!r}")
        try:
            panel_model = PanelModel(
                id=model,
                model_string=model,
                tier=tier,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as exc:
            _fail(str(exc), api_key)
    else:
        panel_model = select_default_model(build_panel_from_env(), tier=tier)

    solver = AgenticSolver(
        client=panel_model.client(max_tokens=max_tokens, timeout=timeout),
        max_turns=max_turns,
        max_tokens=max_tokens,
    )

    try:
        outcome = asyncio.run(
            run_solver_rollout(
                candidate_obj,
                env_image,
                spec_obj,
                oracle_report,
                model=panel_model.model_string,
                solver=solver,
                adapter=adapter,
                command_timeout=timeout,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"solver rollout: {exc}", api_key)
    except (SolverError, DockerError) as exc:
        _fail(f"solver rollout: {exc}", api_key)

    if emit_patch and outcome.patch:
        patch_path = Path(emit_patch)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(outcome.patch, encoding="utf-8")

    if json_out:
        # The raw patch text is never echoed (it can be large); use --emit-patch.
        typer.echo(json.dumps({"outcome": outcome.to_dict()}))
    else:
        colour = "green" if outcome.solved else "yellow"
        console.print(
            f"[{colour}]solve={outcome.solved}[/{colour}] "
            f"model={outcome.model} finished={outcome.finished} turns={outcome.turns}"
        )
        console.print(f"  patch_bytes  : {len(outcome.patch)}")
        console.print(
            f"  usage/cost   : {outcome.usage.total_tokens} tok / {outcome.cost}"
        )
        console.print(f"  score        : {outcome.score.to_dict()}")
        if outcome.error:
            console.print(f"  error        : {outcome.error}")


def _load_solve_matrix(matrix_file: str) -> tuple[list[ModelSolveRecord], int]:
    """Load a per-model solve matrix and (recompute) the canonical pass@k.

    Accepts either a bare list of ``{model, tier, solves, k}`` rows or a
    ``CalibrationRun``-shaped object with a ``models`` list (the panel runner's
    JSON). ``pass_at_k`` is always recomputed from ``solves``/``k`` so the records
    carry the single-source estimator regardless of the input. Returns the
    records plus the inferred rollout budget ``k`` (the max per-model ``k``).
    """
    path = Path(matrix_file)
    if not path.is_file():
        _fail(f"solve-matrix file not found: {matrix_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"invalid solve-matrix JSON: {exc}")

    if isinstance(data, dict) and "models" in data:
        rows = data["models"]
    elif isinstance(data, list):
        rows = data
    else:
        _fail(
            "solve-matrix JSON must be a list of {model,tier,solves,k} rows "
            "or an object with a 'models' list"
        )

    if not isinstance(rows, list) or not rows:
        _fail("solve-matrix has no model rows")

    records: list[ModelSolveRecord] = []
    inferred_k = 0
    for raw in rows:
        if not isinstance(raw, dict):
            _fail(f"each solve-matrix row must be an object; got {type(raw).__name__}")
        try:
            solves = int(raw["solves"])
            k = int(raw["k"])
            records.append(
                ModelSolveRecord(
                    model=str(raw["model"]),
                    tier=str(raw["tier"]),
                    k=k,
                    solves=solves,
                    pass_at_k=pass_at_k(solves, k),
                )
            )
            inferred_k = max(inferred_k, k)
        except (KeyError, ValueError, ModelError) as exc:
            _fail(f"invalid solve-matrix row {raw!r}: {exc}")
    return records, inferred_k


@app.command(name="calibrate-irt")
def calibrate_irt(
    matrix: str = typer.Option(
        ...,
        "--matrix",
        help=(
            "Path to the per-model solve matrix JSON (a list of "
            "{model,tier,solves,k} rows or a CalibrationRun with a 'models' list)."
        ),
    ),
    language: str = typer.Option(
        "python", "--language", help="Task language for the CalibrationReport."
    ),
    difficulty_hint: str = typer.Option(
        "", "--difficulty-hint", help="Optional a-priori difficulty label."
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the CalibrationReport JSON to this path."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Fit the 2-parameter IRT over a panel solve matrix -> CalibrationReport.

    Pure offline math (no Docker, no LLM): turns the recorded per-model/per-rollout
    solves into the canonical per-model ``pass_at_k`` plus ``irt_difficulty`` and
    ``irt_discrimination`` (a steep, high-discrimination slope when only strong
    tiers solve; a flat, low-discrimination slope when tiers do not separate;
    well-defined sentinels for all-pass / all-fail). The ``band_verdict`` is left
    ``pending`` -- the band filter assigns the terminal keep/drop.
    """
    if language not in SUPPORTED_LANGUAGES:
        _fail(f"--language must be one of {SUPPORTED_LANGUAGES}; got {language!r}")

    records, inferred_k = _load_solve_matrix(matrix)
    try:
        report = build_calibration_report(
            language,
            records,
            k=inferred_k,
            difficulty_hint=difficulty_hint,
        )
    except IrtError as exc:
        _fail(f"IRT fit failed: {exc}")

    if out:
        out_path = Path(out)
        if out_path.suffix == ".json":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            report_path = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            report_path = out_path / "calibration_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if json_out:
        typer.echo(json.dumps(report.to_dict()))
        return

    console.print(
        f"[bold]calibration IRT[/bold] language={report.language} k={report.k} "
        f"(tier abilities {DEFAULT_TIER_ABILITIES})"
    )
    console.print(
        f"  irt_difficulty     : {report.irt_difficulty:+.4f}\n"
        f"  irt_discrimination : {report.irt_discrimination:.4f}\n"
        f"  frontier pass@k    : {report.frontier_pass_at_k():.4f}\n"
        f"  band_verdict       : {report.band_verdict}"
    )
    for rec in report.models:
        console.print(
            f"    {rec.tier:8s} {rec.model:32s} "
            f"solves={rec.solves}/{rec.k} pass@k={rec.pass_at_k:.4f}"
        )
    tier_rates = report.tier_pass_rates()
    if tier_rates:
        ordered = ", ".join(
            f"{tier}={tier_rates[tier]:.3f}"
            for tier in ("weak", "mid", "frontier")
            if tier in tier_rates
        )
        console.print(f"  tier pass-rates    : {ordered}")


def _load_calibration_report(report_file: str) -> CalibrationReport:
    """Load a :class:`CalibrationReport` JSON (as written by ``calibrate-irt``)."""
    path = Path(report_file)
    if not path.is_file():
        _fail(f"calibration report file not found: {report_file}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"invalid calibration report JSON: {exc}")
    if not isinstance(data, dict):
        _fail("calibration report JSON must be a CalibrationReport object")
    try:
        return CalibrationReport.from_dict(data)
    except (KeyError, ValueError, ModelError) as exc:
        _fail(f"invalid calibration report: {exc}")


@app.command(name="calibrate-filter")
def calibrate_filter(
    report: str | None = typer.Option(
        None,
        "--report",
        help="Path to a CalibrationReport JSON (e.g. from 'calibrate-irt --out').",
    ),
    matrix: str | None = typer.Option(
        None,
        "--matrix",
        help="Alternatively, a per-model solve matrix JSON to fit + filter in one step.",
    ),
    language: str = typer.Option(
        "python", "--language", help="Task language when building from --matrix."
    ),
    band_high: float = typer.Option(
        DEFAULT_BAND_HIGH,
        "--band-high",
        help="Upper edge of the low-but-nonzero band; frontier pass@k above it is too easy.",
    ),
    discrimination_threshold: float = typer.Option(
        DEFAULT_DISCRIMINATION_THRESHOLD,
        "--discrimination-threshold",
        help="Minimum irt_discrimination required to keep (tiers must separate).",
    ),
    require_keep: bool = typer.Option(
        False,
        "--require-keep",
        help="Exit non-zero when the verdict is 'drop' (e.g. to gate an export step).",
    ),
    out: str | None = typer.Option(
        None, "--out", help="Write the filtered CalibrationReport JSON to this path."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Apply the band filter to a CalibrationReport -> terminal keep/drop verdict.

    Pure offline classification (no Docker, no LLM): KEEP only when the frontier
    pass-rate is in the low-but-nonzero band (``0 < pass@k <= --band-high``) AND
    ``irt_discrimination >= --discrimination-threshold`` AND solves are nonzero;
    otherwise DROP solve-all (too easy), solve-none (broken/impossible),
    out-of-band (too easy), or in-band-low-discrimination. The verdict + an
    attributable reason are written onto the report, and ``keep`` is the necessary
    precondition for export (``drop`` blocks ForgeTask emission).
    """
    if (report is None) == (matrix is None):
        _fail("provide exactly one of --report or --matrix")

    if report is not None:
        calibration = _load_calibration_report(report)
    else:
        if language not in SUPPORTED_LANGUAGES:
            _fail(f"--language must be one of {SUPPORTED_LANGUAGES}; got {language!r}")
        assert matrix is not None
        records, inferred_k = _load_solve_matrix(matrix)
        try:
            calibration = build_calibration_report(language, records, k=inferred_k)
        except IrtError as exc:
            _fail(f"IRT fit failed: {exc}")

    try:
        config = BandFilterConfig(
            band_high=band_high,
            discrimination_threshold=discrimination_threshold,
        )
    except BandFilterError as exc:
        _fail(str(exc))

    apply_band_filter(calibration, config=config)
    decision = calibration.details["band_filter"]
    assert isinstance(decision, dict)

    if out:
        out_path = Path(out)
        if out_path.suffix == ".json":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            report_path = out_path
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            report_path = out_path / "calibration_report.json"
        report_path.write_text(
            json.dumps(calibration.to_dict(), indent=2), encoding="utf-8"
        )

    if json_out:
        typer.echo(json.dumps(calibration.to_dict()))
    else:
        color = "green" if calibration.is_keep else "yellow"
        console.print(
            f"[bold]band filter[/bold] language={calibration.language} "
            f"band=(0, {config.band_high:.4f}] "
            f"disc_threshold={config.discrimination_threshold:.4f}"
        )
        console.print(
            f"  frontier pass@k    : {calibration.frontier_pass_at_k():.4f}\n"
            f"  irt_discrimination : {calibration.irt_discrimination:.4f}\n"
            f"  band_verdict       : [{color}]{calibration.band_verdict}[/{color}] "
            f"({decision['rule']})\n"
            f"  reason             : {calibration.reasons[0] if calibration.reasons else ''}"
        )

    if require_keep and not calibration.is_keep:
        raise typer.Exit(code=1)


def _write_calibration_outcome(out: str, outcome: CalibrationOutcome) -> Path:
    """Write the finalized CalibrationReport JSON to a file or directory path."""
    out_path = Path(out)
    if out_path.suffix == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = out_path
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        report_path = out_path / "calibration_report.json"
    report_path.write_text(
        json.dumps(outcome.report.to_dict(), indent=2), encoding="utf-8"
    )
    return report_path


def _parse_panel_models(spec: str, base_url: str, api_key: str) -> list[PanelModel]:
    """Parse a ``tier:model_string`` panel override into bound PanelModels.

    Each comma-separated entry is ``<tier>:<provider/model_id>`` (the tier is
    everything before the FIRST colon; the model string keeps its ``provider/``
    prefix). Every model inherits the resolved panel endpoint/key.
    """
    panel: list[PanelModel] = []
    for raw in (e.strip() for e in spec.split(",")):
        if not raw:
            continue
        tier, sep, model_string = raw.partition(":")
        tier = tier.strip()
        model_string = model_string.strip()
        if not sep or not model_string:
            _fail(
                f"invalid --models entry {raw!r}; expected 'tier:provider/model_id' "
                f"(e.g. 'frontier:anthropic/claude-opus-4-8')"
            )
        if tier not in VALID_TIERS:
            _fail(f"--models tier must be one of {VALID_TIERS}, got {tier!r}")
        try:
            panel.append(
                PanelModel(
                    id=model_string,
                    model_string=model_string,
                    tier=tier,
                    base_url=base_url,
                    api_key=api_key,
                )
            )
        except Exception as exc:
            _fail(str(exc), api_key)
    if not panel:
        _fail("no panel models parsed from --models")
    return panel


@app.command(name="calibrate")
def calibrate(
    candidate: str = typer.Option(
        ..., "--candidate", help="Path to a valid candidate.json (Stage 2 Candidate)."
    ),
    env: str = typer.Option(
        ..., "--env", help="Path to the EnvImage JSON (Stage 1 green build)."
    ),
    report: str = typer.Option(
        ...,
        "--report",
        help="Path to the oracle-pass OracleReport JSON (its F2P/P2P/test_files set).",
    ),
    spec: str = typer.Option(
        ...,
        "--spec",
        help="Path to the GeneratedSpec JSON (statement+requirements+interface ONLY).",
    ),
    models: str | None = typer.Option(
        None,
        "--models",
        help=(
            "Override the panel: comma-separated 'tier:model_string' entries "
            "(e.g. 'weak:openai/gpt-4o-mini,frontier:anthropic/claude-opus-4-8'). "
            "Default: the env-configured panel (all tiers)."
        ),
    ),
    k: int | None = typer.Option(
        None,
        "--k",
        help="Rollouts per model (default: the difficulty-aware budget for the candidate).",
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", help="Max concurrent in-flight rollouts (semaphore cap)."
    ),
    band_high: float = typer.Option(
        DEFAULT_BAND_HIGH,
        "--band-high",
        help="Upper edge of the low-but-nonzero band; frontier pass@k above it is too easy.",
    ),
    discrimination_threshold: float = typer.Option(
        DEFAULT_DISCRIMINATION_THRESHOLD,
        "--discrimination-threshold",
        help="Minimum irt_discrimination required to keep (tiers must separate).",
    ),
    max_turns: int = typer.Option(
        DEFAULT_MAX_TURNS, "--max-turns", help="Bounded solver tool-loop budget."
    ),
    max_tokens: int = typer.Option(
        SOLVER_DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens per solver call."
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Probe each panel model id once before its k-burst of rollouts.",
    ),
    require_keep: bool = typer.Option(
        False,
        "--require-keep",
        help="Exit non-zero when the band verdict is 'drop' (e.g. to gate an export step).",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        help="Write the finalized CalibrationReport JSON here (file or dir).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Calibrate a candidate end to end: panel run -> finalized CalibrationReport.

    Runs the FULL Stage-4 panel calibration on the candidate's green EnvImage:
    validate every panel model id once (cost discipline), issue k independent,
    uncached, concurrency-bounded rollouts per validated model, score each via the
    SHARED Docker FAIL->PASS path (a solve requires the FULL OracleReport
    test_files[] hidden suite to pass AND P2P/regression green -- never just the
    original F2P), fit the 2-parameter IRT, and apply the band filter. The emitted
    CalibrationReport carries the per-model {model,tier,k,solves,pass_at_k} array,
    irt_difficulty/irt_discrimination, the terminal keep/drop band_verdict +
    reason, full per-call + aggregate usage/cost accounting, and provenance.
    Throwaway containers only (--rm); off-limits containers are never touched.
    """
    from swe_forge.execution.docker_client import DockerError

    candidate_obj = _load_candidate(candidate)
    env_image = _load_env_image(env)
    oracle_report = _load_oracle_report(report)
    spec_obj = _load_spec(spec)
    if spec_obj is None:
        _fail("a valid --spec (GeneratedSpec JSON) is required for calibration")

    registry = build_default_registry()
    try:
        adapter = registry.get(candidate_obj.language)
    except NoAdapterFoundError:
        _fail(
            f"no adapter for language {candidate_obj.language!r}; "
            f"known: {', '.join(registry.names())}"
        )

    base_url, api_key = resolve_panel_endpoint()
    _require_panel_creds(base_url, api_key)

    try:
        config = BandFilterConfig(
            band_high=band_high,
            discrimination_threshold=discrimination_threshold,
        )
    except BandFilterError as exc:
        _fail(str(exc))

    panel = (
        build_panel_from_env()
        if models is None
        else _parse_panel_models(models, base_url, api_key)
    )
    if not panel:
        _fail("panel is empty; configure at least one panel model")

    try:
        outcome = asyncio.run(
            run_calibration(
                candidate_obj,
                env_image,
                spec_obj,
                oracle_report,
                panel,
                k=k,
                concurrency=concurrency,
                validate=validate,
                config=config,
                max_turns=max_turns,
                max_tokens=max_tokens,
                command_timeout=timeout,
                adapter=adapter,
            )
        )
    except BaselineNotGreenError as exc:
        _fail(f"calibration: {exc}", api_key)
    except (SolverError, CalibrationRunnerError, PanelError, DockerError) as exc:
        _fail(f"calibration: {exc}", api_key)

    calibration = outcome.report
    if out:
        _write_calibration_outcome(out, outcome)

    if json_out:
        # The raw patches are never echoed (they can be large); the report carries
        # only counts/usage/cost, never the API key.
        typer.echo(json.dumps({"report": calibration.to_dict()}))
    else:
        color = "green" if calibration.is_keep else "yellow"
        accounting = outcome.run
        console.print(
            f"[bold]calibrate[/bold] language={calibration.language} "
            f"k={calibration.k} band=(0, {config.band_high:.4f}] "
            f"disc_threshold={config.discrimination_threshold:.4f}"
        )
        for rec in calibration.models:
            console.print(
                f"    {rec.tier:8s} {rec.model:32s} "
                f"solves={rec.solves}/{rec.k} pass@k={rec.pass_at_k:.4f}"
            )
        console.print(
            f"  irt_difficulty     : {calibration.irt_difficulty:+.4f}\n"
            f"  irt_discrimination : {calibration.irt_discrimination:.4f}\n"
            f"  frontier pass@k    : {calibration.frontier_pass_at_k():.4f}\n"
            f"  band_verdict       : [{color}]{calibration.band_verdict}[/{color}]\n"
            f"  reason             : "
            f"{calibration.reasons[0] if calibration.reasons else ''}"
        )
        console.print(
            f"  calls              : {accounting.total_calls} "
            f"(validation={accounting.validation_calls}, "
            f"rollout={accounting.rollout_calls})\n"
            f"  usage/cost         : {accounting.usage.total_tokens} tok / "
            f"{accounting.cost}"
        )

    if require_keep and not calibration.is_keep:
        raise typer.Exit(code=1)


def _build_export_request(entry: dict[str, object]) -> ExportRequest:
    """Build an :class:`ExportRequest` from a bundle of artifact file paths."""

    def _need(key: str) -> str:
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            _fail(f"export entry missing required path field {key!r}")
        return value

    candidate_obj = _load_candidate(_need("candidate"))
    env_image = _load_env_image(_need("env"))
    spec_obj = _load_spec(_need("spec"))
    if spec_obj is None:
        _fail("export entry requires a valid GeneratedSpec ('spec')")
    oracle_report = _load_oracle_report(_need("oracle"))
    calibration = _load_calibration_report(_need("calibration"))
    repo_url = entry.get("repo_url")
    base_commit = entry.get("base_commit")
    repo = entry.get("repo")
    task_id = entry.get("task_id")
    return ExportRequest(
        candidate=candidate_obj,
        spec=spec_obj,
        oracle_report=oracle_report,
        calibration_report=calibration,
        env_image=env_image,
        repo_url=str(repo_url) if isinstance(repo_url, str) else "",
        base_commit=str(base_commit) if isinstance(base_commit, str) else "",
        repo=str(repo) if isinstance(repo, str) else None,
        task_id=str(task_id) if isinstance(task_id, str) else None,
    )


@app.command(name="export")
def export_cmd(
    out: str = typer.Option(
        ...,
        "--out",
        help="Output directory (receives tasks/<id>/, dataset.jsonl, dataset.parquet).",
    ),
    manifest: str | None = typer.Option(
        None,
        "--manifest",
        help=(
            "Path to a JSON list of task bundles to export as a batch; each entry "
            "names file paths {candidate, env, spec, oracle, calibration[, repo_url, "
            "base_commit, repo, task_id]}."
        ),
    ),
    candidate: str | None = typer.Option(
        None, "--candidate", help="Single-task: path to the Stage 2 Candidate JSON."
    ),
    env: str | None = typer.Option(
        None, "--env", help="Single-task: path to the Stage 1 EnvImage JSON."
    ),
    spec: str | None = typer.Option(
        None, "--spec", help="Single-task: path to the GeneratedSpec JSON."
    ),
    oracle: str | None = typer.Option(
        None, "--oracle", help="Single-task: path to the OracleReport JSON."
    ),
    calibration: str | None = typer.Option(
        None, "--calibration", help="Single-task: path to the CalibrationReport JSON."
    ),
    repo_url: str = typer.Option(
        "", "--repo-url", help="Single-task: the repo clone URL (for evaluate.sh)."
    ),
    base_commit: str = typer.Option(
        "",
        "--base-commit",
        help="Single-task: base commit (defaults to EnvImage commit).",
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="Single-task: dataset 'owner/repo' slug (derived if unset).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite an existing tasks/<id>/ in place."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Assemble + export the qualified subset of a batch (Stage 5).

    A task is shipped ONLY when its oracle verdict is 'pass' AND its calibration
    band_verdict is 'keep' (fail-fast at assembly; an oracle pass alone is
    necessary but NOT sufficient). Each shipped task gets a self-contained
    ``tasks/<id>/`` workspace (workspace.yaml + gold patch.diff + mutation
    deletion_patch.diff + the FULL hidden ``tests/`` + an executable robust
    ``evaluate.sh``) and exactly one jsonl + one parquet record; the leak audit
    (file scan + .git history) blocks any task that would ship a gold/oracle leak.
    A refused task never aborts its qualified siblings; an empty/zero-kept run
    still writes valid empty datasets.
    """
    if manifest is not None:
        path = Path(manifest)
        if not path.is_file():
            _fail(f"manifest file not found: {manifest}")
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fail(f"invalid manifest JSON: {exc}")
        if not isinstance(entries, list):
            _fail("manifest JSON must be a list of task bundles")
        requests = [
            _build_export_request(entry) for entry in entries if isinstance(entry, dict)
        ]
    else:
        required = {
            "--candidate": candidate,
            "--env": env,
            "--spec": spec,
            "--oracle": oracle,
            "--calibration": calibration,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            _fail(
                "single-task export requires "
                + ", ".join(sorted(missing))
                + " (or use --manifest)"
            )
        requests = [
            _build_export_request(
                {
                    "candidate": candidate,
                    "env": env,
                    "spec": spec,
                    "oracle": oracle,
                    "calibration": calibration,
                    "repo_url": repo_url,
                    "base_commit": base_commit,
                    "repo": repo,
                }
            )
        ]

    result = export_batch(requests, out, overwrite=overwrite)

    if json_out:
        typer.echo(json.dumps(result.to_dict()))
    else:
        console.print(
            f"[bold]export[/bold] out={result.out_dir} "
            f"shipped={len(result.shipped)} kept={len(result.kept)} "
            f"refused={len(result.refused)}"
        )
        for res in result.results:
            color = {
                "shipped": "green",
                "skipped": "cyan",
                "refused": "yellow",
                "failed": "red",
            }.get(res.status, "white")
            line = f"  [{color}]{res.status:8s}[/{color}] {res.task_id}"
            if res.reason:
                line += f"  ({res.reason})"
            console.print(line)
        console.print(
            f"  datasets: {result.jsonl_path.name} / {result.parquet_path.name}"
        )

    # A run that produced no kept task BUT had refusals is a failed export
    # (e.g. a single unqualified task); an empty input or a mixed batch is fine.
    if requests and not result.kept and result.refused:
        raise typer.Exit(code=1)


def _print_gold_eval_report(report: GoldEvalReport) -> None:
    if report.shipped_count == 0:
        console.print(
            f"[red]gold-eval[/red] no tasks/<id>/ workspaces with evaluate.sh "
            f"under {report.tasks_dir}"
        )
        return
    verdict = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
    console.print(
        f"[bold]gold-eval[/bold] {verdict}  "
        f"gold={report.gold_count}/{report.shipped_count} "
        f"({report.gold_rate * 100:.0f}%)  "
        f"deterministic={report.deterministic}"
    )
    for res in report.results:
        if res.gold and res.deterministic:
            color, tag = "green", "gold"
        elif not res.deterministic:
            color, tag = "red", "flip"
        else:
            color, tag = "red", "fail"
        console.print(
            f"  [{color}]{tag:4s}[/{color}] {res.task_id}  "
            f"scores={res.scores}  phase1={res.phase1_all}  image={res.image}"
        )


@app.command(name="gold-eval")
def gold_eval_cmd(
    tasks_dir: str = typer.Option(
        ...,
        "--tasks-dir",
        help=(
            "Directory holding the exported tasks/<id>/ workspaces (or an export "
            "out_dir containing a tasks/ subdir)."
        ),
    ),
    runs: int = typer.Option(
        DEFAULT_DETERMINISM_RUNS,
        "--runs",
        min=DEFAULT_DETERMINISM_RUNS,
        help="Independent --rm container runs per task (>=2 proves determinism).",
    ),
    image: str | None = typer.Option(
        None,
        "--image",
        help=(
            "Override the Docker image for every task (default: each task's own "
            "workspace.yaml environment image)."
        ),
    ),
    timeout: float = typer.Option(
        GOLD_EVAL_TIMEOUT, "--timeout", help="Per-run wall-clock timeout in seconds."
    ),
    docker_arg: list[str] | None = typer.Option(
        None,
        "--docker-arg",
        help="Extra `docker run` argument (repeatable), e.g. an additional -v mount.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """HEADLINE A: run every shipped task's evaluate.sh in Docker and prove gold=100%.

    For each ``tasks/<id>/`` the self-contained ``evaluate.sh`` is run in fresh
    ``--rm`` containers: Phase 1 confirms the broken (mutation) tree FAILS the
    hidden suite while regression stays green, Phase 2 confirms the gold-patched
    tree PASSES the hidden suite AND regression, ending ``{"score": 1}``. The
    command aggregates gold == 100% across the whole shipped set (VAL-EXPORT-010)
    and re-runs each task ``--runs`` times to prove the score never flips
    (VAL-EXPORT-011). Exit 0 iff every shipped task scored gold 1 on every run.
    """
    try:
        report = run_gold_eval(
            tasks_dir,
            runs=runs,
            image=image,
            timeout=timeout,
            extra_args=list(docker_arg or []),
        )
    except GoldEvalError as exc:
        _fail(str(exc))

    if json_out:
        typer.echo(json.dumps(report.to_dict()))
    else:
        _print_gold_eval_report(report)

    ok = report.shipped_count > 0 and report.passed
    raise typer.Exit(code=0 if ok else 1)


def _print_benchmark_report(report: BenchmarkReport) -> None:
    verdict = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
    console.print(
        f"[bold]report[/bold] {verdict}  shipped={report.shipped_count}  "
        f"gold={'%.0f%%' % (report.gold.gold_rate * 100) if report.gold.measured else 'n/a'}  "
        f"frontier={report.frontier_solve_rate:.4f} (< {report.frontier_threshold:.4f}, > 0: "
        f"{report.headline_b_pass})"
    )
    console.print(
        f"  counts: tasks={report.counts.tasks} jsonl={report.counts.jsonl} "
        f"parquet={report.counts.parquet} reconciled={report.counts.reconciled}"
    )
    console.print(
        f"  provenance: complete={report.completeness.complete}/"
        f"{report.completeness.checked} "
        f"consistent={report.consistency.consistent}/{report.consistency.checked}"
    )
    tiers = ", ".join(
        f"{tier}={report.tier_solve_rates[tier]:.4f}"
        for tier in ("weak", "mid", "frontier")
        if tier in report.tier_solve_rates
    )
    console.print(
        f"  tiers: {tiers or 'n/a'} (weak<=mid<=frontier: {report.tier_ordering_ok})"
    )
    console.print(
        f"  breakdown: generators={report.generator_breakdown} "
        f"languages={report.language_breakdown} "
        f"(reconciles: {report.breakdown_reconciles})"
    )


@app.command(name="report")
def report_cmd(
    out_dir: str = typer.Option(
        ...,
        "--out-dir",
        help=(
            "Export out_dir holding tasks/<id>/ plus dataset.jsonl / "
            "dataset.parquet (or the tasks/ dir directly)."
        ),
    ),
    gold_json: str | None = typer.Option(
        None,
        "--gold-json",
        help="Path to a saved `gold-eval --json` report (supplies Headline A).",
    ),
    run_gold_eval_flag: bool = typer.Option(
        False,
        "--run-gold-eval",
        help="Run gold-eval in Docker now to measure Headline A (gold=100%).",
    ),
    image: str | None = typer.Option(
        None, "--image", help="gold-eval image override (with --run-gold-eval)."
    ),
    gold_runs: int = typer.Option(
        DEFAULT_DETERMINISM_RUNS,
        "--gold-runs",
        min=DEFAULT_DETERMINISM_RUNS,
        help="gold-eval --rm runs per task (with --run-gold-eval).",
    ),
    timeout: float = typer.Option(
        GOLD_EVAL_TIMEOUT, "--timeout", help="gold-eval per-run timeout in seconds."
    ),
    docker_arg: list[str] | None = typer.Option(
        None, "--docker-arg", help="Extra `docker run` arg for gold-eval (repeatable)."
    ),
    frontier_threshold: float = typer.Option(
        DEFAULT_FRONTIER_THRESHOLD,
        "--frontier-threshold",
        help="HEADLINE B: the stated frontier solve-rate threshold.",
    ),
    band_high: float = typer.Option(
        DEFAULT_BAND_HIGH,
        "--band-high",
        help="Keep-band upper edge used for the provenance consistency check.",
    ),
    discrimination_threshold: float = typer.Option(
        DEFAULT_DISCRIMINATION_THRESHOLD,
        "--discrimination-threshold",
        help="Minimum IRT discrimination used for the consistency check.",
    ),
    kill_threshold: float = typer.Option(
        DEFAULT_KILL_THRESHOLD,
        "--kill-threshold",
        help="Mutation-adequacy kill ratio required by the completeness check.",
    ),
    write: bool = typer.Option(
        False, "--write", help="Write report.md + report.json into the out_dir."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the JSON report."),
) -> None:
    """Benchmark report + provenance audit over an exported pilot set (Stage 5).

    Audits provenance completeness (VAL-EXPORT-014) and gate-consistency
    (VAL-EXPORT-015) for every shipped task, then rolls up the two headlines:
    gold == 100% from the injected gold-eval result (HEADLINE A / VAL-EXPORT-016),
    and a stated frontier threshold with the measured aggregate frontier
    solve-rate strictly below it yet > 0 (HEADLINE B / VAL-EXPORT-018). Also
    surfaces per-model solve-rates (weak <= mid <= frontier < gold), an IRT
    summary, and a generator/language breakdown summing to the shipped total
    (VAL-EXPORT-017), and reconciles the shipped count with the jsonl/parquet/
    tasks counts (VAL-EXPORT-019). Exit 0 iff every check passes.
    """
    gold: GoldEvalReport | GoldSummary | dict[str, object] | None = None
    if gold_json is not None:
        gpath = Path(gold_json)
        if not gpath.is_file():
            _fail(f"gold-eval JSON not found: {gold_json}")
        try:
            loaded = json.loads(gpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fail(f"invalid gold-eval JSON: {exc}")
        if not isinstance(loaded, dict):
            _fail("gold-eval JSON must be an object (a gold-eval --json report)")
        gold = loaded
    elif run_gold_eval_flag:
        try:
            gold_report = run_gold_eval(
                out_dir,
                runs=gold_runs,
                image=image,
                timeout=timeout,
                extra_args=list(docker_arg or []),
            )
        except GoldEvalError as exc:
            _fail(str(exc))
        gold = gold_report

    try:
        report = build_benchmark_report(
            out_dir,
            gold=gold,
            frontier_threshold=frontier_threshold,
            band_config=BandFilterConfig(
                band_high=band_high,
                discrimination_threshold=discrimination_threshold,
            ),
            kill_threshold=kill_threshold,
        )
    except (ReportError, BandFilterError) as exc:
        _fail(str(exc))

    if write:
        md_path, json_path = write_report(report, out_dir)
        console.print(f"[dim]wrote {md_path.name} + {json_path.name}[/dim]")

    if json_out:
        typer.echo(report.to_json(indent=None))
    else:
        _print_benchmark_report(report)

    raise typer.Exit(code=0 if report.passed else 1)


def _print_pilot_outcome(outcome) -> None:  # type: ignore[no-untyped-def]
    counts = outcome.counts
    verdict = "[green]PASS[/green]" if outcome.ok else "[red]FAIL[/red]"
    console.print(
        f"[bold]build --pilot[/bold] {verdict}  out={outcome.out_dir}  "
        f"shipped={outcome.shipped_count} (in_band[10,30]={outcome.in_band})"
    )
    console.print(
        "  funnel: "
        f"sourced={counts.sourced} >= env={counts.env_built} >= "
        f"synth={counts.synthesized} >= oracle_pass={counts.oracle_pass} >= "
        f"calibrated_keep={counts.calibration_keep} >= "
        f"cap_admitted={counts.cap_admitted} == exported={counts.exported} "
        f"(monotone: {counts.monotone})"
    )
    if outcome.capacity:
        capacity = ", ".join(
            f"{row['repo_id']}: used={row['used']}/{row['cap']} "
            f"remaining={row['remaining']}"
            for row in outcome.capacity
        )
        console.print(f"  source capacity: {capacity}")
    console.print(
        f"  generators: {outcome.generators_used}  "
        f"languages: {outcome.languages_shipped}"
    )
    if outcome.report is not None:
        rep = outcome.report
        console.print(
            f"  headline A (gold=100%): {rep.headline_a_pass}  "
            f"({rep.gold.gold_count}/{rep.gold.shipped_count})"
        )
        console.print(
            f"  headline B (frontier {rep.frontier_solve_rate:.4f} < "
            f"{rep.frontier_threshold:.4f} and > 0): {rep.headline_b_pass}"
        )
    usage = outcome.usage
    console.print(
        f"  usage/cost: teacher={usage.teacher.total_tokens} tok / "
        f"{usage.teacher_cost:.4f}  panel={usage.panel.total_tokens} tok / "
        f"{usage.panel_cost:.4f}  total={usage.total_tokens} tok / "
        f"{usage.total_cost:.4f}"
    )
    for disposition in outcome.dispositions:
        color = {"kept": "green"}.get(disposition.stage, "yellow")
        line = f"  [{color}]{disposition.stage:13s}[/{color}] {disposition.plan.label}"
        if disposition.task_id:
            line += f"  -> {disposition.task_id}"
        if disposition.reason:
            line += f"  ({disposition.reason[:80]})"
        console.print(line)


@app.command(name="build")
def build(
    out: str = typer.Option(
        ...,
        "--out",
        help="Output dir (receives tasks/<id>/, datasets, and the benchmark report).",
    ),
    pilot: bool = typer.Option(
        False,
        "--pilot",
        help="Run the Stage 0->5 pilot preset (many candidates across all 3 languages).",
    ),
    seeds_per_cell: int = typer.Option(
        4,
        "--seeds-per-cell",
        help="Candidates per (repo, generator) cell (more = more in-band candidates).",
    ),
    max_plans: int | None = typer.Option(
        None,
        "--max-plans",
        help="Cap the total candidate count (e.g. a quick smoke run).",
    ),
    languages: str | None = typer.Option(
        None,
        "--languages",
        help="Comma-separated subset of python,javascript,go (default: all three).",
    ),
    k: int | None = typer.Option(
        None,
        "--k",
        help="Rollouts per panel model (default: the difficulty-aware budget).",
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", help="Max concurrent in-flight rollouts (semaphore cap)."
    ),
    candidate_concurrency: int = typer.Option(
        1,
        "--candidate-concurrency",
        help="Max candidates processed concurrently (>1 parallelizes the sweep).",
    ),
    band_high: float = typer.Option(
        DEFAULT_BAND_HIGH,
        "--band-high",
        help="Keep-band upper edge (frontier pass@k above it is dropped as too easy).",
    ),
    discrimination_threshold: float = typer.Option(
        DEFAULT_DISCRIMINATION_THRESHOLD,
        "--discrimination-threshold",
        help="Minimum IRT discrimination required to keep.",
    ),
    frontier_threshold: float = typer.Option(
        DEFAULT_FRONTIER_THRESHOLD,
        "--frontier-threshold",
        help="HEADLINE B: the stated frontier solve-rate threshold.",
    ),
    kill_threshold: float = typer.Option(
        DEFAULT_KILL_THRESHOLD,
        "--kill-threshold",
        help="Mutation-adequacy kill ratio required by the oracle gate.",
    ),
    flakiness_runs: int = typer.Option(
        DEFAULT_FLAKINESS_RUNS,
        "--flakiness-runs",
        help="Determinism repeats in fresh containers (clamped up to 3).",
    ),
    gold_runs: int = typer.Option(
        DEFAULT_DETERMINISM_RUNS,
        "--gold-runs",
        min=DEFAULT_DETERMINISM_RUNS,
        help="gold-eval --rm runs per shipped task.",
    ),
    run_gold: bool = typer.Option(
        True,
        "--gold-eval/--no-gold-eval",
        help="Run gold-eval in Docker after export (Headline A).",
    ),
    validate_models_flag: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Probe each panel model id once before its k-burst of rollouts.",
    ),
    timeout: float = typer.Option(
        600.0, "--timeout", help="Per-command Docker timeout (seconds)."
    ),
    mutation_timeout: float = typer.Option(
        1200.0, "--mutation-timeout", help="Mutation-gate Docker timeout (seconds)."
    ),
    write: bool = typer.Option(
        True, "--write/--no-write", help="Write report.md + report.json into --out."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit the JSON outcome."),
) -> None:
    """Pilot: orchestrate Stage 0->5 end-to-end and ship a benchmark set.

    Runs the entire pipeline in one invocation: source the curated seed repos
    (Stage 0), build one green-baseline image per repo (Stage 1), synthesize many
    candidates across generators/languages/targets (Stage 2), harden each through
    the NON-OFFLINE oracle gates with a real teacher (Stage 3), calibrate the
    oracle-pass candidates against the panel and KEEP only the borderline in-band
    set (Stage 4), then export the qualified subset, prove gold=100% in Docker,
    and write the benchmark report (Stage 5). Only an oracle-pass AND band-keep
    candidate ever becomes a ForgeTask; the per-stage funnel is monotone; teacher
    + panel usage/cost is surfaced; every container is torn down even on failure;
    and no secret reaches any pilot surface. Exit 0 iff the funnel is monotone, a
    non-empty set shipped, gold=100%, the frontier rate is below threshold yet >0,
    and >=2 generators were used.
    """
    base_url, api_key = resolve_panel_endpoint()
    _require_panel_creds(base_url, api_key)

    try:
        band_config = BandFilterConfig(
            band_high=band_high,
            discrimination_threshold=discrimination_threshold,
        )
    except BandFilterError as exc:
        _fail(str(exc))

    language_filter = _split_csv(languages) or None
    if language_filter:
        for lang in language_filter:
            if lang not in DEFAULT_GENERATORS_BY_LANGUAGE:
                _fail(
                    f"--languages entry {lang!r} is not one of "
                    f"{tuple(DEFAULT_GENERATORS_BY_LANGUAGE)}"
                )

    # `--pilot` and the default share the same orchestration; the flag documents
    # intent and (with the default knobs) selects the full Stage 0->5 preset.
    plans = build_pilot_plans(
        seeds_per_cell=seeds_per_cell,
        languages=language_filter,
        max_plans=max_plans,
    )
    if not plans:
        _fail("no candidate plans built; check --languages / --seeds-per-cell")

    config = default_pilot_config(
        out,
        seeds_per_cell=seeds_per_cell,
        languages=language_filter,
        max_plans=max_plans,
        k=k,
        band_config=band_config,
        frontier_threshold=frontier_threshold,
        kill_threshold=kill_threshold,
        flakiness_runs=flakiness_runs,
        concurrency=concurrency,
        candidate_concurrency=candidate_concurrency,
        validate_models=validate_models_flag,
        command_timeout=timeout,
        mutation_timeout=mutation_timeout,
        gold_eval_runs=gold_runs,
        run_gold_eval=run_gold,
        write_report=write,
    )

    try:
        outcome = asyncio.run(run_pilot(config, handle_signals=True))
    except (PilotError, MissingCredentialsError, ModelRoutingError) as exc:
        _fail(f"pilot: {exc}", api_key)

    if json_out:
        typer.echo(json.dumps(outcome.to_dict()))
    else:
        _print_pilot_outcome(outcome)

    raise typer.Exit(code=0 if outcome.ok else 1)
