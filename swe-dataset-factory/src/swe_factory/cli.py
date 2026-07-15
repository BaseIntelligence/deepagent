"""CLI entrypoint for the SWE Dataset Factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from swe_factory import __version__
from swe_factory.config import load_settings

app = typer.Typer(
    name="swe-factory",
    help=(
        "SWE Dataset Factory — produce hard, Docker-verifiable SWE benchmark tasks.\n\n"
        "Pipeline stages (documented commands):\n"
        "  build             — envbuild / produce candidates from sources\n"
        "  envbuild          — pin base_commit, Docker green baseline, digest, dual-build\n"
        "  export            — write agent workspaces + tasks.jsonl after oracle+panel\n"
        "  score             — harness rescoring of candidate patches against a task\n"
        "  offline-fixture   — offline gate-demo artifact (no LLM / OpenRouter)\n"
        "  oracle-gates      — certified G1–G5 mechanical validity (Docker fixture)\n"
        "  synth             — synthetic_grounded multi-file multi-fault producer\n"
        "  pr-mine           — real_pr multi-file merged PR miner\n"
        "  materialize-from-pr — live PR → durable materials tree (inventory + patches)\n"
        "  discover          — DeepSWE real-repo discover (git history + license + multi-file)\n"
        "  real-pr-pool      — merged-PR-only pool; --live (Search/list_pulls) or offline fixture\n"
        "  mine-allowlist    — M8/M9 git-clone-only multi-lang allowlist merge mine\n"
        "  funnel-report     — M9 skip catalog + parallelism caps + funnel docs\n"
        "  oxylabs-probe     — VAL-OXY-005 live universal probe (blocked if no creds)\n"
        "  panel             — hardness panel (band filter / offline matrix)\n"
        "  eval-deepswe      — DeepSWE-grade Pier+mini-swe serial model eval\n"
        "  micro-keep        — live micro path: sources→env→produce→oracle→panel→export\n"
        "  ship-v1           — expand to 20–50 certified keeps + datasets/v1 export\n"
        "  export-harbor     — DeepSWE/Harbor pack tree (task.toml + tests + solution)\n"
        "  export-real-harbor — real_pr product Harbor export (refuse hybrid)\n"
        "  harbor-oracle     — separate-verifier solution=1 / null=0 oracle\n"
        "  deepswe-oracle    — Docker-only DeepSWE cert (sol=1/null=0; refuse fake)\n"
        "  pier-cert         — Pier load + oracle=1 / null=0 (jobs /tmp; --real-pr)\n"
        "  harbor-produce    — multi-lang Harbor motors (Py/Go/TS multi-file hard packs)\n"
        "  ship-harbor       — ship 10–15 DeepSWE Harbor packs to datasets/harbor_v1\n"
        "  ship-deepswe      — product Real-PR ship; --live-mine (not fixture pad) → deepswe_v1\n"
        "  archive-hybrid-deepswe — archive hybrid deepswe_v1 → deepswe_v1_hybrid_archive\n"
        "  archive-seed5-deepswe  — archive prior real_pr seed → deepswe_v1_seed5_archive\n"
        "  gate-audit-product — rewrite dual-truth gate_audit over FULL product tasks/*\n"
        "  ledger            — exact spend ledger summary (cap $600)\n"
        "  config            — show masked OpenRouter / model settings\n\n"
        "Product live mine (M14): real-pr-pool --live → materialize-from-pr →\n"
        "ship-deepswe --source real_pr --live-mine --target 20 --min-packs 15.\n"
        "DeepSWE-grade eval (M15): eval-deepswe --product-root datasets/deepswe_v1 "
        "--max-packs 5 --n-concurrent 1 --hard-stop-usd 300 "
        "(fidelity=pier_miniswe_harbor only; never never-solve panel).\n"
        "fixtures/real_pr_ship and offline-fixture are engineering-only, never product N."
    ),
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """SWE Dataset Factory CLI."""


@app.command("version")
def version_cmd() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command("config")
def config_cmd(
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit masked settings as JSON"),
    ] = False,
) -> None:
    """Show factory settings loaded from the environment (secrets masked)."""
    settings = load_settings()
    summary = settings.masked_summary()
    if json_out:
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    typer.echo("SWE Dataset Factory settings (secrets masked)")
    typer.echo(f"  openrouter_base_url: {summary['openrouter_base_url']}")
    typer.echo(f"  openrouter_api_key:  {summary['openrouter_api_key']}")
    typer.echo(f"  teacher_model:       {summary['teacher_model']}")
    typer.echo(f"  panel_models:        {', '.join(summary['panel_models'])}")  # type: ignore[arg-type]
    typer.echo(f"  budget_usd:          {summary['budget_usd']}")
    typer.echo(f"  github_token:        {summary['github_token']}")


@app.command("build")
def build_cmd(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config only; do not call providers"),
    ] = True,
) -> None:
    """Build candidates (envbuild + produce). Skeleton: offline dry-run only."""
    settings = load_settings()
    if dry_run:
        typer.echo(
            "build: dry-run OK "
            f"(teacher={settings.teacher_model}, "
            f"panel={','.join(settings.panel_models)}, "
            f"budget_usd={settings.budget_usd})"
        )
        raise typer.Exit(code=0)
    typer.secho(
        "build: live pipeline not implemented in skeleton milestone",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("export")
def export_cmd(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config only; do not write datasets"),
    ] = False,
    from_fixture: Annotated[
        bool,
        typer.Option(
            "--from-fixture",
            help="Export fixtures/tiny_offline offline (agent workspace + tasks.jsonl)",
        ),
    ] = False,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Export output directory",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/v1"),
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit export bundle summary as JSON"),
    ] = False,
) -> None:
    """Export certified tasks (agent workspaces + tasks.jsonl + leak scan).

    Agent trees never contain gold (VAL-EXPORT-001). tasks.jsonl keeps full
    records for harness scoring (VAL-EXPORT-002). Leak scan must be clean
    (VAL-EXPORT-003 / VAL-HARNESS-003).
    """
    settings = load_settings()
    if dry_run and not from_fixture:
        typer.echo(f"export: dry-run OK (budget_usd={settings.budget_usd}, out={out})")
        raise typer.Exit(code=0)

    if not from_fixture:
        typer.secho(
            "export: require --from-fixture (live certified keep export arrives later)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    from swe_factory.export.workspace import ExportError, write_export_bundle
    from swe_factory.fixture.offline import build_fixture_task, default_fixture_root

    try:
        task = build_fixture_task()
        # Ensure fixture carries a stub panel so VAL-EXPORT-002 keep fields exist
        if task.panel is None:
            from swe_factory.schema import PanelHardness

            task = task.model_copy(
                update={
                    "panel": PanelHardness(
                        grok_4_5=0.25,
                        opus_4_8=0.0,
                        pass_at_k=0.125,
                        discrimination=1.0,
                    )
                }
            )
        bundle = write_export_bundle(
            tasks=[task],
            out_dir=out,
            broken_repos={task.instance_id: default_fixture_root() / "repo"},
            require_clean_leak_scan=True,
        )
    except (ExportError, Exception) as exc:
        typer.secho(f"export: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "ok": True,
        "out_dir": str(bundle.out_dir),
        "tasks_jsonl": str(bundle.tasks_jsonl),
        "instance_ids": list(bundle.instance_ids),
        "leak_clean": bundle.leak_scan.clean,
        "files_scanned": bundle.leak_scan.files_scanned,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "export: OK "
            f"count={len(bundle.instance_ids)} "
            f"out={bundle.out_dir} "
            f"leak_clean={bundle.leak_scan.clean}"
        )
        typer.echo(f"  tasks_jsonl={bundle.tasks_jsonl}")


@app.command("score")
def score_cmd(
    task_id: Annotated[
        str | None,
        typer.Option("--task-id", help="Task instance_id to score"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate wiring only; no Docker scoring"),
    ] = False,
    from_fixture: Annotated[
        bool,
        typer.Option(
            "--from-fixture",
            help="Score fixtures/tiny_offline gold + null (VAL-HARNESS-001)",
        ),
    ] = False,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Scoring backend: fake (offline) or docker",
        ),
    ] = "docker",
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit GoldNullPair JSON"),
    ] = False,
) -> None:
    """Harness score of a candidate patch against a task.

    Without --from-fixture defaults to dry-run transform for skeleton navigability
    when --dry-run is set. Fixture path proves gold=resolve / null=not without
    manual Docker crafting (VAL-HARNESS-001).
    """
    settings = load_settings()
    label = task_id or "<none>"
    if dry_run and not from_fixture:
        typer.echo(f"score: dry-run OK (task_id={label}, teacher={settings.teacher_model})")
        raise typer.Exit(code=0)

    if not from_fixture:
        typer.secho(
            "score: require --from-fixture or --dry-run (live candidate patch path arrives later)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    from swe_factory.fixture.offline import build_fixture_task, default_fixture_root
    from swe_factory.harness.score import HarnessError, score_gold_and_null

    task = build_fixture_task()
    workspace = default_fixture_root() / "repo"
    backend_kind = backend.strip().lower()

    if backend_kind == "fake":
        from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite

        oracle_runner: object = FakeOracleRunner(
            broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
            gold_runs=[ScriptedSuite(f2p_exits=[0], p2p_exits=[0])],
            null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
        )
    elif backend_kind == "docker":
        from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers
        from swe_factory.oracle.docker_run import OracleDockerRunner

        oracle_runner = OracleDockerRunner(
            docker=DockerCLI(),
            base_image="python:3.12-slim",
            install_commands=["pip install -q pytest"],
            command_timeout=180.0,
        )
    else:
        typer.secho(
            f"score: unknown --backend {backend!r} (use fake|docker)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        pair = score_gold_and_null(
            task=task,
            workspace=workspace,
            runner=oracle_runner,  # type: ignore[arg-type]
        )
        payload = pair.to_dict()
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                f"score: passed={pair.passed} "
                f"gold_resolve={pair.gold.resolve} "
                f"null_resolve={pair.null.resolve} "
                f"instance_id={pair.instance_id}"
            )
        if not pair.passed:
            raise typer.Exit(code=1)
    except HarnessError as exc:
        typer.secho(f"score: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if backend_kind == "docker":
            from swe_factory.envbuild.builder import remove_leftover_sdf_containers

            remove_leftover_sdf_containers()


@app.command("envbuild")
def envbuild_cmd(
    fixture_green: Annotated[
        bool,
        typer.Option(
            "--fixture-green",
            help="Build from fixtures/tiny_green (green baseline fixture)",
        ),
    ] = False,
    fixture_lang: Annotated[
        str | None,
        typer.Option(
            "--fixture-lang",
            help="Multi-lang fixture: python | go | typescript (VAL-ENVR-007)",
        ),
    ] = None,
    dual: Annotated[
        bool,
        typer.Option(
            "--dual",
            help="Run dual-build determinism/usability check (G0 / VAL-ENV-001 / VAL-ENVR-003)",
        ),
    ] = False,
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="Local repo path to envbuild",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    clone_url: Annotated[
        str | None,
        typer.Option("--clone-url", help="Git HTTPS clone URL for real-SHA envbuild"),
    ] = None,
    repo_id: Annotated[
        str,
        typer.Option("--repo-id", help="Repo identity for image tagging"),
    ] = "local",
    base_commit: Annotated[
        str,
        typer.Option("--base-commit", help="Pinned base commit label/SHA (40-char for real)"),
    ] = "local",
    language: Annotated[
        str,
        typer.Option("--language", help="Language recipe: python|go|typescript"),
    ] = "python",
    baseline: Annotated[
        str,
        typer.Option("--baseline", help="Baseline test command"),
    ] = "python -m pytest -q",
    install: Annotated[
        str,
        typer.Option(
            "--install",
            help="Comma-separated install commands",
        ),
    ] = "pip install -q pytest",
    base_image: Annotated[
        str,
        typer.Option("--base-image", help="Docker base image"),
    ] = "python:3.12-slim",
    require_real_sha: Annotated[
        bool,
        typer.Option(
            "--require-real-sha/--allow-synthetic-sha",
            help="Refuse synthetic/placeholder base commits (DeepSWE real path)",
        ),
    ] = False,
    image_namespace: Annotated[
        str,
        typer.Option(
            "--image-namespace",
            help="Owned image namespace (sdf-env | deepswe-env | harbor-sdf)",
        ),
    ] = "sdf-env",
    prune: Annotated[
        bool,
        typer.Option(
            "--prune/--keep-images",
            help="After dual-build, prune owned images (VAL-ENVR-008)",
        ),
    ] = False,
    dump_dockerfile: Annotated[
        bool,
        typer.Option(
            "--dump-dockerfile",
            help="Print agent Dockerfile recipe (offline allow_internet=false) and exit",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit EnvBuildResult JSON"),
    ] = False,
) -> None:
    """Build a reproducible green-baseline Docker env image (envbuild stage).

    Verifies base suite green, records image digest metadata, optionally dual-
    builds for recipe usability (VAL-ENV-001/002, VAL-ENVR-003). Documents
    runtime allow_internet=false, porcelain/hooks, history scrub (VAL-ENVR-002/004/005).
    Always removes owned sdf-*/deepswe-*/harbor-sdf-* containers (VAL-ENVR-006).
    """
    import json as json_lib

    from swe_factory.envbuild.agent_recipe import (
        ALLOW_INTERNET_FALSE,
        render_agent_dockerfile,
    )
    from swe_factory.envbuild.builder import (
        EnvBuilder,
        dual_build,
        prune_owned_env_images,
        remove_leftover_sdf_containers,
    )
    from swe_factory.envbuild.fixture import (
        recipe_for_language,
        recipe_from_clone,
        recipe_from_green_fixture,
    )
    from swe_factory.envbuild.models import EnvRecipe

    if dump_dockerfile:
        force_clone = bool(clone_url)
        df = render_agent_dockerfile(
            base_commit=base_commit if base_commit != "local" else "0" * 40,
            language=language,
            base_image=base_image,
            install_commands=[c.strip() for c in install.split(",") if c.strip()],
            repo_url=clone_url or "",
            # Public clone URLs never get motor COPY (VAL-RCLN-001/002).
            copy_context=clone_url is None,
            force_clone=force_clone,
            source_track="real_pr" if force_clone else "",
        )
        typer.echo(df)
        if ALLOW_INTERNET_FALSE not in df:
            raise typer.Exit(code=1)
        if force_clone and "COPY repo/" in df:
            raise typer.Exit(code=1)
        if force_clone and "git clone" not in df:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    if fixture_green:
        recipe = recipe_from_green_fixture()
    elif fixture_lang:
        recipe = recipe_for_language(fixture_lang)
    elif clone_url:
        recipe = recipe_from_clone(
            repo_id=repo_id if repo_id != "local" else clone_url.rstrip("/").split("/")[-1],
            base_commit=base_commit,
            language=language,
            clone_url=clone_url,
            require_real_sha=require_real_sha or True,
            image_namespace=image_namespace if image_namespace != "sdf-env" else "deepswe-env",
        )
    elif path is not None:
        recipe = EnvRecipe(
            repo_id=repo_id,
            base_commit=base_commit,
            language=language,
            base_image=base_image,
            install_commands=[c.strip() for c in install.split(",") if c.strip()],
            baseline_test_command=baseline,
            local_path=str(path),
            require_real_sha=require_real_sha,
            image_namespace=image_namespace,
            allow_internet=False,
            history_scrub=True,
            hooks_off=True,
        )
    else:
        typer.secho(
            "envbuild: require --fixture-green, --fixture-lang, --path, or --clone-url",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        if dual:
            first, second, verified = dual_build(recipe, prune_on_exit=prune)
            payload = {
                "dual_verified": verified,
                "first": first.to_dict(),
                "second": second.to_dict(),
                "allow_internet": False,
                "runtime_network_policy": ALLOW_INTERNET_FALSE,
            }
            if json_out:
                typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
            else:
                typer.echo(
                    f"envbuild dual: verified={verified} "
                    f"first={first.success} second={second.success} "
                    f"runtime={ALLOW_INTERNET_FALSE}"
                )
                if first.env_image:
                    typer.echo(
                        f"  image_tag={first.env_image.image_tag} "
                        f"digest={first.env_image.image_digest} "
                        f"baseline_green={first.env_image.baseline_green}"
                    )
            if not (first.success and second.success and verified):
                raise typer.Exit(code=1)
            raise typer.Exit(code=0)

        result = EnvBuilder(
            image_namespace=recipe.image_namespace or image_namespace,
        ).build(recipe)
        if json_out:
            typer.echo(json_lib.dumps(result.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"envbuild: success={result.success} stage={result.stage} "
                f"failure_kind={result.failure_kind or '-'} "
                f"runtime={ALLOW_INTERNET_FALSE}"
            )
            if result.env_image:
                typer.echo(
                    f"  image_tag={result.env_image.image_tag}\n"
                    f"  image_digest={result.env_image.image_digest}\n"
                    f"  baseline_green={result.env_image.baseline_green}\n"
                    f"  baseline_cmd={result.env_image.baseline_test_command!r}\n"
                    f"  allow_internet={result.env_image.allow_internet}\n"
                    f"  resolved_head={result.env_image.resolved_head or '-'}\n"
                    f"  history_scrubbed={result.env_image.history_scrubbed}\n"
                    f"  isolation_clean={result.env_image.isolation_clean}"
                )
            elif result.reason:
                typer.echo(f"  reason={result.reason}")
        if not result.success:
            raise typer.Exit(code=1)
        if prune and result.env_image is not None:
            pruned = prune_owned_env_images(image_refs=[result.env_image.image_tag])
            if not json_out:
                typer.echo(f"  pruned_images={pruned}")
    finally:
        remove_leftover_sdf_containers()


@app.command("oracle-gates")
def oracle_gates_cmd(
    fixture_offline: Annotated[
        bool,
        typer.Option(
            "--fixture-offline",
            help="Run certified G1–G5 on fixtures/tiny_offline (Docker)",
        ),
    ] = False,
    dual_runs: Annotated[
        int,
        typer.Option("--dual-runs", help="Number of gold dual-run evaluations (default 2)"),
    ] = 2,
    audit_out: Annotated[
        Path | None,
        typer.Option(
            "--audit-out",
            help="Append gate_audit.jsonl path",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit GateResult JSON"),
    ] = False,
) -> None:
    """Run mechanical oracle gates G1–G5 (certified path).

    Proves F2P fail-on-broken, gold dual-run F2P+P2P, null-patch non-resolve,
    multi-file floor, flake reject, and optional gate_audit.jsonl reason codes
    (VAL-ORACLE-001..006). Always tears down sdf-* containers.
    """
    import json as json_lib

    if not fixture_offline:
        typer.secho(
            "oracle-gates: require --fixture-offline (live candidate path arrives later)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers
    from swe_factory.envbuild.fixture import default_offline_fixture_root
    from swe_factory.oracle.docker_run import OracleDockerRunner
    from swe_factory.oracle.gates import append_gate_audit, run_certified_gates

    root = default_offline_fixture_root()
    gold = (root / "gold.patch").read_text(encoding="utf-8")
    meta = json_lib.loads((root / "task_meta.json").read_text(encoding="utf-8"))
    f2p = list(meta["fail_to_pass"])
    p2p = list(meta.get("pass_to_pass") or [])
    digest = str((meta.get("environment") or {}).get("image_digest") or "sha256:oracle_cli")
    problem = str(meta.get("problem_statement") or "fixture offline")

    runner = OracleDockerRunner(
        docker=DockerCLI(),
        base_image="python:3.12-slim",
        install_commands=["pip install -q pytest"],
        command_timeout=180.0,
    )
    try:
        result = run_certified_gates(
            gold_patch=gold,
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            problem_statement=problem,
            image_digest=digest,
            workspace=root / "repo",
            runner=runner,
            agent_workspace=None,
            require_multi_file=True,
            dual_runs=max(1, dual_runs),
            check_null_patch=True,
            check_leak=True,
        )
        if audit_out is not None:
            append_gate_audit(
                audit_out,
                result,
                str(meta.get("instance_id") or "fixture__tiny_offline__oracle"),
            )
        if json_out:
            typer.echo(json_lib.dumps(result.to_gate_proof(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"oracle-gates: passed={result.passed} mode={result.mode} "
                f"codes={','.join(result.reason_codes)}"
            )
            typer.echo(f"  multi_file={result.multi_file} files_touched={result.files_touched}")
        if not result.passed:
            raise typer.Exit(code=1)
    finally:
        remove_leftover_sdf_containers()


@app.command("synth")
def synth_cmd(
    fixture_green: Annotated[
        bool,
        typer.Option(
            "--fixture-green",
            help="Mutate fixtures/tiny_green offline (synthetic_grounded)",
        ),
    ] = False,
    mutation: Annotated[
        str,
        typer.Option(
            "--mutation",
            help="Mutation kind: multi_fault | function_removal",
        ),
    ] = "multi_fault",
    seed_id: Annotated[
        str,
        typer.Option("--seed-id", help="Allowlist seed_id (default fixture_tiny_green)"),
    ] = "fixture_tiny_green",
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Work/output directory for broken workspace + gold",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/synth_demo"),
    certify_fake: Annotated[
        bool,
        typer.Option(
            "--certify-fake",
            help="Run certified oracle via FakeOracleRunner (offline, no Docker)",
        ),
    ] = False,
    certify_docker: Annotated[
        bool,
        typer.Option(
            "--certify-docker",
            help="Run certified oracle via Docker smoke (sdf-*; no panel)",
        ),
    ] = False,
    audit_out: Annotated[
        Path | None,
        typer.Option(
            "--audit-out",
            help="Append gate_audit.jsonl for certified path",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit candidate summary JSON"),
    ] = False,
) -> None:
    """Produce synthetic_grounded multi-file multi-fault / function-removal tasks.

    Mutates a green base (prefer allowlisted modular fixture), emits inverse gold,
    labels source_track=synthetic_grounded, and optionally validates under oracle
    (VAL-PROD-002/003). Offline fakes by default; optional Docker certify smoke.
    Never calls OpenRouter/panel.
    """
    import json as json_lib

    from swe_factory.producers.synth import (
        MUTATION_FUNCTION_REMOVAL,
        MUTATION_MULTI_FAULT,
        SynthError,
        SynthProducer,
    )
    from swe_factory.sources.allowlist import get_seed

    if not fixture_green and seed_id == "fixture_tiny_green":
        # default path is fixture-green offline
        fixture_green = True

    kind = mutation.strip().lower().replace("-", "_")
    if kind in {"multi_fault", "multifault", "multi"}:
        mutation_kind = MUTATION_MULTI_FAULT
    elif kind in {"function_removal", "function-removal", "removal", "remove"}:
        mutation_kind = MUTATION_FUNCTION_REMOVAL
    else:
        typer.secho(
            f"synth: unknown --mutation {mutation!r} (multi_fault|function_removal)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        seed = get_seed(seed_id)
    except KeyError as exc:
        typer.secho(f"synth: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if not seed.resolve_local_path() and not fixture_green:
        typer.secho(
            f"synth: seed {seed_id!r} has no local green tree; use --fixture-green "
            "or clone the remote seed first",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    producer = SynthProducer(work_root=out)
    try:
        if certify_docker:
            from swe_factory.envbuild.builder import DockerCLI, remove_leftover_sdf_containers
            from swe_factory.oracle.docker_run import OracleDockerRunner

            docker_runner = OracleDockerRunner(
                docker=DockerCLI(),
                base_image=seed.base_image or "python:3.12-slim",
                install_commands=list(seed.install_commands) or ["pip install -q pytest"],
                command_timeout=180.0,
            )
            try:
                candidate = producer.produce_and_certify(
                    seed,
                    runner=docker_runner,
                    mutation_kind=mutation_kind,
                    dual_runs=2,
                    audit_out=audit_out,
                )
            finally:
                remove_leftover_sdf_containers()
        elif certify_fake:
            from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite

            fake_runner = FakeOracleRunner(
                broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                gold_runs=[
                    ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                    ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                ],
                null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
            )
            candidate = producer.produce_and_certify(
                seed,
                runner=fake_runner,
                mutation_kind=mutation_kind,
                dual_runs=2,
                audit_out=audit_out,
            )
        else:
            candidate = producer.produce(
                seed,
                mutation_kind=mutation_kind,
                run_stub_oracle=True,
            )
    except SynthError as exc:
        typer.secho(f"synth: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    task = candidate.task
    track = (
        task.source_track.value if hasattr(task.source_track, "value") else str(task.source_track)
    )
    payload = {
        "ok": True,
        "instance_id": task.instance_id,
        "source_track": track,
        "mutation_kind": candidate.mutation_kind,
        "repo": task.repo,
        "base_commit": task.base_commit,
        "language": task.language,
        "problem_statement_nonempty": bool(task.problem_statement.strip()),
        "gold_files": list(candidate.gold_files),
        "multi_file": len(candidate.gold_files) >= 2,
        "broken_workspace": str(candidate.broken_workspace),
        "provider_calls": candidate.provider_calls,
        "inverse_meta": candidate.inverse_meta,
        "gates_passed": None if candidate.gates is None else candidate.gates.passed,
        "reason_codes": ([] if candidate.gates is None else list(candidate.gates.reason_codes)),
    }
    # Materialize tasks.jsonl sample for VAL-PROD inspection
    out.mkdir(parents=True, exist_ok=True)
    tasks_jsonl = out / "tasks.jsonl"
    tasks_jsonl.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    payload["tasks_jsonl"] = str(tasks_jsonl)

    if json_out:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "synth: OK "
            f"source_track={track} "
            f"mutation={candidate.mutation_kind} "
            f"files={len(candidate.gold_files)} "
            f"instance_id={task.instance_id}"
        )
        typer.echo(f"  broken={candidate.broken_workspace}")
        typer.echo(f"  tasks_jsonl={tasks_jsonl}")
        if candidate.gates is not None:
            typer.echo(
                f"  gates_passed={candidate.gates.passed} "
                f"codes={','.join(candidate.gates.reason_codes)}"
            )


@app.command("pr-mine")
def pr_mine_cmd(
    offline_fixture: Annotated[
        bool,
        typer.Option(
            "--offline-fixture",
            help="Use built-in multi-file PR fixture (no GitHub network)",
        ),
    ] = False,
    repo: Annotated[
        str | None,
        typer.Option("--repo", help="GitHub repo owner/name (live or mocked API)"),
    ] = None,
    pr_number: Annotated[
        int | None,
        typer.Option("--pr", help="Merged PR number to mine"),
    ] = None,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Work/output directory for gold + tasks.jsonl",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/real_pr_demo"),
    certify_fake: Annotated[
        bool,
        typer.Option(
            "--certify-fake",
            help="Run certified oracle via FakeOracleRunner (offline, no Docker)",
        ),
    ] = False,
    audit_out: Annotated[
        Path | None,
        typer.Option(
            "--audit-out",
            help="Append gate_audit.jsonl for certified path",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    base_workspace: Annotated[
        Path | None,
        typer.Option(
            "--base-workspace",
            help="Optional local tree at base_commit (required for --certify-fake)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit candidate summary JSON"),
    ] = False,
) -> None:
    """Mine multi-file merged PRs into labeled real_pr tasks (VAL-PROD-001/004).

    Offline fixture is the default for CI. Live path resolves one merged PR via
    GitHub REST (GITHUB_TOKEN/GH_TOKEN used when present; never logged). Gold is
    extracted from multi-file source diffs; F2P prefers PR test paths.
    """
    import json as json_lib

    from swe_factory.producers.pr_miner import (
        PrMineError,
        PrMiner,
        offline_fixture_pr,
        produce_offline_fixture,
    )
    from swe_factory.sources.github import GitHubClient, GitHubError

    settings = load_settings()

    # Default offline when no live selectors given
    if not offline_fixture and repo is None and pr_number is None:
        offline_fixture = True

    try:
        if offline_fixture:
            if certify_fake:
                from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite

                fixture_repo = (
                    Path(__file__).resolve().parents[2] / "fixtures" / "tiny_offline" / "repo"
                )
                ws = base_workspace or fixture_repo
                if not Path(ws).is_dir():
                    typer.secho(
                        f"pr-mine: base workspace missing: {ws}",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(code=2)
                token = settings.github_token.get_secret_value() if settings.github_token else None
                # Client unused for offline GR; still construct for API symmetry
                client = GitHubClient.from_env(token=token)
                miner = PrMiner(client=client, work_root=out)
                pr = offline_fixture_pr()
                fake_runner = FakeOracleRunner(
                    broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                    gold_runs=[
                        ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                        ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                    ],
                    null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                )
                candidate = miner.produce_and_certify(
                    pr,
                    runner=fake_runner,
                    workspace=Path(ws),
                    dual_runs=2,
                    audit_out=audit_out,
                    fail_to_pass=["python -m pytest tests/test_math.py tests/test_text.py -q"],
                    pass_to_pass=["python -m pytest tests/test_ok.py -q"],
                    instance_suffix="offline",
                )
            else:
                candidate = produce_offline_fixture(
                    work_root=out,
                    run_stub_oracle=True,
                    base_workspace=base_workspace,
                )
        else:
            if not repo or not pr_number:
                typer.secho(
                    "pr-mine: live mode requires --repo owner/name and --pr N "
                    "(or use --offline-fixture)",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            token = settings.github_token.get_secret_value() if settings.github_token else None
            client = GitHubClient.from_env(token=token)
            miner = PrMiner(client=client, work_root=out)
            if certify_fake:
                if base_workspace is None or not Path(base_workspace).is_dir():
                    typer.secho(
                        "pr-mine: --certify-fake live mode requires --base-workspace",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(code=2)
                from swe_factory.oracle.docker_run import FakeOracleRunner, ScriptedSuite

                pr = miner.fetch_merged_pr(repo, int(pr_number))
                fake_runner = FakeOracleRunner(
                    broken=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                    gold_runs=[
                        ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                        ScriptedSuite(f2p_exits=[0], p2p_exits=[0]),
                    ],
                    null=ScriptedSuite(f2p_exits=[1], p2p_exits=[0]),
                )
                candidate = miner.produce_and_certify(
                    pr,
                    runner=fake_runner,
                    workspace=Path(base_workspace),
                    dual_runs=2,
                    audit_out=audit_out,
                )
            else:
                candidate = miner.produce_from_pr_number(
                    repo,
                    int(pr_number),
                    run_stub_oracle=True,
                    base_workspace=base_workspace,
                )
    except (PrMineError, GitHubError) as exc:
        typer.secho(f"pr-mine: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    task = candidate.task
    track = (
        task.source_track.value if hasattr(task.source_track, "value") else str(task.source_track)
    )
    payload = {
        "ok": True,
        "instance_id": task.instance_id,
        "source_track": track,
        "repo": task.repo,
        "repository_url": candidate.repository_url or candidate.provenance.get("repository_url"),
        "base_commit": task.base_commit,
        "language": task.language,
        "license": task.license,
        "pr_number": candidate.pr.number,
        "problem_statement_nonempty": bool(task.problem_statement.strip()),
        "gold_files": list(candidate.gold_files),
        "multi_file": len(candidate.gold_files) >= 2,
        "test_files": list(candidate.pr.test_files),
        "source_files": list(candidate.pr.source_files),
        "test_patch_nonempty": bool((candidate.test_patch or "").strip()),
        "workspace": str(candidate.workspace) if candidate.workspace else None,
        "provider_calls": candidate.provider_calls,
        "provenance": candidate.provenance,
        "gates_passed": None if candidate.gates is None else candidate.gates.passed,
        "reason_codes": ([] if candidate.gates is None else list(candidate.gates.reason_codes)),
        "github_token_present": bool(settings.github_token),
    }
    out.mkdir(parents=True, exist_ok=True)
    tasks_jsonl = out / "tasks.jsonl"
    tasks_jsonl.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    payload["tasks_jsonl"] = str(tasks_jsonl)
    # Staging patches for Harbor downstream (held-out tests + source gold)
    staging = out / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "gold.patch").write_text(task.gold_patch, encoding="utf-8")
    (staging / "solution.patch").write_text(task.gold_patch, encoding="utf-8")
    (staging / "test.patch").write_text(candidate.test_patch or "", encoding="utf-8")
    payload["staging_dir"] = str(staging)

    if json_out:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "pr-mine: OK "
            f"source_track={track} "
            f"repo={task.repo} "
            f"pr=#{candidate.pr.number} "
            f"files={len(candidate.gold_files)} "
            f"instance_id={task.instance_id}"
        )
        typer.echo(f"  base_commit={task.base_commit}")
        typer.echo(f"  repository_url={payload.get('repository_url')}")
        typer.echo(f"  test_patch_nonempty={payload['test_patch_nonempty']}")
        typer.echo(f"  tasks_jsonl={tasks_jsonl}")
        if candidate.gates is not None:
            typer.echo(
                f"  gates_passed={candidate.gates.passed} "
                f"codes={','.join(candidate.gates.reason_codes)}"
            )


@app.command("materialize-from-pr")
def materialize_from_pr_cmd(
    repo: Annotated[
        str | None,
        typer.Option("--repo", help="GitHub owner/name for a single PR materialize"),
    ] = None,
    pr_number: Annotated[
        int | None,
        typer.Option("--pr", help="Merged PR number to materialize"),
    ] = None,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Non-fixture materials root (default datasets/live_materials)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/live_materials"),
    candidates: Annotated[
        Path | None,
        typer.Option(
            "--candidates",
            help="candidates.jsonl (accepted rows) to materialize in batch",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    product_mode: Annotated[
        bool,
        typer.Option(
            "--product-mode",
            help="Apply product hard filters when fetching PRs (≥10 hunks, license, …)",
        ),
    ] = False,
    discovery_path: Annotated[
        str | None,
        typer.Option(
            "--discovery-path",
            help="Label on inventory row: search | list_pulls (optional)",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", help="Language override for inventory (optional)"),
    ] = None,
    license_name: Annotated[
        str | None,
        typer.Option("--license", help="SPDX license override (optional)"),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit materialize report JSON"),
    ] = False,
) -> None:
    """Bridge accepted live PRs → durable materials tree (VAL-LMAT-001/004).

    Writes ``{out}/inventory.json`` plus ``{task_id}/solution.patch`` and
    ``test.patch``. Materials root must **not** be ``fixtures/real_pr_ship``
    (that shortlist is offline unit only); default is ``datasets/live_materials``
    so product ship can load a live bridge without fixture pad (VAL-LMINE-008).
    """
    import json as json_lib

    from swe_factory.producers.materialize_from_pr import (
        DEFAULT_LIVE_MATERIALS_ROOT,
        MaterializeError,
        is_fixture_materials_root,
        materialize_accepted_candidates,
        materialize_from_pr_number,
        read_inventory,
    )
    from swe_factory.sources.github import GitHubClient, GitHubError, resolve_github_token

    materials_root = Path(out) if out is not None else DEFAULT_LIVE_MATERIALS_ROOT
    if is_fixture_materials_root(materials_root):
        typer.secho(
            "materialize-from-pr: refuse fixtures/real_pr_ship as live materials root "
            f"(got {materials_root}); use datasets/live_materials or another non-fixture path "
            "(VAL-LMAT-001 / VAL-LMINE-008)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    token = resolve_github_token()
    # Live single-PR / candidates.jsonl need REST; offline unit testing uses the library API.
    if not token and (repo or candidates is not None):
        typer.secho(
            "materialize-from-pr: live path requires a resolvable GitHub token "
            "(GITHUB_TOKEN | GH_TOKEN | gh auth token); unit offline uses library mocks",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        client = GitHubClient.from_env(token=token)
        if candidates is not None:
            if not candidates.is_file():
                typer.secho(
                    f"materialize-from-pr: candidates file missing: {candidates}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            rows: list[dict[str, Any]] = []
            for line in candidates.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json_lib.loads(line)
                if not isinstance(row, dict):
                    continue
                # Prefer accepted disposition; allow unlabeled rows for smoke.
                disp = str(row.get("disposition") or "accept").lower()
                if disp in {"reject", "rejected", "skip", "skipped"}:
                    continue
                rows.append(row)
            if not rows:
                typer.secho(
                    "materialize-from-pr: no accepted rows in candidates ledger",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)
            report = materialize_accepted_candidates(
                client,
                rows,
                materials_root,
                product_mode=product_mode,
            )
            payload = report.to_dict()
        elif repo and pr_number:
            task = materialize_from_pr_number(
                client,
                repo,
                int(pr_number),
                materials_root,
                language=language,
                license=license_name,
                discovery_path=discovery_path,
                product_mode=product_mode,
            )
            payload = {
                "ok": True,
                "materials_root": str(materials_root),
                "inventory_path": str(Path(materials_root) / "inventory.json"),
                "count": 1,
                "task_ids": [task.task_id],
                "product_materials": True,
                "engineering_fixture": False,
                "task": task.to_dict(),
                "inventory": read_inventory(materials_root),
            }
        else:
            typer.secho(
                "materialize-from-pr: provide --repo + --pr, or --candidates candidates.jsonl",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
    except (MaterializeError, GitHubError) as exc:
        typer.secho(f"materialize-from-pr: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "materialize-from-pr: OK "
            f"count={payload.get('count')} "
            f"root={payload.get('materials_root')} "
            f"tasks={','.join(payload.get('task_ids') or [])}"
        )
        typer.echo(f"  inventory={payload.get('inventory_path')}")
        typer.echo(f"  product_materials={payload.get('product_materials')}")


@app.command("discover")
def discover_cmd(
    offline_fixture: Annotated[
        bool,
        typer.Option(
            "--offline-fixture",
            help="Offline multi-file fixture discover (no network; VAL-MINE-004)",
        ),
    ] = False,
    git_repo: Annotated[
        Path | None,
        typer.Option(
            "--git-repo",
            help="Local git checkout (history authority for base..head diffs)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    base: Annotated[
        str | None,
        typer.Option("--base", help="Base commit (full 40-char SHA)"),
    ] = None,
    head: Annotated[
        str | None,
        typer.Option("--head", help="Head commit / merge tip (resolved via git)"),
    ] = None,
    repo: Annotated[
        str | None,
        typer.Option("--repo", help="owner/name identity for the candidate"),
    ] = None,
    repository_url: Annotated[
        str | None,
        typer.Option(
            "--repository-url",
            help="Public HTTPS repository URL (required for curated/hybrid accepts)",
        ),
    ] = None,
    license_id: Annotated[
        str,
        typer.Option("--license", help="SPDX / free-text license (copyleft fails closed)"),
    ] = "MIT",
    language: Annotated[
        str | None,
        typer.Option("--language", help="python|go|typescript|javascript"),
    ] = None,
    kind: Annotated[
        str,
        typer.Option("--kind", help="real_pr|curated (curated still needs real URL+SHA)"),
    ] = "real_pr",
    title: Annotated[
        str,
        typer.Option("--title", help="Optional candidate title / PR title"),
    ] = "",
    pr_number: Annotated[
        int | None,
        typer.Option("--pr", help="Optional PR number for provenance"),
    ] = None,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Discover output directory (candidates + report.json)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/discover_demo"),
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit DiscoverReport JSON"),
    ] = False,
) -> None:
    """Discover DeepSWE multi-file candidates (git history authority).

    Offline fixture proves wiring with provider_calls=0. Live git mode clones
    are not required here when ``--git-repo`` already holds objects; diffs come
    from local git only (never Oxylabs for history). License gate rejects
    copyleft. Gold is source-only; tests land in held-out test.patch.
    """
    import json as json_lib

    from swe_factory.sources.discover import (
        DiscoverError,
        DiscoverReport,
        discover_merge_range_from_git,
        discover_offline_fixture,
        write_candidate_artifacts,
    )
    from swe_factory.sources.license_gate import LicenseGateError

    # Default offline for organic CI when no selectors provided
    if not offline_fixture and git_repo is None and repo is None:
        offline_fixture = True

    try:
        if offline_fixture:
            report = discover_offline_fixture(work_root=out)
        else:
            if git_repo is None or not Path(git_repo).is_dir():
                typer.secho(
                    "discover: live git mode requires --git-repo pointing at a local clone",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            if not base or not head:
                typer.secho(
                    "discover: live git mode requires --base and --head",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            if not repo:
                typer.secho(
                    "discover: --repo owner/name is required for live discover",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            kind_n = (kind or "real_pr").strip().lower()
            if kind_n not in {"real_pr", "curated"}:
                typer.secho(
                    f"discover: invalid --kind {kind!r}; use real_pr or curated",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            candidate = discover_merge_range_from_git(
                Path(git_repo),
                base=base,
                head=head,
                repo=repo,
                repository_url=repository_url,
                license=license_id,
                language=language,
                title=title,
                pr_number=pr_number,
                kind=kind_n,  # type: ignore[arg-type]
            )
            out.mkdir(parents=True, exist_ok=True)
            write_candidate_artifacts(candidate, out)
            report = DiscoverReport(
                kept=(candidate,),
                rejected=(),
                provider_calls=0,
                network_required=False,
                offline=False,
                history_authority="git",
            )
    except (DiscoverError, LicenseGateError) as exc:
        typer.secho(f"discover: rejected: {exc}", fg=typer.colors.RED, err=True)
        # Funnel-friendly reject report for scripts
        reject_payload = {
            "ok": False,
            "error": str(exc),
            "reason_code": getattr(exc, "reason_code", None) or "discover_rejected",
            "provider_calls": 0,
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(
            json_lib.dumps(reject_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if json_out:
            typer.echo(json_lib.dumps(reject_payload, indent=2, sort_keys=True))
        raise typer.Exit(code=1) from exc

    out.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    (out / "report.json").write_text(
        json_lib.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Also write candidates.jsonl for funnel inspection
    lines = [json_lib.dumps(c.to_dict(), sort_keys=True) for c in report.kept]
    (out / "candidates.jsonl").write_text(
        ("\n".join(lines) + ("\n" if lines else "")),
        encoding="utf-8",
    )
    payload["report_json"] = str(out / "report.json")
    payload["candidates_jsonl"] = str(out / "candidates.jsonl")

    if json_out:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "discover: OK "
            f"kept={report.keep_count} rejected={report.reject_count} "
            f"offline={report.offline} provider_calls={report.provider_calls} "
            f"history={report.history_authority}"
        )
        for c in report.kept:
            typer.echo(
                f"  - {c.candidate_id} repo={c.repo} "
                f"base={c.base_commit[:12]}… files={len(c.gold_files)} "
                f"license={c.license} url={c.repository_url}"
            )
        typer.echo(f"  report={out / 'report.json'}")


@app.command("real-pr-pool")
def real_pr_pool_cmd(
    offline_fixture: Annotated[
        bool,
        typer.Option(
            "--offline-fixture/--live",
            help=(
                "Offline multi-repo real_pr fixture pool (engineering-only, not product N) "
                "or live GitHub REST/Search discovery (requires network+token)"
            ),
        ),
    ] = True,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output directory for real_pr_pool_report.json + candidates.jsonl",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/real_pr_pool"),
    target: Annotated[
        int,
        typer.Option(
            "--target",
            help="Candidate keep target (inventory for later select ≥5 packs)",
            min=1,
        ),
    ] = 5,
    language: Annotated[
        list[str] | None,
        typer.Option(
            "--language",
            help="Restrict live pool seeds (repeatable): python/go/typescript/…",
        ),
    ] = None,
    seed_id: Annotated[
        list[str] | None,
        typer.Option("--seed-id", help="Restrict live pool to seed_id (repeatable)"),
    ] = None,
    max_scan: Annotated[
        int,
        typer.Option(
            "--max-scan",
            help=(
                "Max closed PRs scanned per seed (live). Increase toward 80+ so "
                "after hard floors enough candidates remain for dual-run+HarborDocker."
            ),
            min=1,
        ),
    ] = 80,
    discovery_path: Annotated[
        list[str] | None,
        typer.Option(
            "--discovery-path",
            help="Live discovery path (repeatable): search and/or list_pulls (default both)",
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit RealPrPoolReport JSON"),
    ] = False,
) -> None:
    """Mine a MERGED-PR-only real_pr candidate pool (VAL-RPR-002..005, VAL-LMINE-003/007).

    Offline path emits synthetic multi-repo diversity from the multi-file PR
    fixture (no network) and is **engineering-only** — never product N.

    Live path requires network + GITHUB_TOKEN/GH_TOKEN/`gh auth token`, scans
    public allowlisted remotes via list_pulls and/or Search, and writes durable
    ``candidates.jsonl`` rows with ``discovery_path=search|list_pulls``.
    Harbor motors / hybrid_curated are rejected and never enter the pool.
    """
    import json as json_lib

    from swe_factory.sources.github import DISCOVERY_PATHS, GitHubClient, resolve_github_token
    from swe_factory.sources.real_pr_pool import (
        RealPrPoolError,
        mine_live_merged_pr_pool,
        mine_offline_real_pr_pool,
        summary_seed_pool_for_select5,
    )

    settings = load_settings()
    settings_token = settings.github_token.get_secret_value() if settings.github_token else None
    try:
        if offline_fixture:
            report = mine_offline_real_pr_pool(
                work_root=out,
                target_candidates=target,
            )
        else:
            token = resolve_github_token(settings_token)
            if not token:
                raise RealPrPoolError(
                    "live real-pr-pool requires network+token "
                    "(set GITHUB_TOKEN / GH_TOKEN or run `gh auth token`); "
                    "refusing unauth mass mine as product N evidence"
                )
            client = GitHubClient.from_env(token=token)
            paths = list(discovery_path) if discovery_path else None
            if paths is not None:
                for p in paths:
                    if p not in DISCOVERY_PATHS:
                        raise RealPrPoolError(
                            f"invalid --discovery-path {p!r}; "
                            f"must be one of {sorted(DISCOVERY_PATHS)}"
                        )
            report = mine_live_merged_pr_pool(
                client,
                work_root=out,
                seed_ids=list(seed_id) if seed_id else None,
                languages=list(language) if language else None,
                target_candidates=target,
                max_scan_per_repo=max_scan,
                max_keeps_per_repo=max(8, min(20, target // 2 + 2)),
                max_seeds=36,
                discovery_paths=paths,
                product_mode=True,
                require_token=True,
                token=token,
            )
    except RealPrPoolError as exc:
        typer.secho(f"real-pr-pool: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = report.to_dict()
    payload["select5_inventory"] = summary_seed_pool_for_select5()
    payload["github_token_present"] = bool(resolve_github_token(settings_token))
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_pr_pool_report.json").write_text(
        json_lib.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if json_out:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "real-pr-pool: OK "
            f"mode={report.mode} kept={report.keep_count} "
            f"repos={report.repo_diversity} "
            f"motors_rejected={len(report.motor_rejects)} "
            f"source_track={report.source_track} "
            f"product_n_evidence={report.product_n_evidence} "
            f"engineering_only={report.engineering_only}"
        )
        for row in payload.get("kept", [])[:12]:
            typer.echo(
                f"  - {row.get('repo')} pr={row.get('pr_number')} "
                f"files={len(row.get('gold_files') or [])} "
                f"path={row.get('discovery_path')} "
                f"hunks={row.get('source_hunk_count')} "
                f"track={row.get('source_track', report.source_track)}"
            )
        typer.echo(f"  report={out / 'real_pr_pool_report.json'}")
        typer.echo(f"  candidates={out / 'candidates.jsonl'}")
        inv = payload["select5_inventory"]
        typer.echo(
            f"  select5_inventory seeds={inv.get('seed_count')} "
            f"meets_floor={inv.get('meets_select5_inventory')}"
        )
        if report.engineering_only:
            typer.echo("  note: offline_fixture is engineering-only (not product N / live mine)")


@app.command("mine-allowlist")
def mine_allowlist_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output directory for candidates + git_mine_report.json",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/mine_allowlist_m8"),
    target: Annotated[
        int,
        typer.Option(
            "--target",
            help="Candidate keep target (feed m8-ship ≥30 docker-oracle packs)",
            min=1,
        ),
    ] = 30,
    max_merges: Annotated[
        int,
        typer.Option("--max-merges", help="Max merges/commits scanned per seed", min=1),
    ] = 40,
    max_keeps_per_seed: Annotated[
        int,
        typer.Option("--max-keeps-per-seed", help="Cap keeps emitted per remote seed", min=1),
    ] = 6,
    language: Annotated[
        list[str] | None,
        typer.Option(
            "--language",
            help="Restrict to language (repeatable): python/go/typescript/javascript/rust",
        ),
    ] = None,
    seed_id: Annotated[
        list[str] | None,
        typer.Option("--seed-id", help="Restrict to allowlist seed_id (repeatable)"),
    ] = None,
    no_tests_required: Annotated[
        bool,
        typer.Option(
            "--no-tests-required",
            help="Relax require_tests filter (debug only; cert path still needs tests)",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit GitMineReport JSON"),
    ] = False,
) -> None:
    """Mine multi-lang allowlist via git-clone-only merge commits (M8 default).

    DEFAULT LIVE PATH until Oxylabs credentials are present: local git clone +
    merge-commit history. Never uses GitHub HTTP / Oxylabs for diffs. Records
    language histogram + honest under-supply. Emits candidate trees + report
    under --out for ship consumption.
    """
    import json as json_lib

    from swe_factory.sources.allowlist import scale_inventory_report
    from swe_factory.sources.git_mine import GitMineError, mine_allowlist_git_only

    try:
        report = mine_allowlist_git_only(
            work_root=out,
            target_candidates=max(1, target),
            max_merges_per_seed=max(1, max_merges),
            max_keeps_per_seed=max(1, max_keeps_per_seed),
            languages=language,
            seed_ids=seed_id,
            write_artifacts=True,
            require_tests=not no_tests_required,
        )
    except GitMineError as exc:
        typer.secho(f"mine-allowlist: failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = report.to_dict()
    inv = scale_inventory_report()
    payload["scale_inventory"] = inv
    if json_out:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "mine-allowlist: OK "
            f"kept={report.keep_count} rejected={report.reject_count} "
            f"target={report.target_candidates} mode={report.mode} "
            f"history={report.history_authority} oxylabs={report.oxylabs_status}"
        )
        typer.echo(f"  languages_kept={report.language_kept}")
        typer.echo(f"  languages_inventory={report.language_inventory}")
        if report.under_supply:
            typer.echo("  under_supply:")
            for reason in report.under_supply:
                typer.echo(f"    - {reason}")
        typer.echo(f"  report={out / 'git_mine_report.json'}")
        typer.echo(f"  candidates={out / 'candidates.jsonl'}")
        if (out / "funnel_report.json").is_file():
            typer.echo(f"  funnel={out / 'funnel_report.json'}")
        if (out / "skip_reasons.json").is_file():
            typer.echo(f"  skip_reasons={out / 'skip_reasons.json'}")


@app.command("funnel-report")
def funnel_report_cmd(
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Optional path to write funnel hardening report JSON",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option(
            "--max-workers",
            help="Requested parallel envbuild workers (clamped to hard max 24)",
            min=1,
        ),
    ] = 16,
    target: Annotated[
        int,
        typer.Option("--target", help="Scale target for default funnel config", min=1),
    ] = 70,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit FunnelReport JSON"),
    ] = False,
) -> None:
    """Emit M9 funnel hardening docs: skip reasons, monorepo policy, worker caps.

    Does not run docker or network. Safe icebox audit for scale ≥70 operations:
    documented skip codes, parallelism bound (≤16 default, hard max 24), and
    off-limits docker refuse policy (mission-test-pg / challenge-prism* / acproxy).
    """
    import json as json_lib

    from swe_factory.envbuild.parallel import (
        HARD_MAX_ENVBUILD_WORKERS,
        clamp_envbuild_workers,
    )
    from swe_factory.sources.funnel import (
        FunnelConfig,
        document_all_skip_reasons,
        make_scale_funnel_report,
    )

    workers = clamp_envbuild_workers(max_workers)
    cfg = FunnelConfig(max_envbuild_workers=workers)
    if target >= 70:
        from swe_factory.sources.funnel import default_funnel_config_for_scale

        cfg = default_funnel_config_for_scale(target)
        cfg.max_envbuild_workers = workers
    report = make_scale_funnel_report(
        cfg,
        notes=[
            "funnel-report CLI: docs only (no docker damage)",
            f"requested_workers={max_workers} clamped={workers} "
            f"hard_max={HARD_MAX_ENVBUILD_WORKERS}",
        ],
    )
    payload = report.to_dict()
    payload["skip_catalog"] = document_all_skip_reasons()
    if out is not None:
        report.write_json(out)
        payload["written_to"] = str(out)
    if json_out or out is None:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "funnel-report: OK "
            f"workers={workers}/{HARD_MAX_ENVBUILD_WORKERS} "
            f"skip_codes={len(payload['documented_skip_catalog'])} "
            f"parallelism_bounded={payload['parallelism_bounded']}"
        )
        typer.echo(f"  out={out}")


@app.command("oxylabs-probe")
def oxylabs_probe_cmd(
    url: Annotated[
        str,
        typer.Option("--url", help="Public github.com URL for universal scrape smoke"),
    ] = "https://github.com/psf/requests",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Optional evidence JSON path (records blocked/pass honestly)",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit probe evidence JSON"),
    ] = False,
) -> None:
    """VAL-OXY-005 live Oxylabs universal probe (honest blocked if creds missing).

    Attempts a single public github.com fetch only when OXYLABS_* are set.
    Otherwise writes status=blocked evidence and exits 0 for the blocked case
    (does not invent a pass). Exit 1 when creds were present but probe failed.
    """
    import json as json_lib

    from swe_factory.sources.git_mine import probe_oxylabs_live

    evidence = probe_oxylabs_live(url=url)
    dest = out or Path("datasets/oxylabs_probe_val_oxy_005.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json_lib.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = dict(evidence)
    payload["evidence_path"] = str(dest)
    if json_out:
        typer.echo(json_lib.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"oxylabs-probe: status={evidence.get('status')} ok={evidence.get('ok')} "
            f"credentials_present={evidence.get('credentials_present')} "
            f"content_bytes={evidence.get('content_bytes')}"
        )
        typer.echo(f"  reason={evidence.get('reason')}")
        typer.echo(f"  evidence={dest}")
    status = str(evidence.get("status") or "")
    if status == "passed":
        raise typer.Exit(code=0)
    if status in {"blocked", "pending"}:
        # Honest non-pass without faking success; exit 0 so offline CI can record evidence
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


@app.command("ledger")
def ledger_cmd(
    path: Annotated[
        Path | None,
        typer.Option(
            "--path",
            help="Ledger JSONL path (default datasets/.factory/ledger.jsonl)",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit ledger_summary JSON"),
    ] = False,
    write_summary: Annotated[
        bool,
        typer.Option(
            "--write-summary",
            help="Also write ledger_summary.json next to the ledger",
        ),
    ] = False,
) -> None:
    """Show exact + unresolved reserved spend against the hard $600 cap.

    VAL-HARNESS-002: total exact + reserved must stay ≤ cap; calls linked by
    stage/task/model in the durable ledger.
    """
    from swe_factory.accounting import (
        DEFAULT_CAP_USD,
        AccountingError,
        BudgetLedger,
        default_ledger_path,
    )

    settings = load_settings()
    ledger_path = path or default_ledger_path(Path.cwd())
    cap = settings.budget_usd
    try:
        if not ledger_path.is_file():
            # Empty summary when no ledger yet (honest zero spend).
            summary_payload = {
                "path": str(ledger_path),
                "cap_usd": str(cap),
                "settled_exact_usd": "0",
                "open_reserved_usd": "0",
                "total_commit_usd": "0",
                "remaining_usd": str(cap),
                "open_call_count": 0,
                "settled_call_count": 0,
                "unknown_billing_count": 0,
                "has_unknown_billing": False,
                "under_cap": True,
                "by_stage": {},
                "by_task": {},
                "by_model": {},
                "exists": False,
            }
        else:
            ledger = BudgetLedger(ledger_path, cap_usd=cap)
            summary = ledger.summary()
            summary_payload = summary.to_dict()
            summary_payload["exists"] = True
            if write_summary:
                out = ledger.write_summary_json()
                summary_payload["summary_path"] = str(out)
    except AccountingError as exc:
        typer.secho(f"ledger: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(json.dumps(summary_payload, indent=2, sort_keys=True))
        return
    typer.echo(
        "ledger: "
        f"cap={summary_payload['cap_usd']} "
        f"settled={summary_payload['settled_exact_usd']} "
        f"reserved={summary_payload['open_reserved_usd']} "
        f"remaining={summary_payload['remaining_usd']} "
        f"under_cap={summary_payload['under_cap']} "
        f"path={summary_payload['path']}"
    )
    if float(str(DEFAULT_CAP_USD)) and not summary_payload["under_cap"]:
        raise typer.Exit(code=2)


@app.command("eval-deepswe")
def eval_deepswe_cmd(
    product_root: Annotated[
        Path,
        typer.Option(
            "--product-root",
            help="Product Harbor root (default datasets/deepswe_v1 with tasks/*)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/deepswe_v1"),
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Report output directory (report.json + ledger_summary.json)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/panel_deepswe_eval"),
    max_packs: Annotated[
        int,
        typer.Option("--max-packs", help="Max product packs to score (preferred order first)"),
    ] = 5,
    k: Annotated[
        int,
        typer.Option("--k", help="Trials per model per pack (pass@k; first wave k=1)"),
    ] = 1,
    n_concurrent: Annotated[
        int,
        typer.Option(
            "--n-concurrent",
            help="Pier/docker concurrency (must be 1 for DeepSWE serial fidelity)",
        ),
    ] = 1,
    hard_stop_usd: Annotated[
        float,
        typer.Option(
            "--hard-stop-usd",
            help="Ledger hard spend stop for this eval wave (default $300)",
        ),
    ] = 300.0,
    reserve_usd: Annotated[
        float,
        typer.Option(
            "--reserve-usd",
            help="Worst-case USD reserved per mini-swe trial before settle",
        ),
    ] = 25.0,
    jobs_dir: Annotated[
        Path,
        typer.Option(
            "--jobs-dir",
            help="Pier job workdir root under /tmp/harbor-deepswe-jobs*",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("/tmp/harbor-deepswe-jobs-eval"),
    pier_bin: Annotated[
        Path | None,
        typer.Option(
            "--pier-bin",
            help="Path to pier binary (default /tmp/pier-venv/bin/pier or PIER_BIN)",
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=False,
        ),
    ] = None,
    pack_id: Annotated[
        list[str] | None,
        typer.Option(
            "--pack-id",
            help="Restrict to pack id(s) (repeatable); default preferred python dual-truth",
        ),
    ] = None,
    skip_preflight: Annotated[
        bool,
        typer.Option(
            "--skip-preflight",
            help="Skip oracle/nop dual-truth preflight (not recommended for live)",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Offline unit path: mock pier reward matrix (no docker/LLM)",
        ),
    ] = False,
    no_reclaim: Annotated[
        bool,
        typer.Option(
            "--no-reclaim",
            help="Do not delete jobs_dir before starting",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit eval report path + summary as JSON"),
    ] = False,
) -> None:
    """DeepSWE-grade Pier + mini-swe-agent serial eval (VAL-DEVAL-001..007).

    fidelity=pier_miniswe_harbor only. Models: x-ai/grok-4.5 + moonshotai/kimi-k2.6.
    NEVER uses never-solve CLI panel or host soft L2 as DeepSWE hardness.
    """
    from decimal import Decimal

    from swe_factory.panel.eval_deepswe import (
        DEEPSWE_EVAL_FIDELITY,
        DEEPSWE_EVAL_MODELS,
        DeepSWEEvalError,
        mocked_miniswe_invoker,
        run_deepswe_eval,
    )

    if int(n_concurrent) != 1:
        typer.secho(
            "eval-deepswe: n_concurrent must be 1 (serial pier/docker)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    invoker = None
    if offline:
        # Build a deterministic mock matrix later once packs are known; for help / empty
        # roots we still construct a callable that never invents unsolicited solves.
        # Degenerate offline without packs: empty matrix invoker.
        invoker = mocked_miniswe_invoker({})

    try:
        # Offline: discover packs first so mock matrix covers them if present.
        if offline:
            from swe_factory.panel.eval_deepswe import load_product_packs

            try:
                keeps = load_product_packs(
                    product_root,
                    max_packs=max_packs,
                    pack_ids=list(pack_id) if pack_id else None,
                )
            except DeepSWEEvalError:
                keeps = []
            matrix: dict[str, dict[str, list[bool]]] = {}
            for keep in keeps:
                pid = str(keep["task_id"])
                # Default offline: neither model solves (honest hard bias; never invent keep).
                matrix[pid] = {m: [False] * max(1, int(k)) for m in DEEPSWE_EVAL_MODELS}
            invoker = mocked_miniswe_invoker(matrix)

        report = run_deepswe_eval(
            product_root=product_root,
            out_dir=out,
            max_packs=max_packs,
            pack_ids=list(pack_id) if pack_id else None,
            models=list(DEEPSWE_EVAL_MODELS),
            k=int(k),
            n_concurrent=int(n_concurrent),
            hard_stop_usd=hard_stop_usd,
            reserve_usd=reserve_usd,
            jobs_dir=jobs_dir,
            pier_bin=pier_bin,
            preflight=not skip_preflight,
            invoker=invoker,
            offline=offline,
            reclaim=not no_reclaim,
        )
    except DeepSWEEvalError as exc:
        typer.secho(f"eval-deepswe: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"eval-deepswe: unexpected error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    report_path = Path(out) / "report.json"
    payload = {
        "ok": True,
        "fidelity": report.fidelity,
        "models": list(report.models),
        "n_concurrent": report.n_concurrent,
        "k": report.k,
        "hard_stop_usd": format(Decimal(str(report.hard_stop_usd)), "f"),
        "pass_at_k_present": True,
        "n_packs_requested": report.n_packs_requested,
        "n_packs_scored": report.n_packs_scored,
        "spend_usd": format(report.total_spend_usd, "f"),
        "remaining_usd": format(report.remaining_usd, "f"),
        "budget_stop": report.budget_stop,
        "stop_reason": report.stop_reason,
        "invented_rewards": report.invented_rewards,
        "wall_s": report.wall_s,
        "offline": report.offline,
        "report": str(report_path.resolve()) if report_path.exists() else str(report_path),
        "out_dir": str(Path(out).resolve()),
        "agent": report.agent,
        "scaffold": report.scaffold,
        "never_solve_panel": False,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "eval-deepswe: "
            f"fidelity={report.fidelity} "
            f"models={','.join(report.models)} "
            f"n_scored={report.n_packs_scored}/{report.n_packs_requested} "
            f"n_concurrent={report.n_concurrent} "
            f"spend={format(report.total_spend_usd, 'f')} "
            f"hard_stop={format(Decimal(str(report.hard_stop_usd)), 'f')} "
            f"budget_stop={report.budget_stop} "
            f"wall_s={report.wall_s}"
        )
        typer.echo(f"  report={report_path}")
        if report.fidelity != DEEPSWE_EVAL_FIDELITY:
            typer.secho(
                "eval-deepswe: WARNING fidelity is not pier_miniswe_harbor",
                fg=typer.colors.YELLOW,
                err=True,
            )


@app.command("panel")
def panel_cmd(
    offline_matrix: Annotated[
        bool,
        typer.Option(
            "--offline-matrix",
            help="Run offline puzzle-band demo from a fixed solve matrix (no LLM)",
        ),
    ] = False,
    keep_demo: Annotated[
        bool,
        typer.Option(
            "--keep-demo",
            help="With --offline-matrix, use borderline keep matrix (default)",
        ),
    ] = True,
    solve_all_demo: Annotated[
        bool,
        typer.Option(
            "--solve-all-demo",
            help="With --offline-matrix, demonstrate solve-all drop",
        ),
    ] = False,
    solve_none_demo: Annotated[
        bool,
        typer.Option(
            "--solve-none-demo",
            help="With --offline-matrix, demonstrate solve-none drop",
        ),
    ] = False,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output directory for panel report + ledger",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/panel_offline"),
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit PanelRunResult JSON"),
    ] = False,
    live_canary: Annotated[
        bool,
        typer.Option(
            "--live-canary",
            help="Live k=1 probe on required panel models if remaining budget allows",
        ),
    ] = False,
    real_keeps: Annotated[
        Path | None,
        typer.Option(
            "--real-keeps",
            help=(
                "Real-PR keep root or pack dir(s): run full Grok+Kimi matrix on each "
                "discovered keep until remaining budget is $0 (VAL-RPANEL-002)"
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help=(
                "With --real-keeps: live OpenRouter rollouts (requires key + remaining "
                "budget). Default is offline soft-matrix (no invented live rewards)."
            ),
        ),
    ] = False,
) -> None:
    """Hardness panel: fixed scaffold, required models, pass@k band filter.

    Offline unit/demo is mandatory for CI. Live canary is optional and blocked
    unless remaining budget authorizes worst-case reservations (≤ $600 cap).
    Real-PR models (VAL-RPANEL-001): x-ai/grok-4.5 + moonshotai/kimi-k2.6.
    Full keep panel until remaining $0: ``--real-keeps PATH`` (``--live`` when authorized).
    """
    from decimal import Decimal

    from swe_factory.accounting import BudgetLedger, default_ledger_path
    from swe_factory.panel.runner import (
        REQUIRED_PANEL_MODELS,
        canary_affordable,
        discover_real_pr_panel_keeps,
        offline_panel_from_matrix,
        offline_tworollout_borderline_matrix,
        run_panel_until_budget_zero,
    )

    settings = load_settings()
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "ledger.jsonl"

    if real_keeps is not None:
        keeps = discover_real_pr_panel_keeps([real_keeps])
        if not keeps:
            typer.secho(
                f"panel: no Real-PR keeps with instruction data under {real_keeps}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        if live:
            if not settings.has_api_key():
                typer.secho(
                    "panel: --live requires OPENROUTER_API_KEY",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            from swe_factory.openrouter import OpenRouterClient

            live_path = default_ledger_path(Path.cwd())
            live_path.parent.mkdir(parents=True, exist_ok=True)
            ledger = BudgetLedger(
                live_path,
                cap_usd=settings.budget_usd,
                worst_case_cost_usd=Decimal("1.50"),
                run_id="panel-real-pr-live",
            )
            try:
                with OpenRouterClient.from_settings(settings) as client:
                    batch = run_panel_until_budget_zero(
                        keeps=keeps,
                        ledger=ledger,
                        client=client,
                        models=list(REQUIRED_PANEL_MODELS),
                        k=2,
                        stage="hardness-panel-real-pr",
                        soft_solver=lambda *_a, **_k: False,
                        reserve_usd=Decimal("1.50"),
                        allow_missing_cost_as_zero=False,
                    )
            except Exception as exc:  # noqa: BLE001
                typer.secho(f"panel: live real-keeps error: {exc}", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1) from exc
            summary_path = ledger.write_summary_json(out / "ledger_summary.json")
            report_path = out / "panel_real_keeps_report.json"
            report_path.write_text(
                json.dumps(batch.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            payload = {
                "ok": True,
                "mode": "real-keeps-live",
                "models": list(batch.models),
                "keeps_submitted": len(keeps),
                "completed_keep_ids": list(batch.completed_keep_ids),
                "partial_keep_ids": list(batch.partial_keep_ids),
                "skipped_keep_ids": list(batch.skipped_keep_ids),
                "budget_stop": batch.budget_stop,
                "stop_reason": batch.stop_reason,
                "total_cost_usd": format(batch.total_cost_usd, "f"),
                "remaining_usd": format(batch.remaining_usd, "f"),
                "invented_rewards": False,
                "report": str(report_path),
                "ledger": str(live_path),
                "ledger_summary": str(summary_path),
                "under_cap": ledger.summary().under_cap,
            }
            if json_out:
                typer.echo(json.dumps(payload, indent=2, sort_keys=True))
            else:
                typer.echo(
                    "panel: real-keeps-live "
                    f"models={','.join(batch.models)} "
                    f"completed={len(batch.completed_keep_ids)} "
                    f"partial={len(batch.partial_keep_ids)} "
                    f"skipped={len(batch.skipped_keep_ids)} "
                    f"cost={format(batch.total_cost_usd, 'f')} "
                    f"remaining={format(batch.remaining_usd, 'f')} "
                    f"budget_stop={batch.budget_stop}"
                )
            return

        # Offline multi-keep: per-keep in-band matrix, still hard-stops if ledger cap hits $0.
        offline_ledger = BudgetLedger(
            ledger_path,
            cap_usd=settings.budget_usd,
            worst_case_cost_usd=Decimal("0.01"),
            run_id="panel-real-pr-offline",
        )
        matrix_tpl = offline_tworollout_borderline_matrix()
        from swe_factory.openrouter import ChatResult as _ChatResult
        from swe_factory.openrouter import ScriptedChatClient, TokenUsage
        from swe_factory.panel.runner import run_panel

        keep_results = []
        completed: list[str] = []
        partial: list[str] = []
        skipped: list[str] = []
        budget_stop = False
        stop_reason: str | None = None
        total_cost = Decimal("0")
        full_need = Decimal("0.01") * Decimal(len(REQUIRED_PANEL_MODELS) * 4)
        for idx, entry in enumerate(keeps):
            remaining = offline_ledger.remaining_usd()
            if remaining < full_need:
                budget_stop = True
                stop_reason = (
                    f"budget_stop: remaining_usd={format(remaining, 'f')} "
                    f"< full panel need {format(full_need, 'f')} before "
                    f"keep {entry['task_id']!r}; no invented rewards"
                )
                skipped.extend(str(k["task_id"]) for k in keeps[idx:])
                break
            # Build offline scripted client for this keep from borderline matrix.
            responses: list[_ChatResult | Exception] = []
            for model, row in matrix_tpl.items():
                for i, _s in enumerate(row):
                    responses.append(
                        _ChatResult(
                            model=model,
                            text=f"PATCH {model} #{i}",
                            usage=TokenUsage(10, 5, 15),
                            request_id=f"off-{entry['task_id']}-{model}-{i}",
                            cost_usd=Decimal("0"),
                            finish_reason="stop",
                            raw_usage={},
                        )
                    )
            scripted_client = ScriptedChatClient(responses=responses)
            indices = {m: 0 for m in matrix_tpl}

            def soft(
                model: str,
                messages: object,
                chat: object,
                _indices: dict[str, int] = indices,
                _matrix: dict[str, list[bool]] = matrix_tpl,
            ) -> bool:
                del messages, chat
                row = _matrix[model]
                i = _indices[model]
                if i >= len(row):
                    return False
                _indices[model] = i + 1
                return bool(row[i])

            result = run_panel(
                task_id=str(entry["task_id"]),
                problem_statement=str(entry["problem_statement"]),
                ledger=offline_ledger,
                client=scripted_client,
                models=list(REQUIRED_PANEL_MODELS),
                k=4,
                stage="hardness-panel-real-pr",
                soft_solver=soft,
                reserve_usd=Decimal("0.01"),
                allow_missing_cost_as_zero=True,
                pack_path=entry.get("pack_path"),
                pack_id=entry.get("pack_id") or entry["task_id"],
                stop_on_budget=True,
            )
            keep_results.append(result)
            total_cost += result.total_cost_usd
            if result.budget_stop or not result.panel_complete:
                budget_stop = True
                stop_reason = result.stop_reason or "budget_stop mid-keep"
                partial.append(str(entry["task_id"]))
                skipped.extend(str(k["task_id"]) for k in keeps[idx + 1 :])
                break
            completed.append(str(entry["task_id"]))
        report = {
            "mode": "real-keeps-offline",
            "models": list(REQUIRED_PANEL_MODELS),
            "keeps_submitted": len(keeps),
            "completed_keep_ids": completed,
            "partial_keep_ids": partial,
            "skipped_keep_ids": skipped,
            "budget_stop": budget_stop,
            "stop_reason": stop_reason,
            "total_cost_usd": format(total_cost, "f"),
            "remaining_usd": format(offline_ledger.remaining_usd(), "f"),
            "invented_rewards": False,
            "scaffold_meta": {
                "name": "pier/mini-swe-agent",
                "agent": "mini-swe-agent",
                "runtime": "pier",
            },
            "keep_results": [r.to_dict() for r in keep_results],
        }
        report_path = out / "panel_real_keeps_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary_path = offline_ledger.write_summary_json(out / "ledger_summary.json")
        payload = {
            "ok": True,
            **report,
            "report": str(report_path),
            "ledger": str(ledger_path),
            "ledger_summary": str(summary_path),
            "under_cap": offline_ledger.summary().under_cap,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                "panel: real-keeps-offline "
                f"models={','.join(REQUIRED_PANEL_MODELS)} "
                f"completed={len(completed)} skipped={len(skipped)} "
                f"budget_stop={budget_stop}"
            )
            typer.echo(f"  report={report_path}")
        return

    if offline_matrix:
        if solve_all_demo:
            matrix = {m: [True, True] for m in REQUIRED_PANEL_MODELS}
            task_id = "offline-solve-all-demo"
        elif solve_none_demo:
            matrix = {m: [False, False] for m in REQUIRED_PANEL_MODELS}
            task_id = "offline-solve-none-demo"
        else:
            # Borderline keep: zeros on early models, partial on last → disc high.
            matrix = offline_tworollout_borderline_matrix()
            task_id = "offline-keep-demo"
            _ = keep_demo  # flag exists for CLI discoverability; default path is keep

        result = offline_panel_from_matrix(
            task_id=task_id,
            solve_matrix=matrix,
            ledger_path=ledger_path,
            cap_usd=Decimal(str(settings.budget_usd)),
            pack_path="datasets/deepswe_v1/tasks/offline-demo",
            pack_id=task_id,
            problem_statement=(
                "Offline hardness panel matrix (no OpenRouter calls). "
                "Multi-file regression in core module plumbing."
            ),
        )
        report_path = out / "panel_report.json"
        report_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ledger = BudgetLedger(ledger_path, cap_usd=settings.budget_usd)
        summary_path = ledger.write_summary_json(out / "ledger_summary.json")
        payload = {
            "ok": True,
            "mode": "offline-matrix",
            "is_keep": result.is_keep,
            "verdict": result.decision.verdict,
            "rule": result.decision.rule,
            "models": list(result.reserved_models),
            "pass_at_k": result.decision.frontier_pass_at_k,
            "discrimination": result.decision.discrimination,
            "report": str(report_path),
            "ledger": str(ledger_path),
            "ledger_summary": str(summary_path),
            "under_cap": ledger.summary().under_cap,
            "hardness": result.decision.to_dict(),
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                "panel: offline-matrix "
                f"verdict={result.decision.verdict} rule={result.decision.rule} "
                f"pass@k={result.decision.frontier_pass_at_k:.4f} "
                f"disc={result.decision.discrimination:.4f} "
                f"models={','.join(result.reserved_models)}"
            )
            typer.echo(f"  report={report_path}")
            typer.echo(f"  ledger={ledger_path}")
        return

    if live_canary:
        if not settings.has_api_key():
            typer.secho(
                "panel: live-canary requires OPENROUTER_API_KEY",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        from swe_factory.openrouter import OpenRouterClient
        from swe_factory.panel.runner import run_panel

        # Prefer durable mission ledger for live spend tracking.
        live_path = default_ledger_path(Path.cwd())
        live_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = BudgetLedger(
            live_path,
            cap_usd=settings.budget_usd,
            worst_case_cost_usd=Decimal("1.50"),
            run_id="panel-live-canary",
        )
        if not canary_affordable(ledger, k=1, reserve_usd=Decimal("1.50")):
            typer.secho(
                "panel: live-canary blocked — remaining budget insufficient "
                f"(remaining={ledger.remaining_usd()})",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=3)
        # k=1 probe lunges: never score as certified keep (soft solver always false).
        # Proves model ids + accounting only (cheap canary).
        try:
            with OpenRouterClient.from_settings(settings) as client:
                result = run_panel(
                    task_id="live-canary-probe",
                    problem_statement=(
                        "Trivial probe: report the string ok as a one-line patch "
                        "header comment only."
                    ),
                    ledger=ledger,
                    client=client,
                    models=list(REQUIRED_PANEL_MODELS),
                    k=1,
                    stage="hardness-panel-canary",
                    soft_solver=lambda *_a, **_k: False,
                    reserve_usd=Decimal("1.50"),
                    max_tokens=32,
                    allow_missing_cost_as_zero=False,
                )
        except Exception as exc:  # noqa: BLE001 — surface canary failure cleanly
            typer.secho(f"panel: live-canary error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        summary_path = ledger.write_summary_json()
        payload = {
            "ok": True,
            "mode": "live-canary",
            "models": list(result.reserved_models),
            "is_keep": result.is_keep,
            "rule": result.decision.rule,
            "total_cost_usd": format(result.total_cost_usd, "f"),
            "ledger": str(live_path),
            "ledger_summary": str(summary_path),
            "under_cap": ledger.summary().under_cap,
            "unknown_billing": ledger.has_unknown_billing(),
            "per_model": [m.to_dict() for m in result.models],
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                "panel: live-canary "
                f"models={','.join(result.reserved_models)} "
                f"cost={format(result.total_cost_usd, 'f')} "
                f"under_cap={ledger.summary().under_cap} "
                f"rule={result.decision.rule}"
            )
        return

    typer.secho(
        "panel: require --offline-matrix, --real-keeps PATH, or --live-canary after offline green",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("micro-keep")
def micro_keep_cmd(
    seed_id: Annotated[
        str,
        typer.Option(
            "--seed-id",
            help="Allowlist seed (default fixture_tiny_green offline multi-file)",
        ),
    ] = "fixture_tiny_green",
    mutation: Annotated[
        str,
        typer.Option("--mutation", help="multi_fault | function_removal"),
    ] = "multi_fault",
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Working + export root for micro-keep evidence",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/micro_keep"),
    live: Annotated[
        bool,
        typer.Option(
            "--live/--offline",
            help="Live OpenRouter panel (reserve/settle) vs offline scripted panel",
        ),
    ] = True,
    docker_oracle: Annotated[
        bool,
        typer.Option(
            "--docker-oracle/--fake-oracle",
            help="Certified oracle backend (Docker G1–G5 vs FakeOracle)",
        ),
    ] = True,
    docker_envbuild: Annotated[
        bool,
        typer.Option(
            "--docker-envbuild/--skip-envbuild-docker",
            help="Run Docker envbuild stage (default skip for fixture micro path)",
        ),
    ] = False,
    panel_k: Annotated[
        int,
        typer.Option("--panel-k", help="Rollouts per panel model (keep band uses k)"),
    ] = 2,
    micro_cap: Annotated[
        float,
        typer.Option(
            "--micro-cap",
            help="Stop/escalate if run spend exceeds this USD without a keep",
        ),
    ] = 80.0,
    soft_backend: Annotated[
        str,
        typer.Option(
            "--soft-backend",
            help="Panel soft solver: local (host pytest), oracle, or never",
        ),
    ] = "local",
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit MicroKeepResult JSON"),
    ] = False,
) -> None:
    """Live micro path: sources→env→produce→oracle→panel→export (VAL-CROSS-002).

    Prefer synthetic_grounded multi-file seeds with pinned SHAs. Preserves all
    gate thresholds. Fail closed if keep export lacks panel hardness fields.
    Exact spend is ledgered under datasets/.factory/ledger.jsonl. Escalates
    with funnel numbers if micro cap exceeded without a keep.
    """
    from decimal import Decimal

    from swe_factory.pipeline.micro_keep import MicroKeepError, run_micro_keep

    settings = load_settings()
    if live and not settings.has_api_key():
        typer.secho(
            "micro-keep: --live requires OPENROUTER_API_KEY",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    backend = soft_backend.strip().lower()
    if backend not in {"local", "oracle", "never"}:
        typer.secho(
            f"micro-keep: unknown --soft-backend {soft_backend!r} (local|oracle|never)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if panel_k <= 0:
        typer.secho("micro-keep: --panel-k must be positive", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    try:
        result = run_micro_keep(
            out_dir=out,
            seed_id=seed_id,
            mutation=mutation,
            settings=settings,
            micro_cap_usd=Decimal(str(micro_cap)),
            panel_k=panel_k,
            use_docker_oracle=docker_oracle,
            use_docker_envbuild=docker_envbuild,
            soft_backend=backend,  # type: ignore[arg-type]
            live_panel=live,
            require_immutable_sha=True,
        )
    except MicroKeepError as exc:
        typer.secho(f"micro-keep: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = result.to_dict()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = (
            "KEEP"
            if result.is_keep
            else ("ESCALATE" if result.escalated else ("OK-DROP" if result.ok else "FAIL"))
        )
        typer.echo(
            f"micro-keep: {status} "
            f"keep={result.is_keep} "
            f"escalated={result.escalated} "
            f"spend_usd={format(result.spend_exact_usd, 'f')} "
            f"cap={format(result.micro_cap_usd, 'f')}"
        )
        if result.instance_id:
            typer.echo(f"  instance_id={result.instance_id}")
        if result.source_track:
            typer.echo(f"  source_track={result.source_track}")
        if result.export_dir:
            typer.echo(f"  export={result.export_dir}")
        if result.stage_log_path:
            typer.echo(f"  stage_log={result.stage_log_path}")
        if result.ledger_path:
            typer.echo(f"  ledger={result.ledger_path}")
        typer.echo(f"  reason={result.reason}")
        typer.echo(f"  funnel={json.dumps(result.funnel, sort_keys=True)}")

    if result.is_keep:
        raise typer.Exit(code=0)
    if result.ok and (result.escalated or not result.is_keep):
        # Honest drop / escalate is success of the micro command process but
        # non-zero exit distinguishes from certified keep for automation.
        raise typer.Exit(code=3)
    raise typer.Exit(code=1)


@app.command("ship-v1")
def ship_v1_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="V1 ship output directory (tasks.jsonl, report.md, …)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/v1"),
    target: Annotated[
        int,
        typer.Option("--target", help="Soft target certified keep count (min grade band=20)"),
    ] = 20,
    max_keeps: Annotated[
        int,
        typer.Option("--max-keeps", help="Hard ceiling keep count (≤50)"),
    ] = 50,
    max_attempts: Annotated[
        int,
        typer.Option("--max-attempts", help="Max harvest attempts across diversified seeds"),
    ] = 200,
    panel_k: Annotated[
        int,
        typer.Option("--panel-k", help="Rollouts per panel model"),
    ] = 2,
    live: Annotated[
        bool,
        typer.Option("--live/--offline", help="Live OpenRouter panel vs offline only"),
    ] = True,
    docker_oracle: Annotated[
        bool,
        typer.Option(
            "--docker-oracle/--fake-oracle",
            help="Certified oracle via Docker (required for honest keeps)",
        ),
    ] = True,
    soft_backend: Annotated[
        str,
        typer.Option("--soft-backend", help="Panel soft solver backend: local|oracle|never"),
    ] = "local",
    try_real_pr: Annotated[
        bool,
        typer.Option(
            "--try-real-pr/--skip-real-pr",
            help="Attempt one real_pr keep; emit under-supply note if none",
        ),
    ] = True,
    seed_export: Annotated[
        list[Path] | None,
        typer.Option(
            "--seed-export",
            help="Prior certified export dir to ingest (repeatable)",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    spot_n: Annotated[
        int,
        typer.Option("--spot-n", help="Harness gold/null spot-check sample size"),
    ] = 3,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit ShipV1Result JSON"),
    ] = False,
) -> None:
    """Expand factory to 20–50 certified keeps and write datasets/v1 artifacts.

    Seeds from existing live keeps when present, diversifies modular allowlist
    (python/js/go) with pinned SHAs, multi-file only, exact ledger ≤ $600.
    No gate relaxation. Funnel counters include post-export success.
    """
    from decimal import Decimal

    from swe_factory.pipeline.ship_v1 import run_ship_v1

    settings = load_settings()
    if live and not settings.has_api_key():
        typer.secho("ship-v1: --live requires OPENROUTER_API_KEY", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if target < 1 or max_keeps < target or max_keeps > 50:
        typer.secho(
            "ship-v1: require 1 ≤ --target ≤ --max-keeps ≤ 50",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    backend = soft_backend.strip().lower()
    if backend not in {"local", "oracle", "never"}:
        typer.secho(
            f"ship-v1: unknown soft-backend {soft_backend!r}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    seeds = list(seed_export or [])
    result = run_ship_v1(
        out_dir=out,
        target_keeps=target,
        max_keeps=max_keeps,
        max_attempts=max_attempts,
        panel_k=panel_k,
        settings=settings,
        seed_exports=seeds,
        live_panel=live,
        use_docker_oracle=docker_oracle,
        soft_backend=backend,
        attempt_micro_cap_usd=Decimal("12"),
        soft_stop_spend_usd=Decimal("550"),
        spot_check_n=spot_n,
        spot_backend="local",
        try_real_pr=try_real_pr,
    )
    payload = result.to_dict()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "OK" if result.ok else "PARTIAL/FAIL"
        typer.echo(
            f"ship-v1: {status} keeps={result.keep_count} "
            f"spend={result.spend_total_usd} remaining={result.remaining_usd} "
            f"under_cap={result.under_cap}"
        )
        typer.echo(f"  out={result.out_dir}")
        if result.tasks_jsonl:
            typer.echo(f"  tasks_jsonl={result.tasks_jsonl}")
        if result.report_path:
            typer.echo(f"  report={result.report_path}")
        if result.ledger_summary_path:
            typer.echo(f"  ledger_summary={result.ledger_summary_path}")
        typer.echo(f"  languages={json.dumps(result.languages)}")
        typer.echo(f"  source_tracks={json.dumps(result.source_tracks)}")
        typer.echo(f"  reason={result.reason}")
        if result.under_supply_reasons:
            typer.echo("  under_supply:")
            for u in result.under_supply_reasons[:12]:
                typer.echo(f"    - {u}")

    if result.ok:
        raise typer.Exit(code=0)
    if result.keep_count >= 1:
        raise typer.Exit(code=3)
    raise typer.Exit(code=1)


@app.command("harbor-produce")
def harbor_produce_cmd(
    offline: Annotated[
        bool,
        typer.Option(
            "--offline/--no-offline",
            help="Use offline multi-module motors (default; no OpenRouter)",
        ),
    ] = True,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Filter motors: python, go, typescript (repeatable via comma list)",
        ),
    ] = None,
    seed_id: Annotated[
        str | None,
        typer.Option(
            "--seed-id",
            help="Single Harbor motor seed_id (default: all offline motors)",
        ),
    ] = None,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output root (tasks/<id>/… Harbor pack trees)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/harbor_motors"),
    work: Annotated[
        Path | None,
        typer.Option(
            "--work",
            help="Scratch workspace root for broken/green trees",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit materials/pack summary as JSON"),
    ] = False,
    list_seeds: Annotated[
        bool,
        typer.Option("--list-seeds", help="List offline Harbor motor seeds and exit"),
    ] = False,
) -> None:
    """Produce DeepSWE-ready multi-file Harbor materials (VAL-HARBOR-007).

    Offline multi-module Python/Go/TS motors emit solution.patch (multi-file hard
    floor), held-out test.patch, f2p/p2p node ids, and long-horizon instruction.md.
    Held-out tests do not ship in the agent environment/repo tree.
    """
    settings = load_settings()
    from swe_factory.oracle.gates import count_files_in_patch
    from swe_factory.producers.harbor_motors import (
        MOTOR_SEEDS,
        HarborMotorError,
        get_motor_seed,
        list_motor_seeds,
        produce_all_offline_motors,
        produce_harbor_pack,
    )

    if list_seeds:
        rows: list[dict[str, Any]] = [
            {
                "seed_id": s.seed_id,
                "language": s.language,
                "hard_track": s.hard_track,
                "modules": list(s.green_modules),
            }
            for s in MOTOR_SEEDS
        ]
        if json_out:
            typer.echo(json.dumps({"seeds": rows}, indent=2, sort_keys=True))
        else:
            for row in rows:
                modules = [str(m) for m in row["modules"]]
                typer.echo(
                    f"{row['seed_id']}\tlanguage={row['language']}\tmodules={','.join(modules)}"
                )
        raise typer.Exit(code=0)

    if not offline:
        typer.secho(
            "harbor-produce: live motors not enabled in this feature "
            "(use --offline multi-module motors; live ship is m6-harbor-ship-10-15)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    langs = None
    if language:
        langs = [part.strip() for part in language.split(",") if part.strip()]

    try:
        if seed_id:
            seed = get_motor_seed(seed_id)
            if langs and seed.language not in {
                ("typescript" if x in {"ts", "js", "javascript"} else x) for x in langs
            }:
                typer.secho(
                    f"harbor-produce: seed {seed_id!r} language={seed.language} "
                    f"does not match --language {language!r}",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            results = [
                produce_harbor_pack(
                    seed,
                    out_dir=out,
                    work_root=work,
                    instance_suffix="offline",
                )
            ]
        else:
            if langs is None:
                targets = list_motor_seeds()
                del targets  # discovery only; produce_all filters itself
            results = produce_all_offline_motors(
                out_dir=out,
                work_root=work,
                languages=langs,
                instance_suffix="offline",
            )
    except (HarborMotorError, KeyError) as exc:
        typer.secho(f"harbor-produce: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"harbor-produce: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    summaries: list[dict[str, Any]] = []
    hard_ok = True
    for result in results:
        mat = result.materials
        sol_files = list(mat.solution_files)
        if mat.hard_track and len(sol_files) < 2:
            hard_ok = False
        # Also verify on-disk solution.patch
        if result.pack_dir is not None:
            on_disk = count_files_in_patch(
                (result.pack_dir / "solution" / "solution.patch").read_text(encoding="utf-8")
            )
            product = [p for p in on_disk if not str(p).startswith("tests/")]
            if mat.hard_track and len(product) < 2:
                hard_ok = False
        summaries.append(
            {
                "task_id": mat.task_id,
                "seed_id": mat.seed_id,
                "language": mat.language,
                "solution_files": sol_files,
                "multi_file_ok": mat.multi_file_ok,
                "f2p_node_ids": list(mat.f2p_node_ids),
                "p2p_node_ids": list(mat.p2p_node_ids),
                "pack_dir": str(result.pack_dir) if result.pack_dir else None,
                "missing": list(result.missing),
                "held_out_path": mat.notes.get("held_out_path"),
                "provider_calls": mat.provider_calls,
            }
        )

    languages_sorted = sorted({str(s["language"]) for s in summaries})
    payload: dict[str, Any] = {
        "ok": hard_ok and all(not s["missing"] for s in summaries),
        "count": len(summaries),
        "out_dir": str(out),
        "languages": languages_sorted,
        "hard_multi_file_ok": hard_ok,
        "provider_calls": 0,
        "budget_usd": settings.budget_usd,
        "packs": summaries,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "harbor-produce: OK "
            f"count={payload['count']} languages={','.join(languages_sorted)} "
            f"hard_multi_file_ok={hard_ok} provider_calls=0 out={out}"
        )
        for row in summaries:
            f2p_ids = row["f2p_node_ids"]
            f2p_count = len(f2p_ids) if isinstance(f2p_ids, list) else 0
            typer.echo(
                f"  {row['task_id']}: lang={row['language']} "
                f"files={row['solution_files']} "
                f"f2p={f2p_count}"
            )
    if not payload["ok"]:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command("export-harbor")
def export_harbor_cmd(
    from_fixture: Annotated[
        bool,
        typer.Option(
            "--from-fixture",
            help="Emit offline Harbor pack from fixtures/tiny_offline (no LLM)",
        ),
    ] = False,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Harbor export output directory (tasks/<id>/… pack tree)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/harbor_fixture"),
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit pack summary as JSON"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config only; do not write packs"),
    ] = False,
) -> None:
    """Export DeepSWE/Harbor-complete pack trees (VAL-HARBOR-001/002, VAL-CROSS-006).

    Offline path: ``swe-factory export-harbor --from-fixture``. Live multi-pack
    ship arrives in m6-harbor-ship-10-15. Keeps V1 ``export`` command intact.
    """
    settings = load_settings()
    if dry_run and not from_fixture:
        typer.echo(f"export-harbor: dry-run OK (budget_usd={settings.budget_usd}, out={out})")
        raise typer.Exit(code=0)

    if not from_fixture:
        typer.secho(
            "export-harbor: require --from-fixture "
            "(live harbor ship arrives in later milestone features)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    from swe_factory.harbor.export_pack import HarborExportError
    from swe_factory.harbor.offline_fixture import run_offline_harbor_fixture

    try:
        result = run_offline_harbor_fixture(out_dir=out)
    except (HarborExportError, Exception) as exc:
        typer.secho(f"export-harbor: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "ok": True,
        "task_id": result.task_id,
        "pack_dir": str(result.pack_dir),
        "out_dir": str(result.out_dir),
        "missing": list(result.missing),
        "provider_calls": result.provider_calls,
        "complete": len(result.missing) == 0,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "export-harbor: OK "
            f"task_id={result.task_id} "
            f"complete={payload['complete']} "
            f"provider_calls={result.provider_calls} "
            f"pack={result.pack_dir}"
        )
    if result.missing:
        raise typer.Exit(code=1)


@app.command("export-real-harbor")
def export_real_harbor_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Product Harbor export root (datasets/deepswe_v1 style, tasks/<id>/…)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1"),
    source_track: Annotated[
        str,
        typer.Option(
            "--source-track",
            help="Must be real_pr for product promote (hybrid_curated refused)",
        ),
    ] = "real_pr",
    hybrid_bind: Annotated[
        bool,
        typer.Option(
            "--hybrid-bind/--no-hybrid-bind",
            help="Force hybrid motor bind (always refused on product real_pr path)",
        ),
    ] = False,
    from_spec: Annotated[
        Path | None,
        typer.Option(
            "--from-spec",
            help="JSON HarborPackSpec path for offline real_pr product export",
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate product gates only; do not write packs"),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON summary"),
    ] = False,
) -> None:
    """Export real_pr Harbor packs for product deepswe_v1 (VAL-RPACK-001..005).

    Requires ``source_track=real_pr`` + real HTTPS URL + 40-char base SHA and
    refuses hybrid_bind / motor packaging on the product surface. Offline motors
    must use ``export-harbor --from-fixture`` (non-product).
    """
    from swe_factory.harbor.real_pack import (
        RealPackError,
        assert_product_real_pr_export,
        export_real_harbor_pack,
        is_product_deepswe_dest,
        write_real_harbor_export,
    )
    from swe_factory.harbor.schema import HarborPackSpec

    track = (source_track or "").strip().lower()
    product = is_product_deepswe_dest(out)

    # VAL-RPACK-003: CLI/API errors if hybrid motors are forced on product flag.
    try:
        assert_product_real_pr_export(
            source_track=track or source_track,
            dest=out,
            copy_repo_into_environment="/tmp/hybrid_bind_forced" if hybrid_bind else None,
            force_product=True,
        )
    except RealPackError as exc:
        typer.secho(f"export-real-harbor: refuse: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if hybrid_bind:
        typer.secho(
            f"export-real-harbor: refuse hybrid_bind on product real_pr path (out={out})",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if track != "real_pr":
        typer.secho(
            "export-real-harbor: refuse source_track="
            f"{source_track!r}; product path requires real_pr "
            f"(out={out} product={product})",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if dry_run and from_spec is None:
        payload = {
            "ok": True,
            "dry_run": True,
            "out": str(out),
            "source_track": "real_pr",
            "product_surface": product,
            "required_relpaths": True,
            "refuse_hybrid": True,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                f"export-real-harbor: dry-run OK source_track=real_pr product={product} out={out}"
            )
        raise typer.Exit(code=0)

    if from_spec is None:
        typer.secho(
            "export-real-harbor: require --from-spec <HarborPackSpec.json> "
            "(or --dry-run to validate gates). Offline motors use export-harbor.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        raw = json.loads(Path(from_spec).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            specs = [HarborPackSpec.model_validate(item) for item in raw]
            if dry_run:
                for s in specs:
                    assert_product_real_pr_export(
                        source_track=s.task_toml.metadata.source_track or track,
                        repository_url=s.task_toml.metadata.repository_url,
                        base_commit=s.task_toml.metadata.base_commit_hash,
                        dest=out,
                        force_product=True,
                    )
                typer.echo(f"export-real-harbor: dry-run OK count={len(specs)} out={out}")
                raise typer.Exit(code=0)
            manifest, packs = write_real_harbor_export(specs, out_dir=out, overwrite=True)
            payload = {
                "ok": True,
                "count": len(packs),
                "task_ids": [p.task_id for p in packs],
                "manifest": str(manifest),
                "out": str(out),
                "source_track": "real_pr",
            }
        else:
            spec = HarborPackSpec.model_validate(raw)
            if dry_run:
                assert_product_real_pr_export(
                    source_track=spec.task_toml.metadata.source_track or track,
                    repository_url=spec.task_toml.metadata.repository_url,
                    base_commit=spec.task_toml.metadata.base_commit_hash,
                    dest=out,
                    force_product=True,
                )
                typer.echo(f"export-real-harbor: dry-run OK task_id={spec.task_id}")
                raise typer.Exit(code=0)
            result = export_real_harbor_pack(
                spec,
                dest=Path(out) / "tasks" / spec.task_id,
                overwrite=True,
                require_real_pr_track=True,
            )
            payload = {
                "ok": True,
                "task_id": result.task_id,
                "pack_dir": str(result.pack_dir),
                "validation_ok": result.validation.ok,
                "out": str(out),
                "source_track": "real_pr",
                "tree_complete": result.validation.tree_complete,
                "real_url_ok": result.validation.real_url_ok,
                "real_sha_ok": result.validation.real_sha_ok,
                "multi_file_ok": result.validation.multi_file_ok,
                "test_patch_ok": result.validation.test_patch_ok,
                "f2p_count": result.validation.f2p_count,
            }
    except RealPackError as exc:
        typer.secho(f"export-real-harbor: refuse: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        typer.secho(f"export-real-harbor: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(
            "export-real-harbor: OK "
            f"out={out} "
            f"ids={payload.get('task_ids') or payload.get('task_id')} "
            "source_track=real_pr"
        )


@app.command("harbor-oracle")
def harbor_oracle_cmd(
    from_fixture: Annotated[
        bool,
        typer.Option(
            "--from-fixture",
            help="Emit offline Harbor pack then run separate-verifier oracle",
        ),
    ] = False,
    pack_dir: Annotated[
        Path | None,
        typer.Option(
            "--pack-dir",
            help="Existing Harbor pack directory (tasks/<id>) to oracle",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Verifier backend: fake (offline fixture only) or docker (cert/integration)",
        ),
    ] = "fake",
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="When --from-fixture, pack export root (tasks/<id>/…)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/harbor_fixture"),
    certified: Annotated[
        bool,
        typer.Option(
            "--certified",
            help=(
                "DeepSWE cert path (VAL-ORCD-*): refuse fake backend, require docker "
                "sol=1/null=0 + isolation + audit fields (datasets/deepswe_v1)"
            ),
        ),
    ] = False,
    evidence_out: Annotated[
        Path | None,
        typer.Option(
            "--evidence-out",
            help="When --certified, write oracle_evidence.json path",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    audit_out: Annotated[
        Path | None,
        typer.Option(
            "--audit-out",
            help="When --certified, append gate_audit.jsonl path",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit oracle summary as JSON"),
    ] = False,
) -> None:
    """Separate-verifier oracle: solution reward=1, null reward=0 (VAL-HARBOR-003/004).

    Also checks agent isolation (VAL-HARBOR-005) and cert config/test.patch
    (VAL-HARBOR-006). Offline default: ``--backend fake`` (no Docker).

    DeepSWE certification (VAL-ORCD-001..007): pass ``--certified`` which
    **refuses** ``--backend fake`` and writes dual-truth audit evidence.
    """
    settings = load_settings()
    mode = (backend or "fake").strip().lower()
    if mode not in {"fake", "docker"}:
        typer.secho(
            f"harbor-oracle: unknown --backend {backend!r} (fake|docker)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    # VAL-ORCD-004: certified / deepswe_v1 dest always refuse fake
    if certified or "deepswe" in str(out).lower():
        from swe_factory.harbor.deepswe_cert import FakeBackendRejected, refuse_fake_backend

        try:
            refuse_fake_backend(mode, certified=True, dest=out)
        except FakeBackendRejected as exc:
            typer.secho(f"harbor-oracle: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc

    if certified:
        from swe_factory.harbor.deepswe_cert import (
            DeepSWECertError,
            FakeBackendRejected,
            certify_deepswe_pack,
        )
        from swe_factory.harbor.harbor_oracle import HarborOracleError
        from swe_factory.harbor.offline_fixture import run_offline_harbor_fixture

        provider_calls = 0
        try:
            if from_fixture:
                pack = run_offline_harbor_fixture(out_dir=out)
                target = pack.pack_dir
                provider_calls = pack.provider_calls
            elif pack_dir is not None:
                target = pack_dir
            else:
                typer.secho(
                    "harbor-oracle: --certified requires --from-fixture or --pack-dir",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=2)
            evidence_path = evidence_out or (Path(out) / "oracle_evidence.json")
            result = certify_deepswe_pack(
                target,
                backend=mode,
                evidence_out=evidence_path,
                audit_out=audit_out,
                dest_hint=out,
            )
        except FakeBackendRejected as exc:
            typer.secho(f"harbor-oracle: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
        except (DeepSWECertError, HarborOracleError) as exc:
            typer.secho(f"harbor-oracle: error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"harbor-oracle: error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        payload = {
            "ok": result.certified,
            "passed": result.certified,
            "certified": result.certified,
            "task_id": result.task_id,
            "pack_dir": result.pack_dir,
            "mode": result.backend,
            "backend": result.backend,
            "solution_reward": result.solution_reward,
            "null_reward": result.null_reward,
            "agent_isolated": result.isolation.clean,
            "isolation_status": "clean" if result.isolation.clean else "leak",
            "repository_url": result.pack_meta.repository_url,
            "base_commit_hash": result.pack_meta.base_commit_hash,
            "reason_codes": list(result.reason_codes),
            "reasons": list(result.reasons),
            "pier_ready": result.pier_ready.to_dict(),
            "audit": result.audit,
            "provider_calls": provider_calls,
            "budget_usd": settings.budget_usd,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            status = "OK" if result.certified else "FAIL"
            typer.echo(
                f"harbor-oracle: {status} certified={result.certified} "
                f"task_id={result.task_id} backend={result.backend} "
                f"solution={result.solution_reward} null={result.null_reward} "
                f"isolated={result.isolation.clean}"
            )
            for reason in result.reasons[:12]:
                typer.echo(f"  reason: {reason}")
        raise typer.Exit(code=0 if result.certified else 1)

    from swe_factory.harbor.harbor_oracle import (
        FakeHarborVerifier,
        HarborDockerVerifier,
        HarborOracleError,
        HarborOracleResult,
        run_harbor_oracle,
        run_offline_harbor_oracle_fixture,
    )

    provider_calls = 0
    oracle_result: HarborOracleResult
    try:
        if from_fixture:
            from swe_factory.harbor.harbor_oracle import HarborVerifierBackend

            oracle_backend: HarborVerifierBackend
            if mode == "fake":
                oracle_backend = FakeHarborVerifier()
            else:
                oracle_backend = HarborDockerVerifier(run_id="clihor")
            pack, oracle_result = run_offline_harbor_oracle_fixture(
                out_dir=out,
                backend=oracle_backend,
            )
            task_id = pack.task_id
            pack_path = str(pack.pack_dir)
            provider_calls = pack.provider_calls
        elif pack_dir is not None:
            if mode == "fake":
                oracle_result = run_harbor_oracle(
                    pack_dir, backend=FakeHarborVerifier(), mode="fake"
                )
            else:
                oracle_result = run_harbor_oracle(
                    pack_dir,
                    backend=HarborDockerVerifier(run_id="clihor"),
                    mode="docker",
                )
            task_id = oracle_result.task_id
            pack_path = str(pack_dir)
        else:
            typer.secho(
                "harbor-oracle: require --from-fixture or --pack-dir",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
    except HarborOracleError as exc:
        typer.secho(f"harbor-oracle: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"harbor-oracle: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "ok": oracle_result.passed,
        "passed": oracle_result.passed,
        "task_id": task_id,
        "pack_dir": pack_path,
        "mode": oracle_result.mode,
        "solution_reward": oracle_result.solution.reward,
        "null_reward": oracle_result.null.reward,
        "agent_isolated": oracle_result.agent_isolated,
        "config_ok": oracle_result.config_ok,
        "test_patch_ok": oracle_result.test_patch_ok,
        "provider_calls": provider_calls,
        "budget_usd": settings.budget_usd,
        "reasons": list(oracle_result.reasons),
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "OK" if oracle_result.passed else "FAIL"
        typer.echo(
            f"harbor-oracle: {status} "
            f"task_id={task_id} mode={oracle_result.mode} "
            f"solution={oracle_result.solution.reward} null={oracle_result.null.reward} "
            f"isolated={oracle_result.agent_isolated} "
            f"provider_calls={provider_calls}"
        )
        if oracle_result.reasons:
            for reason in oracle_result.reasons[:12]:
                typer.echo(f"  reason: {reason}")
    raise typer.Exit(code=0 if oracle_result.passed else 1)


@app.command("deepswe-oracle")
def deepswe_oracle_cmd(
    pack_dir: Annotated[
        Path | None,
        typer.Option(
            "--pack-dir",
            help="Harbor pack directory (tasks/<id>) to Docker-oracle certify",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Must be docker on cert path (fake is always refused)",
        ),
    ] = "docker",
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Evidence / audit parent dir (default datasets/deepswe_v1 cert stage)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1"),
    evidence_out: Annotated[
        Path | None,
        typer.Option(
            "--evidence-out",
            help="Write oracle_evidence.json (default: <out>/oracle_evidence.json)",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    audit_out: Annotated[
        Path | None,
        typer.Option(
            "--audit-out",
            help="Append gate_audit.jsonl for cert funnel",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit DeepSWE cert summary as JSON"),
    ] = False,
    no_pier_hooks: Annotated[
        bool,
        typer.Option(
            "--no-pier-hooks",
            help="Skip pier structural load smoke (still checks pack tree)",
        ),
    ] = False,
    real_pr: Annotated[
        bool,
        typer.Option(
            "--real-pr",
            help=(
                "Real-PR product cert path (VAL-RORC-001..004): require "
                "source_track=real_pr, write sol/null evidence files, "
                "refuse fake oracle_mode, isolation fail-closed"
            ),
        ),
    ] = False,
    oracle_mode: Annotated[
        str | None,
        typer.Option(
            "--oracle-mode",
            help=(
                "Explicit oracle mode (docker only on product path). "
                "fake/stub always refused for Real-PR cert (VAL-RORC-004)"
            ),
        ),
    ] = None,
) -> None:
    """Docker-only DeepSWE cert: sol=1, null=0, isolation, refuse fake.

    Default path covers historical VAL-ORCD-*. With ``--real-pr`` enforces
    product Real-PR gates (VAL-RORC-001..004): source_track=real_pr required,
    sol/null evidence files written, isolation blocks promote, fake always
    refused.
    """
    settings = load_settings()
    mode = (backend or "docker").strip().lower()
    explicit_mode = (oracle_mode or mode).strip().lower()

    from swe_factory.harbor.deepswe_cert import (
        DeepSWECertError,
        FakeBackendRejected,
        certify_deepswe_pack,
        refuse_fake_backend,
    )
    from swe_factory.harbor.real_oracle_cert import (
        RealOracleCertError,
        RealPrFakeOracleRejected,
        certify_real_pr_pack,
        refuse_fake_oracle_mode_real_pr,
    )

    # Always refuse fake on this product/cert CLI surface
    try:
        if real_pr or "deepswe" in str(out).lower():
            refuse_fake_oracle_mode_real_pr(
                mode, certified=True, dest=out, oracle_mode=explicit_mode
            )
        else:
            refuse_fake_backend(mode, certified=True, dest=out)
            if explicit_mode in {"fake", "stub", "mock", "offline"}:
                raise RealPrFakeOracleRejected(
                    f"deepswe-oracle refuses oracle_mode={explicit_mode!r} (VAL-RORC-004)"
                )
    except (FakeBackendRejected, RealPrFakeOracleRejected) as exc:
        typer.secho(f"deepswe-oracle: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if pack_dir is None:
        typer.secho(
            "deepswe-oracle: require --pack-dir (Docker cert path; no fake fixture mode)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    evidence_path = evidence_out or (out / "oracle_evidence.json")
    audit_path = audit_out or (out / "gate_audit.jsonl")

    if real_pr:
        try:
            rorc = certify_real_pr_pack(
                pack_dir,
                backend=mode,
                oracle_mode=explicit_mode,
                evidence_dir=out / "oracle_evidence",
                evidence_out=evidence_path,
                audit_out=audit_path,
                run_pier_hooks=not no_pier_hooks,
                dest_hint=out,
                run_id="realpr-cli",
                require_real_pr_track=True,
            )
        except (FakeBackendRejected, RealPrFakeOracleRejected) as exc:
            typer.secho(f"deepswe-oracle: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
        except (RealOracleCertError, DeepSWECertError) as exc:
            typer.secho(f"deepswe-oracle: error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"deepswe-oracle: error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        payload = {
            "ok": rorc.certified,
            "passed": rorc.certified,
            "certified": rorc.certified,
            "task_id": rorc.task_id,
            "pack_dir": rorc.pack_dir,
            "mode": rorc.backend,
            "backend": rorc.backend,
            "oracle_mode": rorc.oracle_mode,
            "source_track": rorc.source_track,
            "solution_reward": rorc.solution_reward,
            "null_reward": rorc.null_reward,
            "sol": rorc.solution_reward,
            "null": rorc.null_reward,
            "agent_isolated": rorc.isolation.clean,
            "isolation_status": "clean" if rorc.isolation.clean else "leak",
            "repository_url": rorc.pack_meta.repository_url,
            "base_commit_hash": rorc.pack_meta.base_commit_hash,
            "reason_codes": list(rorc.reason_codes),
            "reasons": list(rorc.reasons),
            "evidence_files": (
                rorc.evidence_files.to_dict() if rorc.evidence_files is not None else None
            ),
            "audit": rorc.audit,
            "evidence_out": str(evidence_path),
            "budget_usd": settings.budget_usd,
            "provider_calls": 0,
            "real_pr": True,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            status = "OK" if rorc.certified else "FAIL"
            typer.echo(
                f"deepswe-oracle: {status} real_pr certified={rorc.certified} "
                f"task_id={rorc.task_id} backend={rorc.backend} "
                f"sol={rorc.solution_reward} null={rorc.null_reward} "
                f"isolated={rorc.isolation.clean} "
                f"track={rorc.source_track!r}"
            )
            for reason in rorc.reasons[:12]:
                typer.echo(f"  reason: {reason}")
        raise typer.Exit(code=0 if rorc.certified else 1)

    try:
        result = certify_deepswe_pack(
            pack_dir,
            backend=mode,
            evidence_out=evidence_path,
            audit_out=audit_path,
            run_pier_hooks=not no_pier_hooks,
            dest_hint=out,
            run_id="deepswe-cli",
        )
    except FakeBackendRejected as exc:
        typer.secho(f"deepswe-oracle: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except DeepSWECertError as exc:
        typer.secho(f"deepswe-oracle: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"deepswe-oracle: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "ok": result.certified,
        "passed": result.certified,
        "certified": result.certified,
        "task_id": result.task_id,
        "pack_dir": result.pack_dir,
        "mode": result.backend,
        "backend": result.backend,
        "solution_reward": result.solution_reward,
        "null_reward": result.null_reward,
        "agent_isolated": result.isolation.clean,
        "isolation_status": "clean" if result.isolation.clean else "leak",
        "repository_url": result.pack_meta.repository_url,
        "base_commit_hash": result.pack_meta.base_commit_hash,
        "reason_codes": list(result.reason_codes),
        "reasons": list(result.reasons),
        "pier_ready": result.pier_ready.to_dict(),
        "audit": result.audit,
        "evidence_out": str(evidence_path),
        "budget_usd": settings.budget_usd,
        "provider_calls": 0,
        "real_pr": False,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "OK" if result.certified else "FAIL"
        typer.echo(
            f"deepswe-oracle: {status} certified={result.certified} "
            f"task_id={result.task_id} backend={result.backend} "
            f"solution={result.solution_reward} null={result.null_reward} "
            f"isolated={result.isolation.clean} "
            f"url={result.pack_meta.repository_url!r}"
        )
        for reason in result.reasons[:12]:
            typer.echo(f"  reason: {reason}")
    raise typer.Exit(code=0 if result.certified else 1)


@app.command("pier-cert")
def pier_cert_cmd(
    pack_dir: Annotated[
        Path | None,
        typer.Option(
            "--pack-dir",
            help="Harbor pack directory (tasks/<id>) to Pier-certify",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    jobs_dir: Annotated[
        Path,
        typer.Option(
            "--jobs-dir",
            "-o",
            help="Pier jobs root (must be under /tmp/harbor-deepswe-jobs*)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("/tmp/harbor-deepswe-jobs"),
    oracle_mode: Annotated[
        str,
        typer.Option(
            "--oracle-mode",
            help="Must be docker on cert path (fake refused for deepswe keeps)",
        ),
    ] = "docker",
    pier_bin: Annotated[
        Path | None,
        typer.Option(
            "--pier-bin",
            help="Path to pier executable (default: PIER_BIN or /tmp/pier-venv/bin/pier)",
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    evidence_out: Annotated[
        Path | None,
        typer.Option(
            "--evidence-out",
            help="Write pier_evidence.json (default: <jobs-dir>/pier_evidence.json)",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    audit_out: Annotated[
        Path | None,
        typer.Option(
            "--audit-out",
            help="Append pier_gate_audit.jsonl",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    skip_oracle: Annotated[
        bool,
        typer.Option("--skip-oracle", help="Skip pier -a oracle run"),
    ] = False,
    skip_null: Annotated[
        bool,
        typer.Option("--skip-null", help="Skip pier -a nop (null) run"),
    ] = False,
    no_load_smoke: Annotated[
        bool,
        typer.Option(
            "--no-load-smoke",
            help="Skip Harbor TaskConfig load smoke (still checks pack tree)",
        ),
    ] = False,
    n_concurrent: Annotated[
        int,
        typer.Option("--n-concurrent", "-n", help="Pier concurrent trials (cap 1–24)"),
    ] = 1,
    timeout_sec: Annotated[
        float,
        typer.Option("--timeout-sec", help="Pier job timeout seconds"),
    ] = 1800.0,
    real_pr: Annotated[
        bool,
        typer.Option(
            "--real-pr",
            help=(
                "Real-PR product Pier cert (VAL-RPIER-001..004): require "
                "source_track=real_pr, prefer live pier over scripted, "
                "write sol/null evidence under jobs, refuse fake"
            ),
        ),
    ] = False,
    allow_scripted_substitute: Annotated[
        bool,
        typer.Option(
            "--allow-scripted-substitute",
            help=(
                "With --real-pr: allow scripted pier rewards as offline full "
                "substitute when live pier is unavailable (unit/demo only; "
                "not default product smoke)"
            ),
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit Pier cert summary as JSON"),
    ] = False,
) -> None:
    """Pier cert: load pack, oracle reward=1, nop/null reward=0 (VAL-PIER-* / VAL-RPIER-*).

    Jobs land under ``/tmp/harbor-deepswe-jobs*``. Fake oracle_mode is refused.
    With ``--real-pr`` prefers live pier; scripted is not a full substitute
    without ``--allow-scripted-substitute``.
    """
    settings = load_settings()
    mode = (oracle_mode or "docker").strip().lower()

    from swe_factory.harbor.deepswe_cert import FakeBackendRejected
    from swe_factory.harbor.pier_cert import (
        PierCertError,
        certify_pier_pack,
        refuse_fake_oracle_mode,
    )
    from swe_factory.harbor.real_pier_cert import (
        RealPierCertError,
        RealPierFakeOracleRejected,
        RealPierUnavailableError,
        certify_real_pier_pack,
        refuse_fake_oracle_mode_real_pier,
    )

    try:
        if real_pr:
            refuse_fake_oracle_mode_real_pier(
                mode, certified=True, pack_or_dest=pack_dir or jobs_dir
            )
        else:
            refuse_fake_oracle_mode(mode, certified=True, pack_or_dest=pack_dir or jobs_dir)
    except (FakeBackendRejected, RealPierFakeOracleRejected) as exc:
        typer.secho(f"pier-cert: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if pack_dir is None:
        typer.secho(
            "pier-cert: require --pack-dir (Harbor pack tree for Pier load/oracle)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    evidence_path = evidence_out or (Path(jobs_dir) / "pier_evidence.json")

    if real_pr:
        try:
            rpier = certify_real_pier_pack(
                pack_dir,
                jobs_root=jobs_dir,
                oracle_mode=mode,
                run_oracle=not skip_oracle,
                run_null=not skip_null,
                run_load_smoke=not no_load_smoke,
                pier_bin=pier_bin,
                evidence_out=evidence_path,
                evidence_dir=Path(jobs_dir) / "evidence",
                audit_out=audit_out,
                n_concurrent=max(1, min(int(n_concurrent), 24)),
                timeout_sec=timeout_sec,
                require_real_pr_track=True,
                allow_scripted_substitute=allow_scripted_substitute,
                prefer_real_pier=True,
                dest_hint="datasets/deepswe_v1",
            )
        except (FakeBackendRejected, RealPierFakeOracleRejected) as exc:
            typer.secho(f"pier-cert: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
        except RealPierUnavailableError as exc:
            typer.secho(f"pier-cert: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        except (RealPierCertError, PierCertError) as exc:
            typer.secho(f"pier-cert: error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:  # noqa: BLE001
            typer.secho(f"pier-cert: error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

        sol = rpier.solution_reward
        null = rpier.null_reward
        sol_path = rpier.oracle_run.reward.path if rpier.oracle_run else None
        null_path = rpier.null_run.reward.path if rpier.null_run else None
        if rpier.evidence_files is not None:
            sol_path = rpier.evidence_files.sol_path or sol_path
            null_path = rpier.evidence_files.null_path or null_path
        payload = {
            "ok": rpier.certified,
            "passed": rpier.certified,
            "certified": rpier.certified,
            "task_id": rpier.task_id,
            "pack_dir": rpier.pack_dir,
            "jobs_root": rpier.jobs_root,
            "backend": rpier.backend,
            "oracle_mode": rpier.oracle_mode,
            "source_track": rpier.source_track,
            "structural_ok": rpier.structural_ok,
            "sol_reward": sol,
            "null_reward": null,
            "sol_reward_path": sol_path,
            "null_reward_path": null_path,
            "pier_path_class": rpier.pier_path_class,
            "agent_isolated": rpier.isolation.clean,
            "isolation_status": "clean" if rpier.isolation.clean else "leak",
            "repository_url": rpier.pack_meta.repository_url,
            "base_commit_hash": rpier.pack_meta.base_commit_hash,
            "reason_codes": list(rpier.reason_codes),
            "reasons": list(rpier.reasons),
            "evidence_files": (rpier.evidence_files.to_dict() if rpier.evidence_files else None),
            "evidence_out": str(evidence_path),
            "real_pr": True,
            "budget_usd": settings.budget_usd,
            "provider_calls": 0,
        }
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            status = "OK" if rpier.certified else "FAIL"
            typer.echo(
                f"pier-cert: {status} real_pr certified={rpier.certified} "
                f"task_id={rpier.task_id} mode={rpier.oracle_mode} "
                f"sol={sol} null={null} path={rpier.pier_path_class} "
                f"structural={rpier.structural_ok} "
                f"isolated={rpier.isolation.clean} jobs={rpier.jobs_root}"
            )
            for reason in rpier.reasons[:12]:
                typer.echo(f"  - {reason}")
        raise typer.Exit(code=0 if rpier.certified else 1)

    try:
        result = certify_pier_pack(
            pack_dir,
            jobs_root=jobs_dir,
            oracle_mode=mode,
            run_oracle=not skip_oracle,
            run_null=not skip_null,
            run_load_smoke=not no_load_smoke,
            pier_bin=pier_bin,
            evidence_out=evidence_path,
            audit_out=audit_out,
            n_concurrent=max(1, min(int(n_concurrent), 24)),
            timeout_sec=timeout_sec,
        )
    except FakeBackendRejected as exc:
        typer.secho(f"pier-cert: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except PierCertError as exc:
        typer.secho(f"pier-cert: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"pier-cert: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    sol = result.oracle_run.reward.reward if result.oracle_run else None
    null = result.null_run.reward.reward if result.null_run else None
    payload = {
        "ok": result.certified,
        "passed": result.certified,
        "certified": result.certified,
        "task_id": result.task_id,
        "pack_dir": result.pack_dir,
        "jobs_root": result.jobs_root,
        "backend": result.backend,
        "oracle_mode": result.oracle_mode,
        "structural_ok": result.structural_ok,
        "sol_reward": sol,
        "null_reward": null,
        "sol_reward_path": result.oracle_run.reward.path if result.oracle_run else None,
        "null_reward_path": result.null_run.reward.path if result.null_run else None,
        "agent_isolated": result.isolation.clean,
        "isolation_status": "clean" if result.isolation.clean else "leak",
        "repository_url": result.pack_meta.repository_url,
        "base_commit_hash": result.pack_meta.base_commit_hash,
        "reason_codes": list(result.reason_codes),
        "reasons": list(result.reasons),
        "pier_ready": result.pier_ready.to_dict(),
        "evidence_out": str(evidence_path),
        "real_pr": False,
        "budget_usd": settings.budget_usd,
        "provider_calls": 0,
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "OK" if result.certified else "FAIL"
        typer.echo(
            f"pier-cert: {status} certified={result.certified} "
            f"task_id={result.task_id} mode={result.oracle_mode} "
            f"sol={sol} null={null} structural={result.structural_ok} "
            f"isolated={result.isolation.clean} jobs={result.jobs_root}"
        )
        for reason in result.reasons[:12]:
            typer.echo(f"  reason: {reason}")
        if result.oracle_run and result.oracle_run.reward.path:
            typer.echo(f"  oracle_reward: {result.oracle_run.reward.path}")
        if result.null_run and result.null_run.reward.path:
            typer.echo(f"  null_reward: {result.null_run.reward.path}")
    raise typer.Exit(code=0 if result.certified else 1)


@app.command("archive-hybrid-deepswe")
def archive_hybrid_deepswe_cmd(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            help="Hybrid product path to archive (default datasets/deepswe_v1)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1"),
    archive: Annotated[
        Path,
        typer.Option(
            "--archive",
            help="Archive destination (default datasets/deepswe_v1_hybrid_archive)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1_hybrid_archive"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Recopy source even when archive already has pack_manifest/tasks",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit archive result as JSON"),
    ] = False,
) -> None:
    """Archive hybrid datasets/deepswe_v1 → deepswe_v1_hybrid_archive (idempotent).

    VAL-RPR-001 / VAL-RSHIP-001: copy the hybrid_curated motor corpus to the
    archive path before any real-PR product overwrite. Never deletes product
    packs here; product clear is reserved for the real ship step. Hybrid is
    labeled historical only and is not claimed as current certified product.
    """
    from swe_factory.pipeline.archive_hybrid import (
        ArchiveHybridError,
        archive_hybrid_deepswe,
    )

    try:
        result = archive_hybrid_deepswe(
            source_dir=source,
            archive_dir=archive,
            force_recopy=force,
        )
    except ArchiveHybridError as exc:
        typer.secho(
            f"archive-hybrid-deepswe: error: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(
            f"archive-hybrid-deepswe: error: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    payload = result.to_dict()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        status = "OK" if result.ok else "FAIL"
        typer.echo(
            f"archive-hybrid-deepswe: {status} action={result.action} "
            f"archive_packs={result.archive_pack_count} "
            f"source_packs={result.source_pack_count} "
            f"archive={result.archive_dir}"
        )
        typer.echo(f"  reason: {result.reason}")
        typer.echo(
            "  note: hybrid is historical archive only; not claimed as current "
            "product (real_pr ship is a later feature)"
        )
        if result.archive_readme_path:
            typer.echo(f"  archive_readme: {result.archive_readme_path}")
        if result.archive_report_path:
            typer.echo(f"  archive_report: {result.archive_report_path}")
    raise typer.Exit(code=0 if result.ok else 1)


@app.command("archive-seed5-deepswe")
def archive_seed5_deepswe_cmd(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            help="Prior product path to archive (default datasets/deepswe_v1)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1"),
    archive: Annotated[
        Path,
        typer.Option(
            "--archive",
            help="Seed5 archive destination (default datasets/deepswe_v1_seed5_archive)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1_seed5_archive"),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Recopy source even when seed5 archive already has pack_manifest/tasks",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit archive result as JSON"),
    ] = False,
) -> None:
    """Archive prior real_pr product seed → deepswe_v1_seed5_archive (idempotent).

    VAL-LSHIP-001: freeze the M13 honesty product (N≈5 real_pr) before M14
    live-mine overwrite of datasets/deepswe_v1. Hybrid archive remains separate
    under deepswe_v1_hybrid_archive and is never seed5. Does not clear product;
    product clear/overwrite requires dual-truth gate_audit (VAL-LSHIP-007).
    """
    from swe_factory.pipeline.archive_seed5 import (
        ArchiveSeed5Error,
        archive_seed5_deepswe,
    )

    try:
        result = archive_seed5_deepswe(
            source_dir=source,
            archive_dir=archive,
            force_recopy=force,
            require_real_pr=True,
        )
    except ArchiveSeed5Error as exc:
        typer.secho(
            f"archive-seed5-deepswe: error: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(
            f"archive-seed5-deepswe: error: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    payload = result.to_dict()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        status = "OK" if result.ok else "FAIL"
        typer.echo(
            f"archive-seed5-deepswe: {status} action={result.action} "
            f"archive_packs={result.archive_pack_count} "
            f"source_packs={result.source_pack_count} "
            f"archive={result.archive_dir}"
        )
        typer.echo(f"  reason: {result.reason}")
        typer.echo(
            "  note: seed5 is historical prior product only; not live product N "
            "(hybrid archive stays separate)"
        )
        if result.archive_readme_path:
            typer.echo(f"  archive_readme: {result.archive_readme_path}")
        if result.archive_report_path:
            typer.echo(f"  archive_report: {result.archive_report_path}")
    raise typer.Exit(code=0 if result.ok else 1)


@app.command("gate-audit-product")
def gate_audit_product_cmd(
    product: Annotated[
        Path,
        typer.Option(
            "--product",
            help="Product root with tasks/* (default datasets/deepswe_v1)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1"),
    materials: Annotated[
        list[Path] | None,
        typer.Option(
            "--materials",
            help="Optional materials roots for hunk/discovery meta (repeatable)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    min_accepted: Annotated[
        int | None,
        typer.Option(
            "--min-accepted",
            help="Minimum accepted dual-truth keeps (default: len(tasks/*))",
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable summary JSON"),
    ] = False,
) -> None:
    """Regenerate dual-truth gate_audit over FULL product tasks/* (VAL-LSHIP-007).

    Use after additive multilang promote when gate_audit lag behind tasks/*.
    Does **not** wipe packs; rewrites gate_audit + oracle_evidence + ship/pack
    summaries so accepted_count ties to certified product N.
    """
    from swe_factory.pipeline.gate_audit_product import (
        ProductGateAuditError,
        rebuild_product_dual_truth_from_tasks,
    )

    try:
        result = rebuild_product_dual_truth_from_tasks(
            product,
            materials_roots=materials,
            live_mine=True,
            min_accepted=min_accepted,
            require_all_accepted=True,
            write_oracle_evidence=True,
            write_pack_manifest=True,
            write_ship_summary=True,
            write_per_task_docker_oracle=True,
        )
    except ProductGateAuditError as exc:
        if json_out:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            typer.echo(f"gate-audit-product FAIL: {exc}")
        raise typer.Exit(code=1) from exc

    if json_out:
        # Drop bulky per-keep evidence when not needed for CLI scan.
        slim = {k: v for k, v in result.items() if k != "keeps"}
        slim["keep_count"] = len(result.get("keeps") or [])
        typer.echo(json.dumps(slim, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(
            f"gate-audit-product {'OK' if result.get('ok') else 'FAIL'}: "
            f"accepted={result.get('accepted_count')}/{result.get('intended_count')} "
            f"tasks={result.get('task_count')}"
        )
        typer.echo(f"  product: {result.get('product_dir')}")
        typer.echo(f"  gate_audit: {result.get('gate_audit_path')}")
        typer.echo(f"  oracle_evidence: {result.get('oracle_evidence_path')}")
        typer.echo(f"  reason: {result.get('reason')}")
        missing_add = [
            tid
            for tid in ("realpr-qs-487", "realpr-qs-488", "realpr-bitflags-483")
            if tid not in (result.get("accepted_ids") or [])
            and tid in (result.get("task_ids") or [])
        ]
        if missing_add:
            typer.echo(f"  warning: additive tasks not in accepted_ids: {missing_add}")
    raise typer.Exit(code=0 if result.get("ok") else 1)


@app.command("ship-deepswe")
def ship_deepswe_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Product ship root (default datasets/deepswe_v1)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/deepswe_v1"),
    work: Annotated[
        Path | None,
        typer.Option(
            "--work",
            help="Scratch workspace root (default: under parent of --out)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help=(
                "Product track: real_pr (default, refuse hybrid) or "
                "hybrid_curated (historical M10 motor path only — not product rebaseline)"
            ),
        ),
    ] = "real_pr",
    target: Annotated[
        int,
        typer.Option(
            "--target",
            help="Target certified pack count (Real-PR default 5; historical hybrid 113)",
        ),
    ] = 5,
    min_packs: Annotated[
        int,
        typer.Option(
            "--min-packs",
            help="Minimum certified packs for ship OK (Real-PR ≥5)",
        ),
    ] = 5,
    max_packs: Annotated[
        int,
        typer.Option("--max-packs", help="Maximum certified packs to keep this wave"),
    ] = 20,
    oracle: Annotated[
        str,
        typer.Option(
            "--oracle",
            help="Oracle backend (docker only on deepswe product path; fake refused)",
        ),
    ] = "docker",
    panel: Annotated[
        str,
        typer.Option(
            "--panel",
            help="Panel mode: offline (scripted matrix), live (OpenRouter), skip",
        ),
    ] = "offline",
    pier: Annotated[
        str,
        typer.Option(
            "--pier",
            help="Pier cert mode: scripted (offline rewards) or live (pier binary)",
        ),
    ] = "scripted",
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Optional language filter (hybrid path only): python,go,typescript",
        ),
    ] = None,
    materials: Annotated[
        Path | None,
        typer.Option(
            "--materials",
            help=(
                "Real-PR materials root. Product dest deepswe_v1 with --live-mine "
                "defaults to datasets/live_materials (live bridge). Engineering/unit "
                "dests may use fixtures/real_pr_ship. Product dest refuses silent "
                "fixture default (VAL-LMAT-003 / VAL-LSHIP-003)."
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    live_mine: Annotated[
        bool,
        typer.Option(
            "--live-mine/--no-live-mine",
            help=(
                "Live-mine product path: require live materials bridge root "
                "(default datasets/live_materials); refuse fixtures/real_pr_ship "
                "default and empty live yield pad (VAL-LMAT-*/VAL-LSHIP-006)"
            ),
        ),
    ] = False,
    pier_jobs: Annotated[
        Path,
        typer.Option(
            "--pier-jobs",
            help="Pier jobs root under /tmp/harbor-deepswe-jobs*",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("/tmp/harbor-deepswe-jobs-ship"),
    no_archive: Annotated[
        bool,
        typer.Option(
            "--no-archive",
            help="Skip hybrid archive check (still refuses hybrid product promote)",
        ),
    ] = False,
    hybrid_bind: Annotated[
        bool,
        typer.Option(
            "--hybrid-bind/--no-hybrid-bind",
            help="Force hybrid motor bind (always refused on real_pr product)",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit ship summary as JSON"),
    ] = False,
) -> None:
    """Ship Real-PR certified packs to datasets/deepswe_v1 (product rebaseline).

    Default product path (VAL-RSHIP / VAL-RX / VAL-LSHIP): archive hybrid →
    export real_pr packs with clone@SHA agent trees → dual-run labels → docker
    oracle sol=1/null=0 → pier evidence → promote certified packs.
    PROVENANCE/report count real_pr only. Hybrid motors and oracle_mode=fake
    are refused on the product path.

    Live mine (M14): pass ``--live-mine`` with live materials (default
    ``datasets/live_materials`` from materialize-from-pr). Product dest refuses
    silent ``fixtures/real_pr_ship`` default and empty live yield fixture pad
    (VAL-LMAT-003, VAL-LSHIP-003/004/006). Offline unit dests may still use
    fixtures.

    Historical hybrid M10 (113 motors) remains available via
    ``--source hybrid_curated --target 113 --min-packs 113`` for regression only;
    it is never claimed as the Real-PR product rebaseline.
    """
    settings = load_settings()
    mode = (oracle or "docker").strip().lower()
    if mode != "docker":
        typer.secho(
            f"ship-deepswe: refuses --oracle {oracle!r}; docker only "
            "(VAL-RSHIP-005 / VAL-ORCD-004)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    panel_mode = (panel or "offline").strip().lower()
    if panel_mode not in {"offline", "live", "skip"}:
        typer.secho(
            f"ship-deepswe: unknown --panel {panel!r} (offline|live|skip)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    pier_mode = (pier or "scripted").strip().lower()
    if pier_mode not in {"scripted", "live"}:
        typer.secho(
            f"ship-deepswe: unknown --pier {pier!r} (scripted|live)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    source_track = (source or "real_pr").strip().lower()
    if source_track in {"real", "real-pr", "realpr"}:
        source_track = "real_pr"
    if source_track in {"hybrid", "motor", "hybrid-curated"}:
        source_track = "hybrid_curated"

    langs: list[str] | None = None
    if language:
        langs = [p.strip() for p in language.split(",") if p.strip()]

    from swe_factory.harbor.deepswe_cert import FakeBackendRejected
    from swe_factory.pipeline.ship_deepswe import ShipDeepSWEError, run_ship_deepswe
    from swe_factory.pipeline.ship_real_pr import (
        HybridProductPromoteRejected,
        ProductEmptyLiveYieldRejected,
        ProductFixtureMaterialsRejected,
        ShipRealPrError,
        run_ship_deepswe_real_pr,
    )

    if source_track == "real_pr" and hybrid_bind:
        typer.secho(
            "ship-deepswe: refuse --hybrid-bind on real_pr product path "
            "(VAL-RSHIP-005 / VAL-RX-003)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        if source_track == "real_pr":
            result = run_ship_deepswe_real_pr(
                out_dir=out,
                work_root=work,
                target_packs=target,
                min_packs=min_packs,
                max_packs=max_packs,
                oracle_mode=mode,
                panel_mode=panel_mode,  # type: ignore[arg-type]
                pier_mode=pier_mode,  # type: ignore[arg-type]
                settings=settings,
                materials_root=materials,
                pier_jobs_root=pier_jobs,
                ensure_archive=not no_archive,
                hybrid_bind=False,
                source_track="real_pr",
                allow_scripted_pier_substitute=pier_mode == "scripted",
                live_mine=live_mine,
            )
        elif source_track == "hybrid_curated":
            # Historical M10 path — not product Real-PR rebaseline.
            result = run_ship_deepswe(
                out_dir=out,
                work_root=work,
                target_packs=target,
                min_packs=min_packs,
                max_packs=max_packs,
                oracle_mode=mode,
                panel_mode=panel_mode,  # type: ignore[arg-type]
                pier_mode=pier_mode,  # type: ignore[arg-type]
                languages=langs,
                settings=settings,
                pier_jobs_root=pier_jobs,
            )
        else:
            typer.secho(
                f"ship-deepswe: unknown --source {source!r} (real_pr|hybrid_curated)",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
    except (
        FakeBackendRejected,
        HybridProductPromoteRejected,
        ProductFixtureMaterialsRejected,
        ProductEmptyLiveYieldRejected,
    ) as exc:
        typer.secho(f"ship-deepswe: refuse: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except (ShipDeepSWEError, ShipRealPrError) as exc:
        typer.secho(f"ship-deepswe: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"ship-deepswe: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = result.to_dict()
    payload["source_track"] = source_track
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        status = "OK" if result.ok else "FAIL"
        typer.echo(
            f"ship-deepswe: {status} source={source_track} "
            f"certified={result.certified_count} "
            f"langs={result.languages} under_cap={result.under_cap} "
            f"smoke={result.harbor_load_smoke.get('ok')} out={result.out_dir}"
        )
        typer.echo(f"  reason: {result.reason}")
        if result.report_path:
            typer.echo(f"  report: {result.report_path}")
        if result.provenance_path:
            typer.echo(f"  provenance: {result.provenance_path}")
        if result.pack_manifest_path:
            typer.echo(f"  pack_manifest: {result.pack_manifest_path}")
        if result.ledger_summary_path:
            typer.echo(f"  ledger_summary: {result.ledger_summary_path}")
        if result.e2e_drip_path:
            typer.echo(f"  e2e_drip: {result.e2e_drip_path}")
    raise typer.Exit(code=0 if result.ok else 1)


@app.command("ship-harbor")
def ship_harbor_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Ship root for tasks/, report.md, pack_manifest.json, ledger_summary.json",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = Path("datasets/harbor_v1"),
    work: Annotated[
        Path | None,
        typer.Option(
            "--work",
            help="Scratch workspace root (default: under parent of --out)",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    target: Annotated[
        int,
        typer.Option("--target", help="Target certified pack count (band 10–15)"),
    ] = 12,
    min_packs: Annotated[
        int,
        typer.Option("--min-packs", help="Minimum certified packs for ship OK"),
    ] = 10,
    max_packs: Annotated[
        int,
        typer.Option("--max-packs", help="Maximum certified packs to keep"),
    ] = 15,
    oracle: Annotated[
        str,
        typer.Option(
            "--oracle",
            help="Oracle backend: fake (offline separate-verifier) or docker",
        ),
    ] = "fake",
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            "-l",
            help="Optional language filter: python,go,typescript (comma list)",
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit ship summary as JSON"),
    ] = False,
) -> None:
    """Ship 10–15 DeepSWE-complete Harbor packs under project budget (≤$600).

    VAL-HARBOR-008/009/010 + VAL-CROSS-007: multi-lang hard packs with oracle
    solution=1 / null=0 evidence, Harbor structural load smoke, report +
    pack_manifest + ledger_summary. No gate relaxation.
    """
    settings = load_settings()
    mode = (oracle or "fake").strip().lower()
    if mode not in {"fake", "docker"}:
        typer.secho(
            f"ship-harbor: unknown --oracle {oracle!r} (fake|docker)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    langs: list[str] | None = None
    if language:
        langs = [p.strip() for p in language.split(",") if p.strip()]

    from swe_factory.pipeline.ship_harbor import ShipHarborError, run_ship_harbor

    try:
        result = run_ship_harbor(
            out_dir=out,
            work_root=work,
            target_packs=target,
            min_packs=min_packs,
            max_packs=max_packs,
            oracle_mode=mode,  # type: ignore[arg-type]
            languages=langs,
            settings=settings,
        )
    except ShipHarborError as exc:
        typer.secho(f"ship-harbor: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"ship-harbor: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = result.to_dict()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        status = "OK" if result.ok else "FAIL"
        typer.echo(
            f"ship-harbor: {status} certified={result.certified_count} "
            f"langs={result.languages} under_cap={result.under_cap} "
            f"smoke={result.harbor_load_smoke.get('ok')} out={result.out_dir}"
        )
        typer.echo(f"  reason: {result.reason}")
        if result.report_path:
            typer.echo(f"  report: {result.report_path}")
        if result.pack_manifest_path:
            typer.echo(f"  pack_manifest: {result.pack_manifest_path}")
        if result.ledger_summary_path:
            typer.echo(f"  ledger_summary: {result.ledger_summary_path}")
    raise typer.Exit(code=0 if result.ok else 1)


@app.command("offline-fixture")
def offline_fixture_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Output directory for gate-demo artifact (tasks.jsonl, workspace)",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("datasets/fixture_demo"),
    fixture_root: Annotated[
        Path | None,
        typer.Option(
            "--fixture-root",
            help="Override path to fixtures/tiny_offline",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Run offline fixture pipeline (schema + stub gates, no OpenRouter/LLM).

    Produces a gate-demo task artifact proving package wiring offline
    (VAL-CROSS-001). Never makes provider calls.
    """
    # Lazy import keeps CLI help fast and avoids circulars during skeleton boot.
    from swe_factory.fixture.offline import OfflineFixtureError, run_offline_fixture_pipeline

    try:
        result = run_offline_fixture_pipeline(
            out_dir=out,
            fixture_root=fixture_root,
        )
    except OfflineFixtureError as exc:
        typer.secho(f"offline-fixture: error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        "offline-fixture: OK "
        f"instance_id={result.instance_id} "
        f"passed={result.gates.passed} "
        f"provider_calls={result.provider_calls} "
        f"out={result.out_dir}"
    )
    if result.tasks_jsonl is not None:
        typer.echo(f"  tasks_jsonl={result.tasks_jsonl}")
    if result.gate_audit is not None:
        typer.echo(f"  gate_audit={result.gate_audit}")


def run() -> None:
    """Console-script compatible runner."""
    app()


if __name__ == "__main__":
    run()
