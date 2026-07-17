"""DeepAgent primary product CLI (M16).

Primary console entry ``deepagent`` — thin Typer surface that wraps existing
``swe_factory`` implementations without forking honesty logic:

- generate → ship-deepagent / live-mine real_pr path
- upload / pull → HF pack I/O (local validate + hub hooks)
- eval → eval_deepagent Pier mini-swe serial (hard-stop $600)
- oracle → HarborDockerVerifier cert path (sol=1 / null=0)

Compat entry ``swe-factory`` remains on ``swe_factory.cli:app``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import typer

from swe_factory import __version__
from swe_factory.panel.eval_deepagent import (
    DEFAULT_N_CONCURRENT as DEFAULT_EVAL_N_CONCURRENT,
)
from swe_factory.panel.eval_deepagent import (
    MAX_N_CONCURRENT as MAX_EVAL_N_CONCURRENT,
)

# M16 product defaults (architecture.md / AGENTS.md).
DEFAULT_HF_REPO_ID = "BaseIntelligence/deepagent"
DEFAULT_HF_REVISION = "test"
DEFAULT_GENERATE_OUT = Path("datasets/test_n10")
DEFAULT_GENERATE_TARGET = 10
DEFAULT_EVAL_HARD_STOP_USD = 600.0
# M19 concurrent-bench cap aliases DEFAULT_N_CONCURRENT / MAX_N_CONCURRENT from
# eval_deepagent (shared refuse path with swe-factory eval-deepagent wrapper).
DEFAULT_PRODUCT_ROOT = Path("datasets/deepagent_v1")

app = typer.Typer(
    name="deepagent",
    help=(
        "DeepAgent — hard, Docker-verifiable SWE benchmark packs.\n\n"
        "Primary product commands:\n"
        f"  generate  — live-mine / ship-deepagent path → local pack root "
        f"(default {DEFAULT_GENERATE_OUT}, --target {DEFAULT_GENERATE_TARGET})\n"
        "  refresh-instructions — rewrite instruction.md to full DeepSWE-style "
        "(instruction-only; no dual-run/oracle re-cert)\n"
        f"  upload    — push pack trees + pack_manifest to Hugging Face "
        f"dataset {DEFAULT_HF_REPO_ID} (revision default {DEFAULT_HF_REVISION})\n"
        f"  pull      — download pack trees from {DEFAULT_HF_REPO_ID} "
        f"(revision main|test)\n"
        f"  eval      — Pier + mini-swe-agent + HarborDocker model eval "
        f"(n_concurrent default {DEFAULT_EVAL_N_CONCURRENT}, max {MAX_EVAL_N_CONCURRENT}; "
        f"hard-stop-usd={int(DEFAULT_EVAL_HARD_STOP_USD)}; "
        f"fidelity=pier_miniswe_harbor; n>1 raises host Mem risk)\n"
        "  oracle    — HarborDocker dual-truth cert (sol=1 / null=0; refuse fake)\n"
        "  curate-hardness — hardness curate (floors + align + intrinsic; "
        "model success is scoreboard-only, M25)\n"
        "  version   — package version identity\n\n"
        "Compatibility: historical factory stages remain on `swe-factory` "
        "(ship-deepagent, real-pr-pool, ledger, eval-deepagent, …). "
        "Secrets (HF_TOKEN / OPENROUTER / GITHUB_TOKEN) load from env / .env only "
        "— never embedded in defaults or help examples."
    ),
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """DeepAgent product CLI."""


@app.command("version")
def version_cmd() -> None:
    """Print package version identity."""
    typer.echo(__version__)


@app.command("generate")
def generate_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help=(
                "Output pack root (M16 live-mine wave default datasets/test_n10; "
                "product corpus may use datasets/deepagent_v1)"
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = DEFAULT_GENERATE_OUT,
    target: Annotated[
        int,
        typer.Option(
            "--target",
            help="Target certified pack count for the live-mine generate wave (M16=10)",
        ),
    ] = DEFAULT_GENERATE_TARGET,
    min_packs: Annotated[
        int,
        typer.Option(
            "--min-packs",
            help="Minimum certified packs for generate OK (M16 bar ≥5)",
        ),
    ] = 5,
    max_packs: Annotated[
        int,
        typer.Option("--max-packs", help="Maximum certified packs to keep this wave"),
    ] = 20,
    materials: Annotated[
        Path | None,
        typer.Option(
            "--materials",
            help=(
                "Live-mine materials root (default datasets/live_materials when "
                "--live-mine). fixtures/real_pr_ship is engineering-only, never product N"
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = None,
    live_mine: Annotated[
        bool,
        typer.Option(
            "--live-mine/--no-live-mine",
            help=(
                "Live-mine product path: require live materials bridge; refuse "
                "fixtures/real_pr_ship pad and empty live yield pad"
            ),
        ),
    ] = True,
    oracle: Annotated[
        str,
        typer.Option(
            "--oracle",
            help="Oracle backend (docker only on product path; HarborDockerVerifier)",
        ),
    ] = "docker",
    panel: Annotated[
        str,
        typer.Option(
            "--panel",
            help="Panel mode: offline (scripted), live (OpenRouter), skip",
        ),
    ] = "offline",
    pier: Annotated[
        str,
        typer.Option(
            "--pier",
            help="Pier cert mode: scripted (offline rewards) or live (pier binary)",
        ),
    ] = "scripted",
    work: Annotated[
        Path | None,
        typer.Option(
            "--work",
            help="Scratch workspace root (default: under parent of --out)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = None,
    source: Annotated[
        str,
        typer.Option(
            "--source",
            help="Product track: real_pr (default live mine); hybrid refused as product",
        ),
    ] = "real_pr",
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit ship summary as JSON"),
    ] = False,
) -> None:
    """Live mine / ship-deepagent path → local Harbor pack root.

    M16 wave: ``deepagent generate --target 10 --out datasets/test_n10 --live-mine``.
    Wraps the same honesty pipeline as ``swe-factory ship-deepagent`` (real_pr,
    HarborDocker sol=1/null=0, no fixture pad).
    """
    from swe_factory.cli import ship_deepagent_cmd

    # Delegate to existing ship-deepagent implementation (no honesty fork).
    ship_deepagent_cmd(
        out=out,
        work=work,
        source=source,
        target=target,
        min_packs=min_packs,
        max_packs=max_packs,
        oracle=oracle,
        panel=panel,
        pier=pier,
        language=None,
        materials=materials,
        live_mine=live_mine,
        pier_jobs=Path("/tmp/harbor-deepagent-jobs-ship"),
        no_archive=False,
        hybrid_bind=False,
        json_out=json_out,
    )


@app.command("refresh-instructions")
def refresh_instructions_cmd(
    root: Annotated[
        Path,
        typer.Option(
            "--root",
            help=(
                "Certified product root with tasks/<id>/instruction.md (default datasets/test_n10)"
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = DEFAULT_GENERATE_OUT,
    materials: Annotated[
        Path | None,
        typer.Option(
            "--materials",
            help=(
                "Live materials root with meta.json/solution.patch per task_id "
                "(default datasets/live_materials when present)"
            ),
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = None,
    fetch_github: Annotated[
        bool,
        typer.Option(
            "--fetch-github/--no-fetch-github",
            help=(
                "When materials lack PR body, re-fetch from GitHub REST "
                "(requires GITHUB_TOKEN / gh auth)"
            ),
        ),
    ] = True,
    stamp_manifest: Annotated[
        bool,
        typer.Option(
            "--stamp-manifest/--no-stamp-manifest",
            help="Note prompt_style=deepagent_full_v1 on pack_manifest.json",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Build/validate full prompts without writing instruction.md",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit refresh summary as JSON"),
    ] = False,
) -> None:
    """Rewrite pack instruction.md to full DeepSWE-style prompts (VAL-DPRMPT-004).

    Instruction-only re-export: loads materials/PR meta (or re-fetches body via
    GitHub), rewrites instruction.md in place, gold-leak scans each pack.
    Does **not** re-run dual-run/oracle when dual-truth evidence is already valid.
    Never fixture-pads product trees.
    """
    from swe_factory.pipeline.refresh_instructions import (
        RefreshInstructionsError,
        refresh_product_instructions,
    )

    materials_root = materials
    if materials_root is None:
        default_mats = Path("datasets/live_materials")
        if default_mats.is_dir():
            materials_root = default_mats

    try:
        result = refresh_product_instructions(
            root,
            materials_root=materials_root,
            fetch_github=fetch_github,
            dry_run=dry_run,
            stamp_manifest=stamp_manifest,
        )
    except RefreshInstructionsError as exc:
        typer.echo(f"refresh-instructions: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(result.message)
        for pack in result.packs:
            status = "ok" if pack.ok else "FAIL"
            typer.echo(
                f"  [{status}] {pack.task_id}: "
                f"{pack.chars_before}→{pack.chars_after} chars "
                f"body={pack.body_chars} ({pack.body_source})"
                + (f" err={pack.error}" if pack.error else "")
            )
    if not result.ok:
        raise typer.Exit(code=1)


def _resolve_hf_token() -> str | None:
    """Load HF token from env only (never hardcode; never print)."""
    import os

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def _safe_hub_cli_message(action: str, exc: BaseException) -> str:
    """Map Hub/CLI failures to constant auth-safe text (no raw str(exc)).

    Prefer messages already produced by :mod:`swe_factory.export.hf_packs`
    (``HfPacksError`` carries constants). For other exception types
    (fallback HfApi path), classify via ``map_hub_failure``.
    """
    try:
        from swe_factory.export.hf_packs import (
            MSG_PULL_AUTH,
            MSG_PULL_HUB,
            MSG_PULL_REVISION,
            MSG_TOKEN_MISSING,
            MSG_UPLOAD_AUTH,
            MSG_UPLOAD_HUB,
            MSG_UPLOAD_REPO,
            HfPacksError,
            map_hub_failure,
        )
    except ImportError:
        # Absolute last resort when export package is missing — still no raw Hub text.
        if action == "upload":
            return "upload: Hugging Face Hub operation failed"
        return "pull: Hugging Face Hub operation failed"

    # Schema / self-contained HfPacksError messages are already safe constants
    # or intentionally product-owned (never Hub HTTP bodies for auth paths).
    if isinstance(exc, HfPacksError):
        text = str(exc)
        # Known constants always pass through; unknown leftovers stay as-owned.
        known = {
            MSG_TOKEN_MISSING,
            MSG_UPLOAD_AUTH,
            MSG_UPLOAD_HUB,
            MSG_UPLOAD_REPO,
            MSG_PULL_AUTH,
            MSG_PULL_HUB,
            MSG_PULL_REVISION,
        }
        if text in known or text.startswith(("upload: pack ", "pull: no ", "pull: revision")):
            return text
        # Unexpected HfPacksError body → re-map without leaking
        return map_hub_failure(action, exc)

    return map_hub_failure(action, exc)


def _pack_schema_ok(src: Path) -> tuple[bool, list[str]]:
    """Validate Harbor pack tree layout before any HF push (offline-safe).

    Prefers full product schema via ``swe_factory.export.hf_packs`` when
    available; falls back to a minimal required-file set otherwise.
    """
    try:
        from swe_factory.export.hf_packs import validate_pack_corpus

        result = validate_pack_corpus(src)
        return result.ok, list(result.reasons)
    except ImportError:
        pass

    reasons: list[str] = []
    if not src.is_dir():
        return False, [f"source root missing: {src}"]
    tasks_dir = src / "tasks"
    if not tasks_dir.is_dir():
        # Allow flat single-pack roots that look like tasks/<id>
        if (src / "task.toml").is_file():
            pack_roots = [src]
        else:
            return False, [f"no tasks/ under {src} and no task.toml at root"]
    else:
        pack_roots = [p for p in sorted(tasks_dir.iterdir()) if p.is_dir()]
        if not pack_roots:
            return False, [f"tasks/ under {src} is empty"]

    required_files = (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "tests/test.sh",
        "solution/solution.patch",
    )
    for pack in pack_roots:
        for rel in required_files:
            if not (pack / rel).is_file():
                reasons.append(f"missing {rel} in {pack.name}")
        for dname in ("environment", "tests", "solution"):
            if not (pack / dname).is_dir():
                reasons.append(f"missing {dname}/ in {pack.name}")
    return (not reasons), reasons


@app.command("upload")
def upload_cmd(
    src: Annotated[
        Path,
        typer.Option(
            "--src",
            "--source",
            help="Local pack root to push (e.g. datasets/test_n10 or datasets/deepagent_v1)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/test_n10"),
    repo_id: Annotated[
        str,
        typer.Option(
            "--repo-id",
            help=f"Hugging Face dataset id (default {DEFAULT_HF_REPO_ID})",
        ),
    ] = DEFAULT_HF_REPO_ID,
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "--branch",
            help=(
                "HF revision/branch to push (M16 live automation uses test; main is the stable pin)"
            ),
        ),
    ] = DEFAULT_HF_REVISION,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Validate local pack schema only; do not contact Hugging Face",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit upload summary as JSON"),
    ] = False,
) -> None:
    """Push pack trees + pack_manifest to Hugging Face dataset BaseIntelligence/deepagent.

    Auth: ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` from the environment / .env only.
    Prefer revision ``test`` for automated M16 writes. Never prints tokens.
    """
    # Prefer hf_packs for the full path (schema + dry-run + live push).
    _hf_packs: ModuleType | None
    try:
        _hf_packs = importlib.import_module("swe_factory.export.hf_packs")
    except ImportError:
        _hf_packs = None

    if _hf_packs is not None and hasattr(_hf_packs, "upload_pack_tree"):
        token = _resolve_hf_token() if not dry_run else None
        if not dry_run and not token:
            typer.secho(
                "upload: HF_TOKEN / HUGGING_FACE_HUB_TOKEN missing (fail-closed; no network spam)",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        try:
            result = _hf_packs.upload_pack_tree(
                src=src,
                repo_id=repo_id,
                revision=revision,
                token=token,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            # Auth-safe constants only — never raw Hub str(exc).
            typer.secho(_safe_hub_cli_message("upload", exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        payload: dict[str, Any] = {
            "ok": True,
            "action": "upload",
            "src": str(src),
            "repo_id": repo_id,
            "revision": revision,
            "dry_run": dry_run,
            "schema_ok": True,
            "pushed": False,
        }
        if isinstance(result, dict):
            payload.update({k: v for k, v in result.items() if k.lower() not in {"token"}})
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        elif dry_run:
            typer.echo(
                f"deepagent upload: dry-run schema OK src={src} "
                f"repo_id={repo_id} revision={revision}"
            )
        else:
            typer.echo(
                f"deepagent upload: ok src={src} repo_id={repo_id} "
                f"revision={revision} pushed={payload.get('pushed')} "
                f"packs={payload.get('pack_count', '?')}"
            )
        raise typer.Exit(code=0)

    # Fallback scaffolding when hf_packs is unavailable.
    ok, reasons = _pack_schema_ok(src)
    if not ok:
        msg = "upload: pack schema invalid: " + "; ".join(reasons[:12])
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    payload = {
        "ok": True,
        "action": "upload",
        "src": str(src),
        "repo_id": repo_id,
        "revision": revision,
        "dry_run": dry_run,
        "schema_ok": True,
        "pushed": False,
    }

    if dry_run:
        payload["message"] = "schema OK; dry-run (no HF push)"
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(
                f"deepagent upload: dry-run schema OK src={src} "
                f"repo_id={repo_id} revision={revision}"
            )
        raise typer.Exit(code=0)

    token = _resolve_hf_token()
    if not token:
        typer.secho(
            "upload: HF_TOKEN / HUGGING_FACE_HUB_TOKEN missing (fail-closed; no network spam)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        typer.secho(
            "upload: huggingface_hub not installed; "
            "pip install 'huggingface_hub>=0.23' or --dry-run for schema only",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    try:
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
        payload["pushed"] = True
        payload["message"] = "upload_folder complete"
    except Exception as exc:  # noqa: BLE001
        typer.secho(_safe_hub_cli_message("upload", exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(
            f"deepagent upload: ok src={src} repo_id={repo_id} "
            f"revision={revision} pushed={payload.get('pushed')}"
        )


@app.command("pull")
def pull_cmd(
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Local directory to materialize pack trees into",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/hf_pull_test"),
    repo_id: Annotated[
        str,
        typer.Option(
            "--repo-id",
            help=f"Hugging Face dataset id (default {DEFAULT_HF_REPO_ID})",
        ),
    ] = DEFAULT_HF_REPO_ID,
    revision: Annotated[
        str,
        typer.Option(
            "--revision",
            "--branch",
            help="HF revision/branch to download (main | test)",
        ),
    ] = DEFAULT_HF_REVISION,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Print planned pull targets only; do not contact Hugging Face",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit pull summary as JSON"),
    ] = False,
) -> None:
    """Download pack trees from Hugging Face dataset BaseIntelligence/deepagent.

    Revision/branch selectable as ``main`` (stable pin) or ``test`` (M16 wave).
    Auth via env only; never prints tokens.
    """
    rev = (revision or DEFAULT_HF_REVISION).strip()
    if not rev:
        typer.secho(
            "pull: revision/branch required (documented choices: main | test)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    payload: dict[str, Any] = {
        "ok": True,
        "action": "pull",
        "out": str(out),
        "repo_id": repo_id,
        "revision": rev,
        "dry_run": dry_run,
        "pulled": False,
    }

    _hf_packs: ModuleType | None
    try:
        _hf_packs = importlib.import_module("swe_factory.export.hf_packs")
    except ImportError:
        _hf_packs = None

    if _hf_packs is not None and hasattr(_hf_packs, "pull_pack_tree"):
        token = _resolve_hf_token() if not dry_run else None
        try:
            result = _hf_packs.pull_pack_tree(
                out=out,
                repo_id=repo_id,
                revision=rev,
                token=token,
                dry_run=dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            typer.secho(_safe_hub_cli_message("pull", exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        if isinstance(result, dict):
            payload.update({k: v for k, v in result.items() if k.lower() not in {"token"}})
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        elif dry_run:
            typer.echo(f"deepagent pull: dry-run repo_id={repo_id} revision={rev} out={out}")
        else:
            typer.echo(
                f"deepagent pull: ok repo_id={repo_id} revision={rev} "
                f"out={out} pulled={payload.get('pulled')} "
                f"packs={payload.get('pack_count', '?')}"
            )
        raise typer.Exit(code=0)

    if dry_run:
        payload["message"] = f"would download {repo_id}@{rev} → {out} (main|test selectable)"
        if json_out:
            typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            typer.echo(f"deepagent pull: dry-run repo_id={repo_id} revision={rev} out={out}")
        raise typer.Exit(code=0)

    token = _resolve_hf_token()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        typer.secho(
            "pull: huggingface_hub not installed; pip install 'huggingface_hub>=0.23' or --dry-run",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    try:
        Path(out).mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=rev,
            local_dir=str(out),
            token=token,
        )
        payload["pulled"] = True
        payload["path"] = str(path)
        payload["message"] = "snapshot_download complete"
    except Exception as exc:  # noqa: BLE001
        typer.secho(_safe_hub_cli_message("pull", exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(
            f"deepagent pull: ok repo_id={repo_id} revision={rev} "
            f"out={out} pulled={payload.get('pulled')}"
        )


@app.command("eval")
def eval_cmd(
    product_root: Annotated[
        Path,
        typer.Option(
            "--product-root",
            help="Product Harbor root with tasks/* (datasets/test_n10 or deepagent_v1)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = DEFAULT_PRODUCT_ROOT,
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Report output directory (report.json + ledger_summary.json)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/panel_deepagent_eval"),
    max_packs: Annotated[
        int,
        typer.Option("--max-packs", help="Max product packs to score"),
    ] = 5,
    k: Annotated[
        int,
        typer.Option("--k", help="Trials per model per pack (pass@k; default k=1)"),
    ] = 1,
    n_concurrent: Annotated[
        int,
        typer.Option(
            "--n-concurrent",
            help=(
                f"Pier/docker concurrency in 1..{MAX_EVAL_N_CONCURRENT} "
                f"(default {DEFAULT_EVAL_N_CONCURRENT}; refuse outside range; "
                "n>1 raises host Mem / concurrent docker risk)"
            ),
        ),
    ] = DEFAULT_EVAL_N_CONCURRENT,
    hard_stop_usd: Annotated[
        float,
        typer.Option(
            "--hard-stop-usd",
            help="Ledger hard spend stop for this eval wave (M16 project default $600)",
        ),
    ] = DEFAULT_EVAL_HARD_STOP_USD,
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
            help="Pier job workdir root under /tmp/harbor-deepagent-jobs*",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("/tmp/harbor-deepagent-jobs-eval"),
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
            help="Restrict to pack id(s) (repeatable)",
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
    """Pier mini-swe + HarborDocker model eval (fidelity=pier_miniswe_harbor).

    n_concurrent accepted in 1..5 (default 1; refuse outside; n>1 has host Mem risk).
    M16 hard-stop-usd default is 600. Models: x-ai/grok-4.5 + moonshotai/kimi-k2.6.
    Wraps eval_deepagent core.
    """
    from swe_factory.cli import eval_deepagent_cmd

    n_conc = int(n_concurrent)
    if n_conc < 1 or n_conc > MAX_EVAL_N_CONCURRENT:
        typer.secho(
            f"eval: refuse n_concurrent={n_concurrent} "
            f"(must be in 1..{MAX_EVAL_N_CONCURRENT}; default "
            f"{DEFAULT_EVAL_N_CONCURRENT}; n>1 raises host Mem / concurrent docker risk)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    # Delegate after refuse so help shows 600 default while core handler runs.
    eval_deepagent_cmd(
        product_root=product_root,
        out=out,
        max_packs=max_packs,
        k=k,
        n_concurrent=n_concurrent,
        hard_stop_usd=hard_stop_usd,
        reserve_usd=reserve_usd,
        jobs_dir=jobs_dir,
        pier_bin=pier_bin,
        pack_id=pack_id,
        skip_preflight=skip_preflight,
        offline=offline,
        no_reclaim=no_reclaim,
        json_out=json_out,
    )


@app.command("curate-hardness")
def curate_hardness_cmd(
    src: Annotated[
        Path,
        typer.Option(
            "--src",
            help="Source Harbor product root (tasks/* dual-truth packs)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/prod_hard_keep"),
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            help="Curated hardness product root (default: same as --src after drop)",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/prod_hard_keep"),
    scoreboard: Annotated[
        Path,
        typer.Option(
            "--scoreboard",
            help=(
                "Panel/eval scoreboard.json or report.json — labels dual-model "
                "solve-alls for reporting (M25: does not auto-drop hardness)"
            ),
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=False,
        ),
    ] = Path("datasets/panel_prod_hard_bench10_n5/scoreboard.json"),
    panel_report: Annotated[
        Path | None,
        typer.Option(
            "--panel-report",
            help="Optional full panel report.json (verdict/rule enrich keep-band)",
            exists=False,
            file_okay=True,
            dir_okay=False,
            resolve_path=False,
        ),
    ] = None,
    min_keep: Annotated[
        int,
        typer.Option(
            "--min-keep",
            help=(
                "Fail-closed residual N floor (default 0 for post-eval demote; "
                "use 5 for fresh test_n10 → prod_hard waves)"
            ),
        ),
    ] = 0,
    include_explicit: Annotated[
        bool,
        typer.Option(
            "--include-explicit-drops",
            help="Also apply legacy EXPLICIT_DROP name table (m21c); off by default for M24",
        ),
    ] = False,
    no_clean: Annotated[
        bool,
        typer.Option(
            "--no-clean",
            help="Do not rmtree --out before materialize (rare; default cleans)",
        ),
    ] = False,
    no_restore: Annotated[
        bool,
        typer.Option(
            "--no-restore",
            help=(
                "Skip M25b re-admit of packs dropped only for dual-model solve-all "
                "(default restores from product archives when dual-truth+floors+intrinsic OK)"
            ),
        ),
    ] = False,
    restore_from: Annotated[
        list[Path] | None,
        typer.Option(
            "--restore-from",
            help=(
                "Optional product archive root(s) for solve-all-only recovery "
                "(default: datasets/deepagent_v1 + seed5 archive + live_materials)"
            ),
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit curation summary as JSON"),
    ] = False,
) -> None:
    """Curate hardness keep set (M25 intrinsic policy / VAL-DINTR-001 + M25b restore).

    Drops only on misalign, hardness floors (thin F2P etc.), and high-confidence
    intrinsic EASY_REQUEST from prompt+gold. Dual-model pass@1=1.0 is labeled
    EASY_SOLVE_ALL for scoreboard notes but does **not** auto-drop hardness.

    M25b: re-admits packs previously dropped only for model solve-all when they
    still pass dual-truth + alignment + floors + intrinsic non-easy
    (VAL-DINTR-003), then re-uploads stay on the caller (VAL-DINTR-004).

    Example::

        deepagent curate-hardness \\
          --src datasets/prod_hard_keep \\
          --scoreboard datasets/panel_prod_hard_bench10_n5/scoreboard.json \\
          --out datasets/prod_hard_keep --json
    """
    from swe_factory.pipeline.curate_prod_hard import (
        ProdHardCurationError,
        curate_hardness_from_scoreboard,
    )
    from swe_factory.pipeline.easy_detect import classify_scoreboard

    if not src.is_dir():
        typer.secho(f"curate-hardness: src missing: {src}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if not scoreboard.is_file():
        typer.secho(
            f"curate-hardness: scoreboard missing: {scoreboard}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    pack_dirs: dict[str, Path] | None = None
    tasks_root = src / "tasks"
    if tasks_root.is_dir():
        pack_dirs = {
            p.name: p for p in tasks_root.iterdir() if p.is_dir() and not p.name.startswith(".")
        }
    easy = classify_scoreboard(scoreboard, pack_dirs=pack_dirs, drop_on_solve_all=False)
    try:
        result = curate_hardness_from_scoreboard(
            src,
            out,
            scoreboard=scoreboard,
            panel_report=panel_report if panel_report and panel_report.is_file() else None,
            min_keep=min_keep,
            clean_out=not no_clean,
            include_explicit_drops=include_explicit,
            drop_on_solve_all=False,
            apply_intrinsic=True,
            restore_solve_all=not no_restore,
            restore_roots=list(restore_from) if restore_from else None,
        )
    except ProdHardCurationError as exc:
        typer.secho(f"curate-hardness fail-closed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    payload = {
        "ok": result.ok,
        "src": str(src),
        "out": str(out),
        "scoreboard": str(scoreboard),
        "pack_count": result.pack_count,
        "keep_ids": list(result.keep_ids),
        "drop_ids": list(result.drop_ids),
        "drop_reasons": result.drop_reasons,
        "restored_solve_all_only": list((result.meta or {}).get("restored_solve_all_only") or []),
        "easy_detect": easy.to_dict(),
        "policy": "m25_intrinsic_hardness",
        "assertions": [
            "VAL-DINTR-001",
            "VAL-DINTR-002",
            "VAL-DINTR-003",
            "VAL-DINTR-005",
            "VAL-DEASY-002",
            "VAL-DEASY-005",
        ],
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        typer.echo(
            f"curate-hardness: ok keep={result.pack_count} drop={len(result.drop_ids)} out={out}"
        )
        for tid in result.drop_ids:
            info = result.drop_reasons.get(tid) or {}
            typer.echo(f"  DROP {tid}: {info.get('reason_code')}")
        for tid in result.keep_ids:
            typer.echo(f"  KEEP {tid}")


@app.command("oracle")
def oracle_cmd(
    pack_dir: Annotated[
        Path | None,
        typer.Option(
            "--pack-dir",
            help="Harbor pack directory (tasks/<id>) to HarborDocker oracle-certify",
            exists=False,
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
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
            help="Evidence / audit parent dir",
            file_okay=False,
            dir_okay=True,
            resolve_path=False,
        ),
    ] = Path("datasets/deepagent_v1"),
    evidence_out: Annotated[
        Path | None,
        typer.Option(
            "--evidence-out",
            help="Write oracle_evidence.json (default: <out>/oracle_evidence.json)",
            file_okay=True,
            dir_okay=False,
            resolve_path=False,
        ),
    ] = None,
    audit_out: Annotated[
        Path | None,
        typer.Option(
            "--audit-out",
            help="Append gate_audit.jsonl for cert funnel",
            file_okay=True,
            dir_okay=False,
            resolve_path=False,
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit oracle cert summary as JSON"),
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
                "Real-PR product cert path: require source_track=real_pr, "
                "HarborDocker dual-truth sol=1 / null=0, refuse fake oracle_mode"
            ),
        ),
    ] = False,
    oracle_mode: Annotated[
        str | None,
        typer.Option(
            "--oracle-mode",
            help="Explicit oracle mode (docker only on product path; fake refused)",
        ),
    ] = None,
) -> None:
    """HarborDocker dual-truth oracle scoring (sol=1 / null=0).

    Product cert path uses HarborDockerVerifier only; fake/stub backends are refused.
    Wraps the same cert pipeline as ``swe-factory deepagent-oracle``.
    """
    from swe_factory.cli import deepagent_oracle_cmd

    deepagent_oracle_cmd(
        pack_dir=pack_dir,
        backend=backend,
        out=out,
        evidence_out=evidence_out,
        audit_out=audit_out,
        json_out=json_out,
        no_pier_hooks=no_pier_hooks,
        real_pr=real_pr,
        oracle_mode=oracle_mode,
    )


if __name__ == "__main__":
    app()
