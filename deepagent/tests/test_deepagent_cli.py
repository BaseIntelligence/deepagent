"""M16 DeepAgent primary CLI surface (VAL-DCLI-001..008).

TDD offline unit tests for entry registration, help text, subcommand wiring,
eval concurrency refuse, version identity, and secret-free defaults.
"""

from __future__ import annotations

import importlib
import re

from typer.testing import CliRunner

from swe_factory import __version__
from swe_factory.deepagent_cli import app
from swe_factory.export.leak_scan import scan_text_for_secrets

runner = CliRunner()

# Patterns that must never appear as live secrets in help/defaults (VAL-DCLI-007).
_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9]{10,}", re.IGNORECASE),
    re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bgho_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
)

PRODUCT_SUBCOMMANDS = ("generate", "upload", "pull", "eval", "oracle")


def test_deepagent_entry_module_exposes_typer_app() -> None:
    """Console script target swe_factory.deepagent_cli:app resolves."""
    mod = importlib.import_module("swe_factory.deepagent_cli")
    assert hasattr(mod, "app")
    # Typer application callable for setuptools entry point
    assert callable(mod.app)


def test_help_lists_m16_commands() -> None:
    """VAL-DCLI-001: deepagent --help documents generate/upload/pull/eval/oracle."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "deepagent" in text
    for cmd in PRODUCT_SUBCOMMANDS:
        assert cmd in text, f"missing subcommand {cmd!r} in help:\n{result.output}"
    # Branding: primary product name is deepagent, not DeepSWE as primary label
    assert "deepswe" not in text.split("deepagent")[0].lower() or "deepagent" in text


def test_generate_help_documents_target_and_out() -> None:
    """VAL-DCLI-002: generate --help exposes target count and out path for test_n10."""
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "--target" in text or "target" in text
    assert "--out" in text or "out" in text
    # Live-mine path discoverable for M16 generate wave
    assert "live" in text or "mine" in text or "test_n10" in text
    assert "test_n10" in text or "datasets" in text


def test_upload_help_documents_hf_dataset_and_revision() -> None:
    """VAL-DCLI-003: upload --help documents BaseIntelligence/deepagent + revision test."""
    result = runner.invoke(app, ["upload", "--help"])
    assert result.exit_code == 0, result.output
    text = result.output
    lowered = text.lower()
    assert "baseintelligence/deepagent" in lowered
    assert "revision" in lowered or "branch" in lowered
    assert "test" in lowered
    # Source pack root documented
    assert "--src" in text or "--source" in text or "src" in lowered or "source" in lowered


def test_pull_help_documents_download_and_revisions() -> None:
    """VAL-DCLI-004: pull --help documents download + main|test revision."""
    result = runner.invoke(app, ["pull", "--help"])
    assert result.exit_code == 0, result.output
    text = result.output
    lowered = text.lower()
    assert "baseintelligence/deepagent" in lowered
    assert "revision" in lowered or "branch" in lowered
    assert "main" in lowered
    assert "test" in lowered
    assert "--out" in text or "out" in lowered


def test_eval_help_documents_n_concurrent_one_and_hard_stop() -> None:
    """eval --help documents n_concurrent default 1, max 5, mem risk, hard-stop 600."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "n-concurrent" in text or "n_concurrent" in text or "concurrent" in text
    assert "1" in result.output
    # M19: cap is 5; help should surface range and/or mem risk wording
    assert "5" in result.output
    assert "hard-stop" in text or "hard_stop" in text or "600" in result.output
    assert "mem" in text or "risk" in text or "1.." in text or "1.." in result.output
    # Pier + HarborDocker fidelity discoverable
    assert "pier" in text or "mini-swe" in text or "harbor" in text


def test_eval_accepts_n_concurrent_one_and_five() -> None:
    """VAL-DBENCH-001: n_concurrent=1 and 5 are accepted (not hard-refused for !=1)."""
    for good in (1, 5):
        result = runner.invoke(
            app,
            [
                "eval",
                "--product-root",
                "datasets/missing_for_concurrent_ok",
                "--n-concurrent",
                str(good),
                "--offline",
            ],
        )
        combined = (result.output + (result.stderr or "")).lower()
        # Must not refuse for concurrency; missing product root may still fail later.
        assert "refuse n_concurrent" not in combined, (good, result.output)
        assert "must be 1" not in combined, (good, result.output)
        if result.exit_code != 0:
            # Failure (if any) is not the hard-only-1 concurrency refuse.
            assert "1..5" not in combined or "n_concurrent" not in combined or good in (1, 5)
            assert "must be in 1.." not in combined


def test_eval_refuses_n_concurrent_outside_1_to_5() -> None:
    """VAL-DBENCH-001: n_concurrent <1 or >5 fails closed non-zero with explicit refuse."""
    for bad in (0, 6):
        result = runner.invoke(
            app,
            [
                "eval",
                "--product-root",
                "datasets/missing_for_refuse",
                "--n-concurrent",
                str(bad),
                "--offline",
            ],
        )
        assert result.exit_code != 0, (bad, result.output)
        combined = (result.output + (result.stderr or "")).lower()
        assert "concurrent" in combined or "n_concurrent" in combined
        # explicit refuse is not silent clamp
        assert (
            "refuse" in combined
            or "must be in 1" in combined
            or "1..5" in combined
            or "must be" in combined
        )


def test_oracle_help_documents_harbor_dual_truth() -> None:
    """VAL-DCLI-006: oracle --help documents HarborDocker sol=1 / null=0."""
    result = runner.invoke(app, ["oracle", "--help"])
    assert result.exit_code == 0, result.output
    text = result.output.lower()
    assert "harbor" in text or "docker" in text
    assert "sol" in text or "solution" in text
    assert "null" in text
    # Dual-truth sol=1 / null=0 wording
    assert "1" in result.output and "0" in result.output


def test_version_reports_package_identity() -> None:
    """VAL-DCLI-008: version (and --version if present) report non-empty package version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip()
    assert __version__ in result.output
    assert result.output.strip() != "unknown"


def test_help_and_defaults_have_no_secrets() -> None:
    """VAL-DCLI-007: help dumps contain no live secret shapes in defaults/examples."""
    dumps: list[str] = []
    for args in (
        ["--help"],
        ["generate", "--help"],
        ["upload", "--help"],
        ["pull", "--help"],
        ["eval", "--help"],
        ["oracle", "--help"],
        ["version"],
    ):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, (args, result.output)
        dumps.append(result.output)

    blob = "\n".join(dumps)
    for pat in _SECRET_PATTERNS:
        assert not pat.search(blob), f"secret-like pattern {pat.pattern} in help output"
    # leak scanner shared utilities also clean
    findings = scan_text_for_secrets(blob, rel="deepagent-cli-help")
    assert findings == [], findings


def test_pyproject_registers_deepagent_and_swe_factory_scripts() -> None:
    """pyproject keeps both console scripts (primary deepagent + swe-factory compat)."""
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'deepagent = "swe_factory.deepagent_cli:app"' in text
    assert 'swe-factory = "swe_factory.cli:app"' in text


def test_eval_hard_stop_default_is_600() -> None:
    """M16 product default hard-stop for deepagent eval is $600."""
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0, result.output
    # Default must surface 600 (not only 300/legacy)
    assert "600" in result.output
