"""Package layout and documentation expectations for first-visit CLI use."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_build_export_score() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "swe-factory --help" in lowered or "swe-factory" in lowered
    assert "build" in lowered
    assert "export" in lowered
    assert "score" in lowered
    assert ".env.example" in readme


def test_readme_product_surface_is_deepagent_v1() -> None:
    """Product docs: current hardness is prod_hard_deepswe_med N=9; archives labeled historical."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    # Authoritative current product (M27/M28/M29).
    assert "datasets/prod_hard_deepswe_med" in readme
    assert "n=9" in lowered or "**n=9**" in lowered or "n = **9**" in lowered
    assert "baseintelligence/deepagent" in lowered
    assert "deepagent" in lowered
    # Compatibility / historical factory still documented.
    assert "ship-deepagent" in lowered
    assert "datasets/deepagent_v1" in readme
    assert "datasets/harbor_v1" in readme
    assert "datasets/v1" in readme
    # Historical surfaces must be clearly non-product.
    assert (
        "historical" in lowered
        or "historical only" in lowered
        or "not** current product" in lowered
        or "not current product" in lowered
        or "audit only" in lowered
    )
    # Current product called out as authoritative / current product surface.
    assert "current product" in lowered or "authoritative" in lowered
    # Mission diary / assertion ledger must stay out of the tracked README.
    assert "worker session" not in lowered
    assert "/root/.factory/missions" not in readme
    assert "m10 sealed" not in lowered


def test_env_example_models_and_budget() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY=" in text
    assert "FACTORY_TEACHER_MODEL=anthropic/claude-opus-4.8" in text
    assert "x-ai/grok-4.5" in text
    assert "moonshotai/kimi-k2.6" in text
    # Real-PR panel pair only by default; teacher may still name Opus.
    assert "FACTORY_PANEL_MODELS=x-ai/grok-4.5,moonshotai/kimi-k2.6" in text
    assert "FACTORY_BUDGET_USD=600" in text
    assert "OXYLABS_USERNAME" in text
    assert "OXYLABS_PASSWORD" in text


def test_pyproject_console_script() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'swe-factory = "swe_factory.cli:app"' in text
    assert 'requires-python = ">=3.12"' in text
