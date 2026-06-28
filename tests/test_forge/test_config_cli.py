"""Unit tests for the forge config + CLI skeleton (offline, no network).

Covers the M1 foundation invariants:
- ForgeSettings reads endpoint config strictly from TEACHER_LLM_* env vars.
- Both ForgeSettings and the shared Settings tolerate arbitrary extra env vars.
- The `swe-forge forge` CLI group is wired in with stubbed subcommands.
- No provider/brand string ("yunwu") appears in src/.
- The forge package never imports the bespoke LLM clients or llm/cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from swe_forge.config import Settings
from swe_forge.forge.config import ForgeSettings

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "swe_forge"
FORGE_ROOT = SRC_ROOT / "forge"

runner = CliRunner()


class TestForgeSettings:
    def test_loads_strictly_from_teacher_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TEACHER_LLM_BASE_URL": "https://probe.example",
                "TEACHER_LLM_API_KEY": "sk-probe",
                "TEACHER_LLM_MODEL": "anthropic/x",
                "TEACHER_LLM_PROVIDER": "anthropic",
            },
            clear=True,
        ):
            settings = ForgeSettings(_env_file=None)

        assert settings.teacher_llm_base_url == "https://probe.example"
        assert settings.teacher_llm_api_key == "sk-probe"
        assert settings.teacher_llm_model == "anthropic/x"
        assert settings.teacher_llm_provider == "anthropic"

    def test_defaults_empty_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = ForgeSettings(_env_file=None)

        assert settings.teacher_llm_base_url == ""
        assert settings.teacher_llm_api_key == ""
        assert settings.teacher_llm_model == ""
        assert settings.teacher_llm_provider == ""
        assert settings.panel_llm_base_url == ""
        assert settings.panel_llm_api_key == ""

    def test_no_default_overrides_a_set_var(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TEACHER_LLM_BASE_URL": "https://only-from-env",
                "TEACHER_LLM_MODEL": "openai/from-env",
            },
            clear=True,
        ):
            settings = ForgeSettings(_env_file=None)

        assert settings.teacher_llm_base_url == "https://only-from-env"
        assert settings.teacher_llm_model == "openai/from-env"

    def test_panel_falls_back_to_teacher_when_unset(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TEACHER_LLM_BASE_URL": "https://teacher.example",
                "TEACHER_LLM_API_KEY": "sk-teacher",
            },
            clear=True,
        ):
            settings = ForgeSettings(_env_file=None)

        assert settings.effective_panel_base_url == "https://teacher.example"
        assert settings.effective_panel_api_key == "sk-teacher"

    def test_panel_override_is_honored_when_set(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TEACHER_LLM_BASE_URL": "https://teacher.example",
                "TEACHER_LLM_API_KEY": "sk-teacher",
                "PANEL_LLM_BASE_URL": "https://panel.example",
                "PANEL_LLM_API_KEY": "sk-panel",
            },
            clear=True,
        ):
            settings = ForgeSettings(_env_file=None)

        assert settings.panel_llm_base_url == "https://panel.example"
        assert settings.effective_panel_base_url == "https://panel.example"
        assert settings.effective_panel_api_key == "sk-panel"


class TestExtraEnvTolerance:
    def test_forge_settings_ignores_extra_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TEACHER_LLM_BASE_URL": "https://probe.example",
                "TEACHER_LLM_API_KEY": "sk-probe",
                "TEACHER_LLM_MODEL": "anthropic/x",
                "TEACHER_LLM_PROVIDER": "anthropic",
                "TEACHER_LLM_SOMETHING_NEW": "1",
                "SOME_RANDOM_VAR": "z",
            },
            clear=True,
        ):
            # Must not raise ValidationError/ExtraForbidden.
            settings = ForgeSettings(_env_file=None)
            assert settings.teacher_llm_base_url == "https://probe.example"

    def test_shared_settings_ignores_extra_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "x",
                "GITHUB_TOKEN": "y",
                "TEACHER_LLM_BASE_URL": "https://probe.example",
                "TEACHER_LLM_API_KEY": "sk-probe",
                "TEACHER_LLM_MODEL": "anthropic/x",
                "TEACHER_LLM_PROVIDER": "anthropic",
                "SOME_RANDOM_VAR": "z",
            },
            clear=True,
        ):
            # TEACHER_LLM_* and other unknowns must not break the shared Settings.
            settings = Settings(_env_file=None)
            assert settings.openrouter_api_key == "x"


class TestForgeCli:
    def test_help_lists_group_and_subcommands(self) -> None:
        from swe_forge.forge.cli import app as forge_app

        result = runner.invoke(forge_app, ["--help"])
        assert result.exit_code == 0
        assert "info" in result.output
        assert "llm-check" in result.output
        assert "panel-info" in result.output

    def test_info_runs_and_redacts_secret(self) -> None:
        from swe_forge.forge.cli import app as forge_app

        with patch.dict(
            os.environ,
            {
                "TEACHER_LLM_BASE_URL": "https://probe.example",
                "TEACHER_LLM_API_KEY": "sk-super-secret-value",
                "TEACHER_LLM_MODEL": "anthropic/x",
                "TEACHER_LLM_PROVIDER": "anthropic",
            },
            clear=True,
        ):
            result = runner.invoke(forge_app, ["info"])

        assert result.exit_code == 0
        assert "sk-super-secret-value" not in result.output
        assert "teacher" in result.output.lower()

    def test_panel_info_is_implemented(self) -> None:
        from swe_forge.forge.cli import app as forge_app

        with patch.dict(
            os.environ,
            {
                "TEACHER_LLM_BASE_URL": "https://teacher.example",
                "TEACHER_LLM_API_KEY": "sk-teacher",
            },
            clear=True,
        ):
            result = runner.invoke(forge_app, ["panel-info"])
        assert result.exit_code == 0
        assert "panel" in result.output.lower()

    def test_forge_group_wired_into_main(self) -> None:
        from swe_forge.__main__ import app as root_app

        result = runner.invoke(root_app, ["--help"])
        assert result.exit_code == 0
        assert "forge" in result.output


class TestSourceInvariants:
    def test_no_provider_brand_string_in_src(self) -> None:
        offenders = []
        for path in SRC_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "yunwu" in text.lower():
                offenders.append(str(path))
        assert offenders == [], f"provider brand string found in: {offenders}"

    def test_forge_does_not_import_bespoke_llm_or_cache(self) -> None:
        offenders = []
        for path in FORGE_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "from swe_forge.llm" in text or "llm.cache" in text:
                offenders.append(str(path))
        assert offenders == [], f"forbidden LLM import found in: {offenders}"
