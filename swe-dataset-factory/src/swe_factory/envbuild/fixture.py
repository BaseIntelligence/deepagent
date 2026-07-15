"""Helpers for the offline/green envbuild fixture recipes (multi-lang)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from swe_factory.envbuild.agent_recipe import (
    SUPPORTED_RECIPE_LANGUAGES,
    base_image_for_language,
    default_baseline_test_command,
    default_install_commands,
    normalize_recipe_language,
)
from swe_factory.envbuild.models import EnvRecipe

LangChoice = Literal["python", "go", "typescript", "javascript", "rust"]

_FIXTURE_GREEN = Path("fixtures") / "tiny_green"
_FIXTURE_OFFLINE = Path("fixtures") / "tiny_offline"
_MOTOR_GO = Path("fixtures") / "harbor_motors" / "go_kvstore"
_MOTOR_TS = Path("fixtures") / "harbor_motors" / "ts_registry"
_MOTOR_PY = Path("fixtures") / "harbor_motors" / "python_orders"

# Language-specific defaults for multi-lang real-clone image path
# (VAL-ENVR-007 / VAL-MLANG-001: python|go|javascript|typescript|rust).
_LANG_DEFAULTS: dict[str, dict[str, object]] = {
    lang: {
        "base_image": base_image_for_language(lang),
        "install_commands": default_install_commands(lang),
        "baseline_test_command": default_baseline_test_command(lang),
        "workspace_dir": "/workspace/repo",
    }
    for lang in SUPPORTED_RECIPE_LANGUAGES
}


def _as_cmd_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(c) for c in value]
    return []


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.is_dir() and (path / "repo").is_dir():
            return path
    return candidates[0]


def default_green_fixture_root() -> Path:
    candidates = [
        _package_root() / _FIXTURE_GREEN,
        Path.cwd() / _FIXTURE_GREEN,
        Path("/projects/swe-dataset-factory") / _FIXTURE_GREEN,
    ]
    return _first_existing(candidates)


def default_offline_fixture_root() -> Path:
    candidates = [
        _package_root() / _FIXTURE_OFFLINE,
        Path.cwd() / _FIXTURE_OFFLINE,
        Path("/projects/swe-dataset-factory") / _FIXTURE_OFFLINE,
    ]
    return _first_existing(candidates)


def default_go_fixture_root() -> Path:
    candidates = [
        _package_root() / _MOTOR_GO,
        Path.cwd() / _MOTOR_GO,
        Path("/projects/swe-dataset-factory") / _MOTOR_GO,
    ]
    return _first_existing(candidates)


def default_ts_fixture_root() -> Path:
    candidates = [
        _package_root() / _MOTOR_TS,
        Path.cwd() / _MOTOR_TS,
        Path("/projects/swe-dataset-factory") / _MOTOR_TS,
    ]
    return _first_existing(candidates)


def default_python_motor_fixture_root() -> Path:
    candidates = [
        _package_root() / _MOTOR_PY,
        Path.cwd() / _MOTOR_PY,
        Path("/projects/swe-dataset-factory") / _MOTOR_PY,
    ]
    return _first_existing(candidates)


def _load_meta(root: Path) -> dict[str, object]:
    meta_path = root / "task_meta.json"
    if meta_path.is_file():
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return dict(raw)
    return {}


def recipe_from_green_fixture(fixture_root: Path | None = None) -> EnvRecipe:
    """Buildable green (full suite passes) EnvRecipe from fixtures/tiny_green."""
    root = fixture_root or default_green_fixture_root()
    meta = _load_meta(root)
    repo_id = str(meta.get("repo") or "fixtures/tiny_green")
    base_commit = str(meta.get("base_commit") or "0000000000000000000000000000000000000001")
    language = normalize_recipe_language(str(meta.get("language") or "python"))
    defaults = _LANG_DEFAULTS.get(language, _LANG_DEFAULTS["python"])
    base_image = str(meta.get("base_image") or defaults["base_image"])
    install = meta.get("install_commands")
    if isinstance(install, list) and install:
        install_commands = [str(c) for c in install]
    else:
        install_commands = _as_cmd_list(defaults["install_commands"])
    baseline = str(meta.get("baseline_test_command") or defaults["baseline_test_command"])
    return EnvRecipe(
        repo_id=repo_id,
        base_commit=base_commit,
        language=language,
        base_image=base_image,
        install_commands=install_commands,
        baseline_test_command=baseline,
        local_path=str(root / "repo"),
        allow_internet=False,
        history_scrub=True,
        hooks_off=True,
    )


def recipe_from_offline_broken_fixture(fixture_root: Path | None = None) -> EnvRecipe:
    """Broken offline fixture recipe (baseline intentionally fails)."""
    root = fixture_root or default_offline_fixture_root()
    meta = _load_meta(root)
    repo_id = str(meta.get("repo") or "fixtures/tiny_offline")
    base_commit = str(meta.get("base_commit") or "fixture00000000000000000000000000000001")
    return EnvRecipe(
        repo_id=repo_id,
        base_commit=base_commit,
        language=str(meta.get("language") or "python"),
        base_image="python:3.12-slim",
        install_commands=["pip install -q pytest"],
        baseline_test_command="python -m pytest -q",
        local_path=str(root / "repo"),
        allow_internet=False,
    )


def recipe_from_go_fixture(fixture_root: Path | None = None) -> EnvRecipe:
    """Green Go motor seed (go_kvstore) for multi-lang envbuild path."""
    root = fixture_root or default_go_fixture_root()
    meta = _load_meta(root)
    defaults = _LANG_DEFAULTS["go"]
    return EnvRecipe(
        repo_id=str(meta.get("repo") or "fixtures/harbor_motors/go_kvstore"),
        base_commit=str(meta.get("base_commit") or "b100000000000000000000000000000000000001"),
        language="go",
        base_image=str(defaults["base_image"]),
        install_commands=_as_cmd_list(defaults["install_commands"]),
        baseline_test_command=str(defaults["baseline_test_command"]),
        workspace_dir=str(defaults["workspace_dir"]),
        local_path=str(root / "repo"),
        allow_internet=False,
        history_scrub=True,
        hooks_off=True,
    )


def recipe_from_ts_fixture(fixture_root: Path | None = None) -> EnvRecipe:
    """Green TypeScript motor seed (ts_registry) for multi-lang envbuild path."""
    root = fixture_root or default_ts_fixture_root()
    meta = _load_meta(root)
    defaults = _LANG_DEFAULTS["typescript"]
    return EnvRecipe(
        repo_id=str(meta.get("repo") or "fixtures/harbor_motors/ts_registry"),
        base_commit=str(meta.get("base_commit") or "c100000000000000000000000000000000000001"),
        language="typescript",
        base_image=str(defaults["base_image"]),
        install_commands=_as_cmd_list(defaults["install_commands"]),
        baseline_test_command=str(defaults["baseline_test_command"]),
        workspace_dir=str(defaults["workspace_dir"]),
        local_path=str(root / "repo"),
        allow_internet=False,
        history_scrub=True,
        hooks_off=True,
    )


def recipe_for_language(language: str, fixture_root: Path | None = None) -> EnvRecipe:
    """Dispatch multi-lang fixture recipe (python / go / typescript / js / rust).

    Rust / spray recipes without local motor trees still return correct base image
    + install/test commands via ``recipe_from_clone``-style defaults (no motor COPY).
    """
    lang = normalize_recipe_language(language)
    if lang == "go":
        return recipe_from_go_fixture(fixture_root)
    if lang in {"typescript", "javascript"}:
        # JS instrumentation reuses node TS motor / node base until js_motor lands.
        return recipe_from_ts_fixture(fixture_root)
    if lang == "rust":
        return recipe_from_rust_defaults(fixture_root)
    return recipe_from_green_fixture(fixture_root)


def recipe_from_rust_defaults(
    fixture_root: Path | None = None,
    *,
    repo_id: str = "fixtures/rust_cargo_recipe",
    base_commit: str = "d100000000000000000000000000000000000001",
) -> EnvRecipe:
    """Rust agent recipe defaults (cargo base image + install/test) for VAL-MLANG-001/002.

    No motor fixture is required for unit proof of base image / install commands.
    Live dual-run / cert still use clone@SHA when product claims rust packs.
    """
    del fixture_root  # reserved for future cargo fixture motor
    defaults = _LANG_DEFAULTS["rust"]
    return EnvRecipe(
        repo_id=repo_id,
        base_commit=base_commit,
        language="rust",
        base_image=str(defaults["base_image"]),
        install_commands=_as_cmd_list(defaults["install_commands"]),
        baseline_test_command=str(defaults["baseline_test_command"]),
        workspace_dir=str(defaults["workspace_dir"]),
        allow_internet=False,
        history_scrub=True,
        hooks_off=True,
    )


def recipe_from_clone(
    *,
    repo_id: str,
    base_commit: str,
    language: str = "python",
    clone_url: str | None = None,
    require_real_sha: bool = True,
    image_namespace: str = "deepswe-env",
) -> EnvRecipe:
    """EnvRecipe for a real remote clone pinned at a 40-char base SHA."""
    lang = normalize_recipe_language(language)
    defaults = _LANG_DEFAULTS.get(lang, _LANG_DEFAULTS["python"])
    url = clone_url
    if not url and "/" in repo_id and not Path(repo_id).exists():
        url = f"https://github.com/{repo_id}.git"
    return EnvRecipe(
        repo_id=repo_id,
        base_commit=base_commit,
        language=lang,
        base_image=str(defaults["base_image"]),
        install_commands=_as_cmd_list(defaults["install_commands"]),
        baseline_test_command=str(defaults["baseline_test_command"]),
        workspace_dir=str(defaults["workspace_dir"]),
        clone_url=url,
        require_real_sha=require_real_sha,
        image_namespace=image_namespace,
        allow_internet=False,
        history_scrub=True,
        hooks_off=True,
    )


def language_recipe_table() -> dict[str, dict[str, object]]:
    """Stable language → base_image / install / baseline map (VAL-MLANG-001)."""
    return {
        lang: {
            "language": lang,
            "base_image": str(_LANG_DEFAULTS[lang]["base_image"]),
            "install_commands": list(_as_cmd_list(_LANG_DEFAULTS[lang]["install_commands"])),
            "baseline_test_command": str(_LANG_DEFAULTS[lang]["baseline_test_command"]),
        }
        for lang in SUPPORTED_RECIPE_LANGUAGES
    }


__all__ = [
    "SUPPORTED_RECIPE_LANGUAGES",
    "default_go_fixture_root",
    "default_green_fixture_root",
    "default_offline_fixture_root",
    "default_python_motor_fixture_root",
    "default_ts_fixture_root",
    "language_recipe_table",
    "recipe_for_language",
    "recipe_from_clone",
    "recipe_from_go_fixture",
    "recipe_from_green_fixture",
    "recipe_from_offline_broken_fixture",
    "recipe_from_rust_defaults",
    "recipe_from_ts_fixture",
]
