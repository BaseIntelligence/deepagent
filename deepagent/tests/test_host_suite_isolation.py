"""Dual-run host suite isolation (VAL-DMED-010).

Product host dual-run must not poison the factory venv:
- no ``pip install -U`` / upgrade into factory site-packages
- missing suite deps (pytest-xprocess, simplejson, redis, …) install into an
  isolated per-pack env (or a purpose-built host env), never the factory venv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from swe_factory.pipeline.ship_real_pr import (
    FORBIDDEN_FACTORY_PIP_FLAGS,
    HOST_SUITE_COMMON_DEPS,
    HostSuiteEnv,
    _prepare_host_suite_env,
    assert_host_suite_pip_safe,
    build_host_suite_pip_command,
)


def test_common_deps_include_xprocess_simplejson_redis() -> None:
    deps = {d.lower().split("==")[0].split(">=")[0] for d in HOST_SUITE_COMMON_DEPS}
    assert "pytest-xprocess" in deps
    assert "simplejson" in deps
    assert "redis" in deps
    assert "pytest" in deps


def test_forbidden_factory_pip_flags() -> None:
    assert "-U" in FORBIDDEN_FACTORY_PIP_FLAGS or "--upgrade" in FORBIDDEN_FACTORY_PIP_FLAGS
    assert "--upgrade" in FORBIDDEN_FACTORY_PIP_FLAGS


def test_assert_host_suite_pip_safe_rejects_upgrade_into_factory() -> None:
    factory_py = sys.executable
    with pytest.raises(ValueError, match="factory|upgrade|isolat"):
        assert_host_suite_pip_safe(
            [factory_py, "-m", "pip", "install", "-U", "pytest"],
            factory_python=factory_py,
        )
    with pytest.raises(ValueError, match="factory|upgrade|isolat"):
        assert_host_suite_pip_safe(
            [factory_py, "-m", "pip", "install", "--upgrade", "simplejson"],
            factory_python=factory_py,
        )


def test_assert_host_suite_pip_safe_allows_isolated_python(tmp_path: Path) -> None:
    iso_py = tmp_path / ".sdf_host_venv" / "bin" / "python"
    iso_py.parent.mkdir(parents=True)
    iso_py.write_text("#!/bin/sh\n", encoding="utf-8")
    cmd = [str(iso_py), "-m", "pip", "install", "pytest-xprocess", "simplejson"]
    assert_host_suite_pip_safe(cmd, factory_python=sys.executable)


def test_build_host_suite_pip_command_no_upgrade_flag(tmp_path: Path) -> None:
    iso_py = str(tmp_path / "venv" / "bin" / "python")
    cmd = build_host_suite_pip_command(
        iso_py,
        packages=["pytest-xprocess", "simplejson", "redis"],
    )
    assert cmd[0] == iso_py
    assert "-m" in cmd and "pip" in cmd
    assert "install" in cmd
    joined = " ".join(cmd)
    assert " -U " not in f" {joined} "
    assert "--upgrade" not in cmd
    assert "pytest-xprocess" in cmd
    assert "simplejson" in cmd
    assert "redis" in cmd


def test_prepare_host_suite_env_creates_isolated_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isolated venv under work_root; pip installs target isolated python only."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    calls: list[list[str]] = []

    def fake_run(cmd: list[str] | tuple[str, ...], **kwargs: Any) -> MagicMock:
        c = [str(x) for x in cmd]
        calls.append(c)
        # Simulate venv creation by writing the bin/python path when invoked.
        if len(c) >= 3 and c[1:3] == ["-m", "venv"]:
            dest = Path(c[3])
            (dest / "bin").mkdir(parents=True, exist_ok=True)
            py = dest / "bin" / "python"
            py.write_text("#!/bin/sh\n", encoding="utf-8")
            py.chmod(0o755)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return proc

    monkeypatch.setattr("subprocess.run", fake_run)

    env = _prepare_host_suite_env(repo, language="python", work_root=work)
    assert isinstance(env, HostSuiteEnv)
    assert env.isolated is True
    assert env.python != sys.executable
    assert (
        Path(env.python).as_posix().find(".sdf_host_venv") >= 0
        or "host_venv" in Path(env.python).as_posix()
    )
    # No pip install used factory python with upgrade.
    factory_pip_upgrades = [
        c
        for c in calls
        if c and c[0] == sys.executable and "pip" in c and ("-U" in c or "--upgrade" in c)
    ]
    assert factory_pip_upgrades == []
    # At least one pip install targeted the isolated interpreter.
    iso_installs = [c for c in calls if c and c[0] == env.python and "pip" in c and "install" in c]
    assert iso_installs, f"expected isolated pip install, got {calls!r}"
    for c in iso_installs:
        assert "-U" not in c
        assert "--upgrade" not in c


def test_prepare_host_suite_env_non_python_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = _prepare_host_suite_env(repo, language="go")
    assert env.python  # may fall back to sys.executable for non-python
    assert env.isolated is False or env.language != "python"


def test_prepare_host_suite_result_documents_no_factory_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "setup.py").write_text(
        "from setuptools import setup\nsetup(name='x')\n", encoding="utf-8"
    )
    work = tmp_path / "w"
    work.mkdir()

    def fake_run(cmd: list[str] | tuple[str, ...], **kwargs: Any) -> MagicMock:
        c = [str(x) for x in cmd]
        if len(c) >= 3 and c[1:3] == ["-m", "venv"]:
            dest = Path(c[3])
            (dest / "bin").mkdir(parents=True, exist_ok=True)
            py = dest / "bin" / "python"
            py.write_text("#!/bin/sh\n", encoding="utf-8")
            py.chmod(0o755)
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return proc

    monkeypatch.setattr("subprocess.run", fake_run)
    env = _prepare_host_suite_env(repo, language="python", work_root=work)
    assert env.factory_site_packages_untouched is True
    assert env.to_dict()["isolated"] is True
    assert (
        "factory" in str(env.to_dict().get("notes", {})).lower()
        or env.factory_site_packages_untouched
    )
