"""Regression tests for Harbor base_commit / pre_artifacts capture.

Root cause: motor Dockerfiles create a real git SHA at build time, but pack
metadata historically stored synthetic placeholders (a1000…). Diffing against
a non-object/non-ancestor ref yields empty model.patch → Pier reward 0.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from swe_factory.harbor.export_pack import export_harbor_pack
from swe_factory.harbor.grader_frame import offline_environment_dockerfile, render_grader_py
from swe_factory.harbor.pre_artifacts import render_pre_artifacts_sh
from swe_factory.harbor.schema import (
    MODEL_PATCH_ARTIFACT,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
)

PLACEHOLDER = "a100000000000000000000000000000000000001"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@local",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@local",
        },
    )


def _init_motor_like_repo(repo: Path) -> str:
    """Mimic Dockerfile: init + commit base, return real SHA."""
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "broken.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@local")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-B", "main")
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / ".harbor_base_commit").write_text(sha + "\n", encoding="utf-8")
    return sha


def test_fake_placeholder_diff_is_empty(tmp_path: Path) -> None:
    """Old bug case: git diff against non-ancestor placeholder captures 0 bytes."""
    repo = tmp_path / "app"
    real = _init_motor_like_repo(repo)
    assert real != PLACEHOLDER
    # Agent "solves" and commits (like solve.sh)
    (repo / "broken.py").write_text("x = 2\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=oracle",
        "-c",
        "user.email=oracle@local",
        "commit",
        "-q",
        "--no-verify",
        "-m",
        "Apply reference solution",
    )
    out = tmp_path / "model_old.patch"
    # Old pre_artifacts briefly: hard-coded fake ref
    r = subprocess.run(
        ["git", "diff", "--binary", PLACEHOLDER, "HEAD"],
        cwd=repo,
        capture_output=True,
    )
    out.write_bytes(r.stdout)
    assert out.stat().st_size == 0, "placeholder base must not yield a usable patch"


def test_dynamic_root_pre_artifacts_captures_nonempty(tmp_path: Path) -> None:
    """New pre_artifacts resolves root commit and captures solution diff."""
    repo = tmp_path / "app"
    real = _init_motor_like_repo(repo)
    (repo / "broken.py").write_text("x = 2\nfixed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=oracle",
        "-c",
        "user.email=oracle@local",
        "commit",
        "-q",
        "--no-verify",
        "-m",
        "Apply reference solution",
    )
    script = tmp_path / "pre_artifacts.sh"
    script.write_text(render_pre_artifacts_sh(PLACEHOLDER), encoding="utf-8")
    script.chmod(0o755)
    logs = tmp_path / "logs"
    # Script cds to /app; simulate by running body with APP override via bash -c
    body = script.read_text(encoding="utf-8").replace("cd /app || exit 0", f"cd {repo} || exit 0")
    body = body.replace("/logs/artifacts", str(logs / "artifacts"))
    runner = tmp_path / "run_pre.sh"
    runner.write_text(body, encoding="utf-8")
    runner.chmod(0o755)
    subprocess.run(["bash", str(runner)], check=True, capture_output=True, text=True)
    patch = logs / "artifacts" / "model.patch"
    assert patch.is_file()
    size = patch.stat().st_size
    assert size > 0, "dynamic root must capture nonempty model.patch after solution commit"
    text = patch.read_text(encoding="utf-8", errors="replace")
    assert "broken.py" in text
    # Sanity: same content as real-base diff
    expected = subprocess.run(
        ["git", "diff", "--binary", real, "HEAD"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    assert patch.read_bytes() == expected


def test_grader_source_resolves_invalid_configured_base() -> None:
    """Shipped grader.py embeds resolve_base_commit fallback."""
    src = render_grader_py()
    assert "resolve_base_commit" in src
    assert "rev-list" in src
    assert ".harbor_base_commit" in src
    assert "git_commit_exists" in src


def test_offline_dockerfile_writes_harbor_base_commit_marker() -> None:
    df = offline_environment_dockerfile()
    assert "git rev-parse HEAD > /app/.harbor_base_commit" in df
    assert 'git commit -q -m "base"' in df
    # Marker is written after the base commit and excluded so solve commits
    # do not accidentally track the helper file.
    assert ".git/info/exclude" in df


def test_export_pre_artifacts_includes_dynamic_fallback(tmp_path: Path) -> None:
    spec = HarborPackSpec.model_validate(
        {
            "task_id": "demo-base-fix",
            "instruction_md": "Fix it and commit.\n",
            "task_toml": HarborTaskToml(
                schema_version="1.1",
                artifacts=[MODEL_PATCH_ARTIFACT],
                task=HarborTaskIdentity(name="swe-factory/demo-base-fix"),
                metadata=HarborMetadata(
                    language="python",
                    repository_url="https://example.com/demo.git",
                    base_commit_hash=PLACEHOLDER,
                    task_id="demo-base-fix",
                ),
                verifier=HarborVerifier(environment_mode="separate", timeout_sec=120.0),
            ),
            "tests_config": TestsConfig(
                base_commit=PLACEHOLDER,
                f2p_node_ids=["tests.test_x.test_y"],
            ),
            "solution_patch": textwrap.dedent(
                """\
                diff --git a/a.py b/a.py
                --- a/a.py
                +++ b/a.py
                @@ -1 +1 @@
                -x
                +y
                """
            ),
            "test_patch": textwrap.dedent(
                """\
                diff --git a/tests/t.py b/tests/t.py
                --- /dev/null
                +++ b/tests/t.py
                @@ -0,0 +1 @@
                +def test_y():
                +    assert True
                """
            ),
            "environment_dockerfile": offline_environment_dockerfile(),
            "tests_dockerfile": "FROM demo:local\nCOPY test.sh /tests/test.sh\n",
        }
    )
    pack = export_harbor_pack(spec, dest=tmp_path / "demo-base-fix")
    pre = (pack.pack_dir / "pre_artifacts.sh").read_text(encoding="utf-8")
    assert PLACEHOLDER in pre
    assert "rev-list --max-parents=0" in pre
    assert ".harbor_base_commit" in pre
    grader = (pack.pack_dir / "tests" / "grader.py").read_text(encoding="utf-8")
    assert "resolve_base_commit" in grader
    df = (pack.pack_dir / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert ".harbor_base_commit" in df


def test_shipped_python_orders_v1_pre_artifacts_is_dynamic() -> None:
    root = Path(__file__).resolve().parents[1]
    pack = (
        root
        / "datasets"
        / "harbor_v1"
        / "tasks"
        / "harbor-python-python_orders_v1-python_orders_v1"
    )
    if not pack.is_dir():
        pytest.skip("harbor_v1 ship not present")
    pre = (pack / "pre_artifacts.sh").read_text(encoding="utf-8")
    assert "rev-list --max-parents=0" in pre
    assert "resolve_base_ref" in pre or "BASE_REF" in pre
    # Must not hard-diff only the placeholder without fallback
    assert "git cat-file" in pre
    cfg = json.loads((pack / "tests" / "config.json").read_text(encoding="utf-8"))
    assert cfg["base_commit"]  # metadata still present
    grader = (pack / "tests" / "grader.py").read_text(encoding="utf-8")
    assert "resolve_base_commit" in grader
    df = (pack / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert ".harbor_base_commit" in df
