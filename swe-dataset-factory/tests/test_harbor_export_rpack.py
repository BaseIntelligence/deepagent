"""Harbor product real_pr export gates (VAL-RPACK-001..005).

Product had path: full DeepSWE tree, source_track=real_pr, real URL/SHA,
multi-file gold + held-out tests, pre_artifacts nonempty, refuse hybrid_bind.
Offline motors remain ``export_harbor_pack`` / archive-only; they must never
land under product deepswe_v1 via ``export_real_harbor_pack``.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swe_factory.cli import app
from swe_factory.envbuild.agent_recipe import render_real_pr_agent_dockerfile
from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS, verify_pack_tree
from swe_factory.harbor.real_pack import (
    REAL_PR_SOURCE_TRACK,
    RealPackError,
    assert_product_real_pr_export,
    export_real_harbor_pack,
    is_product_deepswe_dest,
    run_pre_artifacts_capture,
    validate_real_harbor_pack,
    write_real_harbor_export,
)
from swe_factory.harbor.schema import (
    MODEL_PATCH_ARTIFACT,
    HarborMetadata,
    HarborPackSpec,
    HarborTaskIdentity,
    HarborTaskToml,
    HarborVerifier,
    TestsConfig,
)

_REAL_URL = "https://github.com/pallets/click.git"
_REAL_SHA = "b1c2d3e4f5a6789012345678901234567890abcd"
_MULTI_SOL = textwrap.dedent(
    """\
    diff --git a/src/pkg/core.py b/src/pkg/core.py
    --- a/src/pkg/core.py
    +++ b/src/pkg/core.py
    @@ -1,2 +1,3 @@
     def run():
    -    return 0
    +    return 1
    diff --git a/src/pkg/util.py b/src/pkg/util.py
    --- a/src/pkg/util.py
    +++ b/src/pkg/util.py
    @@ -1,2 +1,3 @@
     def helper():
    -    return "x"
    +    return "y"
    """
)
_TEST_PATCH = textwrap.dedent(
    """\
    diff --git a/tests/test_held_out.py b/tests/test_held_out.py
    new file mode 100644
    --- /dev/null
    +++ b/tests/test_held_out.py
    @@ -0,0 +1,3 @@
    +def test_held():
    +    from src.pkg.core import run
    +    assert run() == 1
    """
)
_INSTRUCTION = (
    "Restore multi-module package behavior after a partial refactor. "
    "Fix `run()` / `helper()` so the held-out contract holds. "
    "Commit your work on a branch; do not leave uncommitted changes."
)


def _real_spec(**overrides: object) -> HarborPackSpec:
    df = render_real_pr_agent_dockerfile(
        repository_url=_REAL_URL,
        base_commit=_REAL_SHA,
        language="python",
    )
    base: dict[str, object] = {
        "task_id": "real-click-rpack-demo",
        "instruction_md": _INSTRUCTION,
        "task_toml": HarborTaskToml(
            schema_version="1.1",
            artifacts=[MODEL_PATCH_ARTIFACT],
            task=HarborTaskIdentity(name="swe-factory/real-click-rpack-demo"),
            metadata=HarborMetadata(
                language="python",
                repository_url=_REAL_URL,
                base_commit_hash=_REAL_SHA,
                task_id="real-click-rpack-demo",
                source_track=REAL_PR_SOURCE_TRACK,
                license="BSD-3-Clause",
            ),
            verifier=HarborVerifier(environment_mode="separate", timeout_sec=1800.0),
        ),
        "tests_config": TestsConfig(
            base_commit=_REAL_SHA,
            f2p_node_ids=["tests.test_held_out.test_held"],
            p2p_node_ids=["tests.test_ok.test_always_ok"],
        ),
        "solution_patch": _MULTI_SOL,
        "test_patch": _TEST_PATCH,
        "environment_dockerfile": df,
        "tests_dockerfile": (
            "FROM deepswe-agent:local\n"
            "COPY test.sh /tests/test.sh\n"
            "COPY grader.py /tests/grader.py\n"
            "COPY config.json /tests/config.json\n"
            "COPY test.patch /tests/test.patch\n"
        ),
    }
    base.update(overrides)
    return HarborPackSpec.model_validate(base)


# --- destination helpers ------------------------------------------------------


def test_is_product_deepswe_dest_marks_product_only() -> None:
    assert is_product_deepswe_dest("datasets/deepswe_v1")
    assert is_product_deepswe_dest("/tmp/foo/datasets/deepswe_v1/tasks/x")
    assert not is_product_deepswe_dest("datasets/deepswe_v1_hybrid_archive")
    assert not is_product_deepswe_dest("datasets/harbor_v1")
    assert not is_product_deepswe_dest("datasets/v1")


# --- VAL-RPACK-001 tree complete ----------------------------------------------


def test_rpack_001_product_harbor_tree_complete(tmp_path: Path) -> None:
    """VAL-RPACK-001: full Harbor v1.1 relpaths for real product pack."""
    pack = export_real_harbor_pack(
        _real_spec(),
        dest=tmp_path / "datasets" / "deepswe_v1" / "tasks" / "real-click-rpack-demo",
    )
    assert pack.validation.ok
    assert pack.validation.tree_complete
    missing = verify_pack_tree(pack.pack_dir)
    assert missing == []
    for rel in REQUIRED_PACK_RELPATHS:
        assert (pack.pack_dir / rel).is_file(), rel


# --- VAL-RPACK-002 real_url + real_sha + real_pr track -------------------------


def test_rpack_002_task_toml_real_url_sha_and_track(tmp_path: Path) -> None:
    """VAL-RPACK-002: public HTTPS url, 40-char sha, source_track=real_pr."""
    pack = export_real_harbor_pack(
        _real_spec(),
        dest=tmp_path / "datasets" / "deepswe_v1" / "tasks" / "real-click-rpack-demo",
    )
    import tomllib

    data = tomllib.loads((pack.pack_dir / "task.toml").read_text(encoding="utf-8"))
    assert data["metadata"]["repository_url"].startswith("https://")
    assert not data["metadata"]["repository_url"].startswith("file://")
    assert len(data["metadata"]["base_commit_hash"]) == 40
    assert data["metadata"]["source_track"] == "real_pr"
    assert data["verifier"]["environment_mode"] == "separate"
    assert pack.validation.real_url_ok
    assert pack.validation.real_sha_ok
    assert pack.validation.details.get("source_track") == "real_pr"


def test_rpack_002_missing_or_hybrid_track_refused(tmp_path: Path) -> None:
    hybrid_meta = HarborTaskToml(
        schema_version="1.1",
        artifacts=[MODEL_PATCH_ARTIFACT],
        task=HarborTaskIdentity(name="swe-factory/hybrid-bad"),
        metadata=HarborMetadata(
            language="python",
            repository_url=_REAL_URL,
            base_commit_hash=_REAL_SHA,
            task_id="hybrid-bad",
            source_track="hybrid_curated",
            license="MIT",
        ),
        verifier=HarborVerifier(environment_mode="separate"),
    )
    with pytest.raises(RealPackError, match="source_track|real_pr|hybrid"):
        export_real_harbor_pack(
            _real_spec(task_toml=hybrid_meta, task_id="hybrid-bad"),
            dest=tmp_path / "datasets" / "deepswe_v1" / "tasks" / "hybrid-bad",
        )

    missing_track = HarborTaskToml(
        schema_version="1.1",
        artifacts=[MODEL_PATCH_ARTIFACT],
        task=HarborTaskIdentity(name="swe-factory/no-track"),
        metadata=HarborMetadata(
            language="python",
            repository_url=_REAL_URL,
            base_commit_hash=_REAL_SHA,
            task_id="no-track",
            source_track=None,
            license="MIT",
        ),
        verifier=HarborVerifier(environment_mode="separate"),
    )
    with pytest.raises(RealPackError, match="source_track|real_pr"):
        export_real_harbor_pack(
            _real_spec(task_toml=missing_track, task_id="no-track"),
            dest=tmp_path / "no-track",
        )


# --- VAL-RPACK-003 refuse hybrid product packaging ----------------------------


def test_rpack_003_refuse_hybrid_bind_copy_repo(tmp_path: Path) -> None:
    """VAL-RPACK-003: product export refuses hybrid_bind motor tree packaging."""
    motor = tmp_path / "motor"
    motor.mkdir()
    (motor / "orderlib").mkdir()
    (motor / "orderlib" / "pricing.py").write_text("x=1\n", encoding="utf-8")
    with pytest.raises(RealPackError, match="copy_repo|hybrid|clone@SHA"):
        export_real_harbor_pack(
            _real_spec(),
            dest=tmp_path / "datasets" / "deepswe_v1" / "tasks" / "bound",
            copy_repo_into_environment=motor,
        )


def test_rpack_003_assert_product_refuses_hybrid_track_on_dest() -> None:
    with pytest.raises(RealPackError, match="hybrid|real_pr"):
        assert_product_real_pr_export(
            source_track="hybrid_curated",
            repository_url=_REAL_URL,
            base_commit=_REAL_SHA,
            dest="datasets/deepswe_v1",
            force_product=True,
        )


def test_rpack_003_cli_refuses_hybrid_source_track() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-real-harbor",
            "--out",
            "datasets/deepswe_v1",
            "--source-track",
            "hybrid_curated",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    joined = (result.stdout or "") + (result.stderr or "")
    assert "refuse" in joined.lower() or "hybrid" in joined.lower() or "real_pr" in joined.lower()


def test_rpack_003_cli_refuses_hybrid_bind_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-real-harbor",
            "--out",
            "datasets/deepswe_v1",
            "--source-track",
            "real_pr",
            "--hybrid-bind",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    joined = (result.stdout or "") + (result.stderr or "")
    assert "hybrid" in joined.lower() or "refuse" in joined.lower()


def test_rpack_003_cli_dry_run_accepts_real_pr() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-real-harbor",
            "--out",
            "datasets/deepswe_v1",
            "--source-track",
            "real_pr",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["source_track"] == "real_pr"
    assert payload["refuse_hybrid"] is True


def test_rpack_003_write_export_refuses_hybrid_on_product_dest(tmp_path: Path) -> None:
    product = tmp_path / "datasets" / "deepswe_v1"
    hybrid_spec = _real_spec(
        task_id="hybrid-noship",
        task_toml=HarborTaskToml(
            schema_version="1.1",
            artifacts=[MODEL_PATCH_ARTIFACT],
            task=HarborTaskIdentity(name="swe-factory/hybrid-noship"),
            metadata=HarborMetadata(
                language="python",
                repository_url=_REAL_URL,
                base_commit_hash=_REAL_SHA,
                task_id="hybrid-noship",
                source_track="hybrid_curated",
                license="MIT",
            ),
            verifier=HarborVerifier(environment_mode="separate"),
        ),
    )
    with pytest.raises(RealPackError, match="hybrid|real_pr|source_track"):
        write_real_harbor_export([hybrid_spec], out_dir=product, overwrite=True)


# --- VAL-RPACK-004 multi-file solution + held-out tests -----------------------


def test_rpack_004_multi_file_and_held_out_metadata(tmp_path: Path) -> None:
    """VAL-RPACK-004: ≥2 product sources; test.patch + f2p/p2p in config.json."""
    pack = export_real_harbor_pack(
        _real_spec(),
        dest=tmp_path / "datasets" / "deepswe_v1" / "tasks" / "real-click-rpack-demo",
    )
    assert pack.validation.multi_file_ok
    assert len(pack.validation.solution_files) >= 2
    assert pack.validation.test_patch_ok
    assert pack.validation.config_ok
    assert pack.validation.f2p_count >= 1
    cfg = json.loads((pack.pack_dir / "tests" / "config.json").read_text(encoding="utf-8"))
    assert cfg["f2p_node_ids"]
    assert "p2p_node_ids" in cfg
    assert (pack.pack_dir / "tests" / "test.patch").stat().st_size > 0


def test_rpack_004_single_file_refused(tmp_path: Path) -> None:
    single = textwrap.dedent(
        """\
        diff --git a/src/only.py b/src/only.py
        --- a/src/only.py
        +++ b/src/only.py
        @@ -1 +1 @@
        -x
        +y
        """
    )
    with pytest.raises(RealPackError, match="multi-file|product|files"):
        export_real_harbor_pack(
            _real_spec(solution_patch=single),
            dest=tmp_path / "single",
        )


# --- VAL-RPACK-005 pre_artifacts nonempty -------------------------------------


def test_rpack_005_pre_artifacts_nonempty_after_sol(tmp_path: Path) -> None:
    """VAL-RPACK-005: pre_artifacts after sol commit produces model.patch > 0."""
    import os
    import subprocess

    from swe_factory.harbor.pre_artifacts import render_pre_artifacts_sh

    repo = tmp_path / "app"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "core.py").write_text("def run():\n    return 0\n", encoding="utf-8")
    (repo / "src" / "pkg" / "util.py").write_text(
        'def helper():\n    return "x"\n', encoding="utf-8"
    )
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@local",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@local",
    }

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

    git("init")
    git("config", "user.email", "fixture@local")
    git("config", "user.name", "fixture")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    sha = git("rev-parse", "HEAD").stdout.strip()
    (repo / ".harbor_base_commit").write_text(sha + "\n", encoding="utf-8")
    (repo / "src" / "pkg" / "core.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "pkg" / "util.py").write_text(
        'def helper():\n    return "y"\n', encoding="utf-8"
    )
    git("add", "-A")
    git("commit", "-q", "-m", "Apply reference solution")

    pack = export_real_harbor_pack(
        _real_spec(),
        dest=tmp_path / "datasets" / "deepswe_v1" / "tasks" / "real-click-rpack-demo",
    )
    pre = (pack.pack_dir / "pre_artifacts.sh").read_text(encoding="utf-8")
    assert "model.patch" in pre
    capture = run_pre_artifacts_capture(
        repo,
        pre_artifacts_sh=render_pre_artifacts_sh(sha),
        logs_dir=tmp_path / "logs",
    )
    assert capture.ok
    assert capture.byte_size > 0


# --- product vs hybrid archive validation cohesion ----------------------------


def test_validate_hybrid_archive_skips_clone_force_without_real_pr_flag(
    tmp_path: Path,
) -> None:
    """Historical hybrid_curated tree may still pass HPACK url/sha/tree when not product-forced."""
    # Emit a minimal pack tree via real_pr, rewrite track to hybrid for inventory-only.
    pack = export_real_harbor_pack(
        _real_spec(),
        dest=tmp_path / "pack",
    )
    toml = pack.pack_dir / "task.toml"
    text = toml.read_text(encoding="utf-8").replace(
        'source_track = "real_pr"',
        'source_track = "hybrid_curated"',
    )
    toml.write_text(text, encoding="utf-8")
    # Inventory-only (ship hybrid historical): no require_real_pr_track
    v = validate_real_harbor_pack(pack.pack_dir, require_real_pr_track=False)
    # Dockerfile is still clone@SHA from export; hybrid track skips *extra* hybrid refuse
    assert v.ok
    # Product product force refuses hybrid track
    v2 = validate_real_harbor_pack(pack.pack_dir, require_real_pr_track=True)
    assert not v2.ok
    assert any("source_track" in r or "hybrid" in r for r in v2.reasons)
