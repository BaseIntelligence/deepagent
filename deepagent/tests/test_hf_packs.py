"""HF pack upload/pull (VAL-DHF-001..005) offline unit tests with mocked Hub.

TDD: schema packing, pack_manifest write, dry-run, fail-closed missing token,
pull round-trip task_ids, bad revision → non-zero, never log tokens.
Live smoke (when HF_TOKEN present) is integration-marked and optional.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from swe_factory.deepagent_cli import app
from swe_factory.export.hf_packs import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    MSG_PULL_AUTH,
    MSG_PULL_HUB,
    MSG_PULL_REVISION,
    MSG_TOKEN_MISSING,
    MSG_UPLOAD_AUTH,
    MSG_UPLOAD_HUB,
    MSG_UPLOAD_REPO,
    HfPacksError,
    build_pack_manifest,
    is_auth_hub_failure,
    list_pack_dirs,
    map_hub_failure,
    pull_pack_tree,
    pull_packs,
    resolve_hf_token,
    upload_pack_tree,
    upload_packs,
    validate_pack_corpus,
)
from swe_factory.export.leak_scan import scan_text_for_secrets
from swe_factory.harbor.export_pack import REQUIRED_PACK_RELPATHS

runner = CliRunner()

_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9]{10,}", re.IGNORECASE),
    re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bgho_[A-Za-z0-9_]{10,}\b"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}", re.IGNORECASE),
)


def _write_minimal_pack(pack_dir: Path, task_id: str = "demo-task") -> Path:
    """Write a complete Harbor pack tree satisfying REQUIRED_PACK_RELPATHS."""
    pack_dir.mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_PACK_RELPATHS:
        path = pack_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if rel == "task.toml":
            path.write_text(
                "[metadata]\n"
                f'task_id = "{task_id}"\n'
                'language = "python"\n'
                'repository_url = "https://github.com/example/demo.git"\n'
                'base_commit_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
                'source_track = "real_pr"\n',
                encoding="utf-8",
            )
        elif rel.endswith(".patch"):
            path.write_text(
                "diff --git a/demo.py b/demo.py\n"
                "--- a/demo.py\n"
                "+++ b/demo.py\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n",
                encoding="utf-8",
            )
        elif rel.endswith(".sh"):
            path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            path.chmod(0o755)
        elif rel.endswith(".json"):
            path.write_text(
                json.dumps({"f2p_node_ids": ["t1"], "p2p_node_ids": ["t2"]}) + "\n",
                encoding="utf-8",
            )
        elif "Dockerfile" in rel:
            path.write_text("FROM python:3.12-slim\n", encoding="utf-8")
        elif rel == "instruction.md":
            path.write_text("# Fix the demo bug\n\nRestore correct behaviour.\n", encoding="utf-8")
        else:
            path.write_text("# placeholder\n", encoding="utf-8")
    return pack_dir


def _write_corpus(root: Path, task_ids: list[str] | None = None) -> Path:
    ids = task_ids or ["task-alpha", "task-beta"]
    tasks = root / "tasks"
    for tid in ids:
        _write_minimal_pack(tasks / tid, task_id=tid)
    (root / "pack_manifest.json").write_text(
        json.dumps({"count": len(ids), "task_ids": ids, "ok": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# Schema / local packing
# ---------------------------------------------------------------------------


def test_validate_pack_corpus_accepts_complete_tree(tmp_path: Path) -> None:
    """VAL-DHF-001: complete Harbor pack corpus validates schema OK."""
    root = _write_corpus(tmp_path / "corpus")
    result = validate_pack_corpus(root)
    assert result.ok is True
    assert result.task_ids == ("task-alpha", "task-beta")
    assert result.reasons == ()


def test_validate_pack_corpus_rejects_missing_required(tmp_path: Path) -> None:
    """VAL-DHF-001: missing task.toml / tests / solution fails closed."""
    root = _write_corpus(tmp_path / "corpus", ["broken"])
    (root / "tasks" / "broken" / "task.toml").unlink()
    (root / "tasks" / "broken" / "tests" / "test.sh").unlink()
    result = validate_pack_corpus(root)
    assert result.ok is False
    assert any("task.toml" in r for r in result.reasons)
    joined = " ".join(result.reasons)
    assert "test.sh" in joined or "tests" in joined


def test_validate_pack_corpus_rejects_empty_tasks(tmp_path: Path) -> None:
    """Empty corpus is refused (no product pad)."""
    root = tmp_path / "empty"
    (root / "tasks").mkdir(parents=True)
    result = validate_pack_corpus(root)
    assert result.ok is False
    assert any("empty" in r.lower() or "no pack" in r.lower() for r in result.reasons)


def test_upload_builds_manifest_layout(tmp_path: Path) -> None:
    """Offline mock: upload packs write/refresh pack_manifest and push folder."""
    root = _write_corpus(tmp_path / "src")
    mock_api = MagicMock()
    mock_api.create_repo.return_value = None
    mock_api.create_branch.return_value = None
    mock_api.upload_folder.return_value = "commit-sha-demo"

    result = upload_packs(
        local_root=root,
        repo_id="BaseIntelligence/deepagent",
        revision="test",
        token="hf_UNITTEST_TOKEN_NOT_REAL_XXXX",
        dry_run=False,
        api=mock_api,
    )

    assert result["ok"] is True
    assert result["pushed"] is True
    assert result["revision"] == "test"
    assert result["repo_id"] == "BaseIntelligence/deepagent"
    assert set(result["task_ids"]) == {"task-alpha", "task-beta"}
    assert "token" not in result
    # manifest written/refreshed
    manifest = json.loads((root / "pack_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["task_ids"]) == {"task-alpha", "task-beta"}
    mock_api.upload_folder.assert_called_once()
    call_kwargs = mock_api.upload_folder.call_args.kwargs
    assert call_kwargs["repo_id"] == "BaseIntelligence/deepagent"
    assert call_kwargs["revision"] == "test"
    assert call_kwargs["repo_type"] == "dataset"


def test_upload_dry_run_skips_hub(tmp_path: Path) -> None:
    """VAL-DHF-001: dry-run validates schema and never calls HfApi."""
    root = _write_corpus(tmp_path / "src")
    mock_api = MagicMock()
    result = upload_packs(
        local_root=root,
        revision="test",
        dry_run=True,
        token=None,
        api=mock_api,
    )
    mock_api.upload_folder.assert_not_called()
    mock_api.create_repo.assert_not_called()
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["pushed"] is False
    assert result["schema_ok"] is True


def test_upload_fail_closed_without_token(tmp_path: Path) -> None:
    """Live mode refuses missing HF token (no network spam)."""
    root = _write_corpus(tmp_path / "src")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(HfPacksError, match="token|HF_TOKEN|missing"),
    ):
        upload_packs(
            local_root=root,
            revision="test",
            dry_run=False,
            token=None,
        )


def test_upload_invalid_schema_never_pushes(tmp_path: Path) -> None:
    """Invalid layout fails before any Hub client call."""
    root = tmp_path / "bad"
    (root / "tasks" / "x").mkdir(parents=True)
    (root / "tasks" / "x" / "readme.txt").write_text("nope\n", encoding="utf-8")
    mock_api = MagicMock()
    with pytest.raises(HfPacksError, match="schema|missing|invalid"):
        upload_packs(local_root=root, token="hf_x", dry_run=False, api=mock_api)
    mock_api.upload_folder.assert_not_called()
    mock_api.create_repo.assert_not_called()


def test_pull_roundtrip_task_ids(tmp_path: Path) -> None:
    """Pulled tree task_ids match pack_manifest."""
    remote_tree = _write_corpus(tmp_path / "remote_snapshot")
    out = tmp_path / "out"

    def _fake_snapshot(**kwargs: Any) -> str:
        # Materialize like snapshot_download into local_dir
        dest = Path(kwargs["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        import shutil

        for item in remote_tree.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        return str(dest)

    result = pull_packs(
        out_dir=out,
        repo_id=DEFAULT_HF_REPO_ID,
        revision="test",
        token="hf_UNITTEST_TOKEN",
        snapshot_fn=_fake_snapshot,
    )

    assert result["ok"] is True
    assert result["pulled"] is True
    assert set(result["task_ids"]) == {"task-alpha", "task-beta"}
    assert (out / "pack_manifest.json").is_file()
    assert (out / "tasks" / "task-alpha" / "task.toml").is_file()
    assert (out / "tasks" / "task-alpha" / "environment" / "Dockerfile").is_file()
    assert (out / "tasks" / "task-alpha" / "tests" / "test.sh").is_file()
    assert (out / "tasks" / "task-alpha" / "solution" / "solution.patch").is_file()
    assert "token" not in result


def test_pull_bad_revision_fails_closed(tmp_path: Path) -> None:
    """VAL-DHF-004: nonexistent revision must raise (caller maps to non-zero)."""
    out = tmp_path / "bad_rev"
    creative = "RevisionNotFound: does-not-exist-xyz UNIQUE_HUB_REV_BANANA_99"

    def _boom(**_kwargs: Any) -> str:
        raise RuntimeError(creative)

    with pytest.raises(HfPacksError) as ei:
        pull_packs(
            out_dir=out,
            revision="does-not-exist-xyz",
            token="hf_UNITTEST",
            snapshot_fn=_boom,
        )
    msg = str(ei.value)
    assert msg == MSG_PULL_REVISION
    assert "UNIQUE_HUB_REV_BANANA_99" not in msg
    assert "does-not-exist-xyz" not in msg


def test_resolve_token_reads_env_and_never_returns_empty() -> None:
    with patch.dict(
        os.environ,
        {"HF_TOKEN": "hf_secret_value_abc", "HUGGING_FACE_HUB_TOKEN": ""},
        clear=False,
    ):
        assert resolve_hf_token() == "hf_secret_value_abc"
    with patch.dict(
        os.environ,
        {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "hf_alt_token_xyz"},
        clear=False,
    ):
        assert resolve_hf_token(token=None) == "hf_alt_token_xyz"
    assert resolve_hf_token(token="explicit") == "explicit"
    with patch.dict(
        os.environ,
        {"HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": ""},
        clear=False,
    ):
        assert resolve_hf_token() is None


def test_build_pack_manifest_lists_task_ids(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "c")
    packs = list_pack_dirs(root)
    manifest = build_pack_manifest(root, packs)
    assert manifest["count"] == 2
    assert set(manifest["task_ids"]) == {"task-alpha", "task-beta"}
    assert manifest["schema"] == "deepagent.hf_packs.v1"
    assert DEFAULT_HF_REPO_ID in str(manifest.get("repo_id", DEFAULT_HF_REPO_ID))


def test_upload_pack_tree_alias(tmp_path: Path) -> None:
    """CLI hook name upload_pack_tree is provided."""
    root = _write_corpus(tmp_path / "src")
    mock_api = MagicMock()
    mock_api.create_repo.return_value = None
    mock_api.create_branch.return_value = None
    mock_api.upload_folder.return_value = "ok"
    result = upload_pack_tree(
        src=root,
        repo_id=DEFAULT_HF_REPO_ID,
        revision=DEFAULT_HF_REVISION,
        token="hf_t",
        api=mock_api,
    )
    assert result["pushed"] is True
    assert result["ok"] is True


def test_pull_pack_tree_alias(tmp_path: Path) -> None:
    remote = _write_corpus(tmp_path / "remote")
    out = tmp_path / "pulled"

    def _snap(**kwargs: Any) -> str:
        import shutil

        dest = Path(kwargs["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        for item in remote.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return str(dest)

    result = pull_pack_tree(
        out=out,
        repo_id=DEFAULT_HF_REPO_ID,
        revision="test",
        token="hf_t",
        snapshot_fn=_snap,
    )
    assert result["pulled"] is True
    assert set(result["task_ids"]) == {"task-alpha", "task-beta"}


# ---------------------------------------------------------------------------
# CLI surface wiring
# ---------------------------------------------------------------------------


def test_cli_upload_dry_run_validates(tmp_path: Path) -> None:
    """VAL-DHF-001: deepagent upload --dry-run validates without push."""
    root = _write_corpus(tmp_path / "src")
    result = runner.invoke(
        app,
        ["upload", "--src", str(root), "--revision", "test", "--dry-run", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_ok"] is True
    assert payload["dry_run"] is True
    assert payload.get("pushed") is False
    for pat in _SECRET_PATTERNS:
        assert not pat.search(result.output)


def test_cli_upload_rejects_broken_layout(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    (bad / "tasks" / "x").mkdir(parents=True)
    result = runner.invoke(app, ["upload", "--src", str(bad), "--dry-run", "--json"])
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert (
        "schema" in combined.lower()
        or "missing" in combined.lower()
        or "invalid" in combined.lower()
    )


def test_cli_upload_fail_closed_without_token(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "src")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    result = runner.invoke(
        app,
        ["upload", "--src", str(root), "--revision", "test", "--json"],
        env=env,
    )
    assert result.exit_code != 0
    combined = (result.output + (result.stderr or "")).lower()
    assert "token" in combined or "missing" in combined


def test_cli_pull_bad_revision_nonzero(tmp_path: Path) -> None:
    """VAL-DHF-004 via CLI — inject boom through pull_pack_tree path."""
    out = tmp_path / "pull_out"
    creative = (
        "404 Revision 'no-such-rev-zzzz' not found UNIQUE_HUB_REV_LEAK_XYZ "
        "on BaseIntelligence/deepagent"
    )

    def _boom(**_kwargs: Any) -> dict[str, Any]:
        # Module already maps to constant; CLI must surface that constant only.
        raise HfPacksError(MSG_PULL_REVISION)

    with patch("swe_factory.export.hf_packs.pull_pack_tree", side_effect=_boom):
        result = runner.invoke(
            app,
            [
                "pull",
                "--out",
                str(out),
                "--revision",
                "no-such-rev-zzzz",
                "--json",
            ],
            env={**os.environ, "HF_TOKEN": "UNITTEST_FAKE_TOKEN_VALUE"},
        )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert MSG_PULL_REVISION in combined
    assert "UNIQUE_HUB_REV_LEAK_XYZ" not in combined
    assert "no-such-rev-zzzz" not in combined or MSG_PULL_REVISION in combined
    # Even if revision name is in CLI argv echo, raw Hub creative text must not appear.
    assert creative not in combined
    assert "UNITTEST_FAKE_TOKEN_VALUE" not in combined
    for pat in _SECRET_PATTERNS:
        assert not pat.search(combined)
    findings = scan_text_for_secrets(combined, rel="cli-pull-bad-rev")
    assert findings == [], findings


def test_cli_pull_dry_run_ok() -> None:
    result = runner.invoke(
        app,
        ["pull", "--revision", "test", "--out", "datasets/hf_pull_test", "--dry-run", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload.get("pulled") is False


def test_result_dicts_never_embed_token(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "src")
    secret = "TOKEN_SHOULD_NOT_LEAK_12345_ABC"
    mock_api = MagicMock()
    mock_api.create_repo.return_value = None
    mock_api.create_branch.return_value = None
    mock_api.upload_folder.return_value = "cmt"
    result = upload_packs(local_root=root, token=secret, dry_run=False, api=mock_api)
    blob = json.dumps(result, default=str)
    assert secret not in blob
    assert "SHOULD_NOT_LEAK" not in blob
    findings = scan_text_for_secrets(blob, rel="upload-result")
    assert findings == [], findings


# ---------------------------------------------------------------------------
# Auth-safe Hub error sanitization (misc-m16-hf-error-sanitization)
# ---------------------------------------------------------------------------


class _FakeHubAuthError(Exception):
    """Standalone mock auth failure with creative Hub-style body text."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        super().__init__(detail)


def test_map_hub_failure_auth_upload_uses_constant() -> None:
    """Known auth/401 body is classified as upload auth fail — never echoed."""
    creative = (
        "401 Client Error: Unauthorized for url: https://huggingface.co/api/ "
        "Invalid username or password. UNIQUE_AUTH_LEAK_TOKEN_hf_should_never_echo"
    )
    exc = _FakeHubAuthError(401, creative)
    assert is_auth_hub_failure(exc) is True
    mapped = map_hub_failure("upload", exc)
    assert mapped == MSG_UPLOAD_AUTH
    assert "UNIQUE_AUTH_LEAK_TOKEN" not in mapped
    assert "hf_should_never_echo" not in mapped
    assert creative not in mapped


def test_map_hub_failure_auth_pull_uses_constant() -> None:
    creative = "403 Forbidden — GatedRepo UNIQUE_GATED_BODY_DETAIL_XYZ"

    # Force via type name path by subclassing as GatedRepoError-like name
    class GatedRepoError(Exception):
        pass

    gated = GatedRepoError(creative)
    assert is_auth_hub_failure(gated) is True
    mapped = map_hub_failure("pull", gated)
    assert mapped == MSG_PULL_AUTH
    assert "UNIQUE_GATED_BODY_DETAIL_XYZ" not in mapped


def test_map_hub_failure_generic_hub_uses_constant() -> None:
    creative = "500 Internal Server Error UNIQUE_HUB_500_BODY_DO_NOT_ECHO"
    mapped_u = map_hub_failure("upload", RuntimeError(creative), stage="upload_folder")
    mapped_p = map_hub_failure("pull", RuntimeError(creative), stage="snapshot_download")
    assert mapped_u == MSG_UPLOAD_HUB
    assert mapped_p == MSG_PULL_HUB
    assert "UNIQUE_HUB_500_BODY_DO_NOT_ECHO" not in mapped_u
    assert "UNIQUE_HUB_500_BODY_DO_NOT_ECHO" not in mapped_p


def test_upload_auth_failure_sanitizes_hub_text(tmp_path: Path) -> None:
    """Mock Hub upload_folder 401 → HfPacksError constant, no raw str(exc)."""
    root = _write_corpus(tmp_path / "src")
    creative = (
        "401 Unauthorized Invalid credentials UNIQUE_UPLOAD_AUTH_BANANA "
        "Bearer hf_CREATIVE_LEAK_SHOULD_VANISH for repo demo/x"
    )
    mock_api = MagicMock()
    mock_api.create_repo.return_value = None
    mock_api.create_branch.return_value = None
    mock_api.upload_folder.side_effect = _FakeHubAuthError(401, creative)

    with pytest.raises(HfPacksError) as ei:
        upload_packs(
            local_root=root,
            token="hf_UNITTEST_TOKEN_NOT_REAL_XXXX",
            dry_run=False,
            api=mock_api,
        )
    msg = str(ei.value)
    assert msg == MSG_UPLOAD_AUTH
    assert "UNIQUE_UPLOAD_AUTH_BANANA" not in msg
    assert "hf_CREATIVE_LEAK_SHOULD_VANISH" not in msg
    assert "Bearer" not in msg


def test_upload_create_repo_auth_failure_sanitizes(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "src")
    creative = "403 Forbidden UNIQUE_CREATE_REPO_AUTH_BODY_zz99"
    mock_api = MagicMock()
    mock_api.create_repo.side_effect = _FakeHubAuthError(403, creative)
    with pytest.raises(HfPacksError) as ei:
        upload_packs(
            local_root=root,
            token="hf_UNITTEST",
            dry_run=False,
            api=mock_api,
        )
    msg = str(ei.value)
    assert msg == MSG_UPLOAD_AUTH
    assert "UNIQUE_CREATE_REPO_AUTH_BODY_zz99" not in msg
    mock_api.upload_folder.assert_not_called()


def test_upload_generic_hub_failure_sanitizes(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "src")
    creative = "SSL handshake failed UNIQUE_UPLOAD_SSL_BODY_AA11 at edge node"
    mock_api = MagicMock()
    mock_api.create_repo.return_value = None
    mock_api.create_branch.return_value = None
    mock_api.upload_folder.side_effect = RuntimeError(creative)
    with pytest.raises(HfPacksError) as ei:
        upload_packs(
            local_root=root,
            token="hf_UNITTEST",
            dry_run=False,
            api=mock_api,
        )
    msg = str(ei.value)
    assert msg == MSG_UPLOAD_HUB
    assert "UNIQUE_UPLOAD_SSL_BODY_AA11" not in msg


def test_upload_create_repo_non_auth_uses_repo_constant(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "src")
    creative = "503 Service unavailable UNIQUE_CREATE_REPO_503_BODY"
    mock_api = MagicMock()
    mock_api.create_repo.side_effect = RuntimeError(creative)
    with pytest.raises(HfPacksError) as ei:
        upload_packs(
            local_root=root,
            token="hf_UNITTEST",
            dry_run=False,
            api=mock_api,
        )
    msg = str(ei.value)
    assert msg == MSG_UPLOAD_REPO
    assert "UNIQUE_CREATE_REPO_503_BODY" not in msg


def test_pull_auth_failure_sanitizes_hub_text(tmp_path: Path) -> None:
    out = tmp_path / "pull_auth_out"
    creative = (
        "401 Client Error Unauthorized UNIQUE_PULL_AUTH_PINEAPPLE "
        "token hf_FAKESECRET_BOTTOM_of_page"
    )

    def _boom(**_kwargs: Any) -> str:
        raise _FakeHubAuthError(401, creative)

    with pytest.raises(HfPacksError) as ei:
        pull_packs(
            out_dir=out,
            revision="test",
            token="hf_UNITTEST",
            snapshot_fn=_boom,
        )
    msg = str(ei.value)
    assert msg == MSG_PULL_AUTH
    assert "UNIQUE_PULL_AUTH_PINEAPPLE" not in msg
    assert "hf_FAKESECRET_BOTTOM_of_page" not in msg


def test_cli_upload_auth_failure_uses_constant(tmp_path: Path) -> None:
    """CLI upload path echoes constant, never creative Hub body from raise."""
    root = _write_corpus(tmp_path / "src")
    creative = "401 Unauthorized UNIQUE_CLI_UPLOAD_AUTH_LEAK_99 hf_should_vanish"

    def _boom(**_kwargs: Any) -> dict[str, Any]:
        # Simulate hf_packs already mapping to constant (production path).
        raise HfPacksError(MSG_UPLOAD_AUTH)

    with patch("swe_factory.export.hf_packs.upload_pack_tree", side_effect=_boom):
        result = runner.invoke(
            app,
            ["upload", "--src", str(root), "--revision", "test", "--json"],
            env={**os.environ, "HF_TOKEN": "UNITTEST_FAKE_TOKEN_VALUE_ZZ"},
        )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert MSG_UPLOAD_AUTH in combined
    assert "UNIQUE_CLI_UPLOAD_AUTH_LEAK_99" not in combined
    assert creative not in combined
    assert "UNITTEST_FAKE_TOKEN_VALUE_ZZ" not in combined
    for pat in _SECRET_PATTERNS:
        assert not pat.search(combined)


def test_cli_upload_raw_hub_exc_remapped_to_constant(tmp_path: Path) -> None:
    """If a non-HfPacksError leaks, CLI still remaps via map_hub_failure."""
    root = _write_corpus(tmp_path / "src")
    creative = "401 Unauthorized UNIQUE_RAW_CLI_HUB_EXC_XYZ token=hf_cli_leak"

    def _boom(**_kwargs: Any) -> dict[str, Any]:
        raise _FakeHubAuthError(401, creative)

    with patch("swe_factory.export.hf_packs.upload_pack_tree", side_effect=_boom):
        result = runner.invoke(
            app,
            ["upload", "--src", str(root), "--revision", "test", "--json"],
            env={**os.environ, "HF_TOKEN": "UNITTEST_FAKE_TOKEN_RAW"},
        )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert MSG_UPLOAD_AUTH in combined
    assert "UNIQUE_RAW_CLI_HUB_EXC_XYZ" not in combined
    assert "hf_cli_leak" not in combined
    assert creative not in combined
    assert "UNITTEST_FAKE_TOKEN_RAW" not in combined


def test_cli_pull_auth_failure_uses_constant(tmp_path: Path) -> None:
    out = tmp_path / "cli_pull_auth"
    creative = "403 Forbidden UNIQUE_CLI_PULL_AUTH_LEAK_77 GatedRepo body"

    def _boom(**_kwargs: Any) -> dict[str, Any]:
        raise HfPacksError(MSG_PULL_AUTH)

    with patch("swe_factory.export.hf_packs.pull_pack_tree", side_effect=_boom):
        result = runner.invoke(
            app,
            ["pull", "--out", str(out), "--revision", "test", "--json"],
            env={**os.environ, "HF_TOKEN": "UNITTEST_FAKE_TOKEN_PULL"},
        )
    assert result.exit_code != 0
    combined = result.output + (result.stderr or "")
    assert MSG_PULL_AUTH in combined
    assert "UNIQUE_CLI_PULL_AUTH_LEAK_77" not in combined
    assert creative not in combined
    assert "UNITTEST_FAKE_TOKEN_PULL" not in combined


def test_token_missing_message_is_constant(tmp_path: Path) -> None:
    root = _write_corpus(tmp_path / "src")
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    with patch.dict(os.environ, env, clear=True), pytest.raises(HfPacksError) as ei:
        upload_packs(local_root=root, revision="test", dry_run=False, token=None)
    assert str(ei.value) == MSG_TOKEN_MISSING


# ---------------------------------------------------------------------------
# Optional live smoke (skipped without token)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_hf_upload_pull_smoke(tmp_path: Path) -> None:
    """Live smoke: upload one mini pack to revision test, pull back (when token)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not present")

    root = tmp_path / "live_src"
    task_id = "m16b-live-smoke-pack"
    _write_corpus(root, [task_id])
    # keep corpus small: only pack_manifest + one pack
    up = upload_packs(
        local_root=root,
        repo_id=DEFAULT_HF_REPO_ID,
        revision="test",
        token=token,
        dry_run=False,
        # allow our synthetic smoke label
        allow_empty=False,
    )
    assert up["ok"] is True and up["pushed"] is True
    assert "token" not in up

    out = tmp_path / "live_pull"
    down = pull_packs(
        out_dir=out,
        repo_id=DEFAULT_HF_REPO_ID,
        revision="test",
        token=token,
    )
    assert down["ok"] is True and down["pulled"] is True
    # May contain more packs if corpus already has others; our smoke id must be present
    assert task_id in down["task_ids"] or (out / "tasks" / task_id / "task.toml").is_file()
    assert (out / "tasks" / task_id / "environment").is_dir() or any(
        (p / "environment").is_dir() for p in (out / "tasks").iterdir() if p.is_dir()
    )
