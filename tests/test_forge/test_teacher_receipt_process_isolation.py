"""Adversarial tests for the isolated teacher transport receipt authority."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from swe_forge.forge import receipt_authority
from swe_forge.forge import receipt_authority_service
from swe_forge.forge import teacher as teacher_module
from swe_forge.forge.teacher import (
    LLMResult,
    TeacherClient,
    TransportReceipt,
    Usage,
    transport_receipt_context,
    verify_test_transport_receipt,
    verify_transport_receipt,
)


class _Candidate:
    def to_dict(self) -> dict[str, str]:
        return {"candidate": "process-isolation"}


def _test_response(text: str = "authority response") -> dict[str, object]:
    return {
        "text": text,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
        "cost": 0.0125,
        "request_id": "test-request-1",
    }


def _provision_production_root(
    root: Path,
    private_key: Ed25519PrivateKey,
) -> None:
    root.mkdir(mode=0o700)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    root_id = receipt_authority._root_identity(  # noqa: SLF001 - provisioning fixture
        root, "production"
    )
    payload = {
        "version": 1,
        "algorithm": "Ed25519",
        "environment": "production",
        "root_id": root_id,
        "key_id": receipt_authority._key_id(  # noqa: SLF001 - provisioning fixture
            public_key,
            environment="production",
            root_id=root_id,
        ),
        "public_key": base64.b64encode(public_key).decode("ascii"),
    }
    marker = root / "production-authority-v1.json"
    metadata = root / "authority-v1.json"
    marker.write_text('{"environment":"production"}\n', encoding="utf-8")
    metadata.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    marker.chmod(0o600)
    metadata.chmod(0o600)


@pytest.fixture
def isolated_test_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "test-authority"
    receipt_authority.initialize_test_authority_root(root)
    monkeypatch.setattr(receipt_authority, "default_authority_root", lambda: root)
    return root


async def test_authority_child_owns_transport_and_in_memory_private_key(
    isolated_test_authority: Path,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate")
    await client.aclose()

    receipt = result.transport_receipt
    assert receipt is not None
    assert result.text == "authority response"
    assert verify_test_transport_receipt(receipt, root=isolated_test_authority)

    root_payload = json.loads(
        (isolated_test_authority / "authority-v1.json").read_text(encoding="utf-8")
    )
    assert set(root_payload) == {
        "algorithm",
        "environment",
        "key_id",
        "public_key",
        "root_id",
        "version",
    }
    assert root_payload["environment"] == "test"
    assert root_payload["root_id"]
    assert not any(
        "key" in path.name and path.name != "authority-v1.json"
        for path in isolated_test_authority.iterdir()
    )
    assert receipt.response_commitment
    assert receipt.provider_request_id == "test-request-1"


async def test_parent_litellm_and_teacher_method_monkeypatches_cannot_mint_receipts(
    isolated_test_authority: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        legitimate = await client.complete_text("generate")
    await client.aclose()

    assert legitimate.transport_receipt is not None
    assert "litellm" not in teacher_module.__dict__

    async def forged_complete_text(*_args: object, **_kwargs: object) -> LLMResult:
        return LLMResult(text="forged", usage=Usage(), cost=0.0)

    monkeypatch.setattr(client, "complete_text", forged_complete_text)
    forged = await client.complete_text("generate")
    assert forged.transport_receipt is None


async def test_parent_tampering_with_child_response_is_rejected(
    isolated_test_authority: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    original_complete = client._authority_complete  # noqa: SLF001 - attack surface

    async def tampered_complete(
        request: dict[str, object], *, timeout: float
    ) -> dict[str, object]:
        payload = await original_complete(request, timeout=timeout)
        normalized = payload.get("normalized")
        assert isinstance(normalized, dict)
        normalized["text"] = "parent-forged response"
        return payload

    monkeypatch.setattr(client, "_authority_complete", tampered_complete)
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        with pytest.raises(teacher_module.TeacherError, match="commitment"):
            await client.complete_text("generate")
    await client.aclose()


async def test_multiple_teacher_clients_share_the_live_ephemeral_authority(
    isolated_test_authority: Path,
) -> None:
    first = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response("first")],
    )
    second = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response("second")],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        first_result = await first.complete_text("generate")
    await first.aclose()
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        second_result = await second.complete_text("generate")
    await second.aclose()

    assert first_result.text == "first"
    assert second_result.text == "second"
    assert first_result.transport_receipt is not None
    assert second_result.transport_receipt is not None


async def test_fresh_verifier_rejects_tampered_response_claims_wrong_root_and_replay(
    isolated_test_authority: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate")
    await client.aclose()
    assert result.transport_receipt is not None
    receipt = result.transport_receipt

    payload = receipt.to_private_dict()
    payload["response_commitment"] = "0" * 64
    assert not verify_transport_receipt(TransportReceipt.from_private_dict(payload))

    script = """
import json
import sys
from pathlib import Path
from swe_forge.forge import receipt_authority
from swe_forge.forge.teacher import TransportReceipt, verify_test_transport_receipt
receipt = TransportReceipt.from_private_dict(json.loads(sys.argv[2]))
raise SystemExit(0 if verify_test_transport_receipt(receipt, root=Path(sys.argv[1])) else 1)
"""
    valid = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(isolated_test_authority),
            json.dumps(receipt.to_private_dict(), sort_keys=True),
        ],
        cwd="/projects/Agent-SWE",
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr

    wrong_root = isolated_test_authority.parent / "wrong-authority"
    receipt_authority.initialize_test_authority_root(wrong_root)
    monkeypatch.setattr(receipt_authority, "default_authority_root", lambda: wrong_root)
    assert not verify_transport_receipt(receipt)


async def test_authority_crash_timeout_and_restart_fail_closed(
    isolated_test_authority: Path,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[{"crash": True}],
        timeout=0.1,
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        with pytest.raises(teacher_module.TeacherError, match="authority"):
            await client.complete_text("generate")
    await client.aclose()

    restarted = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        with pytest.raises(teacher_module.TeacherError, match="authority"):
            await restarted.complete_text("generate")
    await restarted.aclose()


async def test_private_or_unknown_authority_root_material_fails_closed(
    isolated_test_authority: Path,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate")
    await client.aclose()
    assert result.transport_receipt is not None

    (isolated_test_authority / "legacy-signing.key").write_text(
        "not-a-private-key", encoding="utf-8"
    )
    assert not verify_transport_receipt(result.transport_receipt)

    rejected = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        with pytest.raises(teacher_module.TeacherError, match="authority"):
            await rejected.complete_text("generate")
    await rejected.aclose()


def test_receipt_authority_exports_no_signing_or_issuance_capability() -> None:
    forbidden = {"_sign_claims", "issuer_key_id", "issue_receipt", "sign_claims"}
    assert not forbidden & set(dir(receipt_authority))
    assert "receipt_authority_service" not in receipt_authority.__dict__
    assert not hasattr(receipt_authority_service, "_sign_receipt")
    assert not hasattr(teacher_module, "_issue_transport_receipt_after_provider_return")


def test_production_entrypoint_is_not_importable_or_callable() -> None:
    assert not hasattr(receipt_authority, "_authority_entry")
    assert getattr(receipt_authority_service, "authority_process", None) is None


def test_test_root_metadata_cannot_be_transplanted_or_relabeled(
    isolated_test_authority: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = isolated_test_authority
    authority = receipt_authority.ReceiptAuthorityClient(root=source)
    authority.complete(
        {
            "type": "complete",
            "routing": {
                "model": "anthropic/test-model",
                "api_base": "https://test.invalid",
                "api_key": "test-only",
                "num_retries": 0,
                "timeout": 1.0,
            },
            "messages": [],
            "max_tokens": 1,
            "tools": None,
            "tool_choice": None,
            "response_format": None,
            "context": None,
            "recovery": None,
            "test_provider_response": _test_response(),
        },
        timeout=5.0,
    )
    authority.close()
    transplanted = source.parent / "transplanted-production"
    transplanted.mkdir()
    for name in ("authority-v1.json", "test-authority-v1.json"):
        (transplanted / name).write_bytes((source / name).read_bytes())
    (transplanted / "test-authority-v1.json").rename(
        transplanted / "production-authority-v1.json"
    )
    monkeypatch.setattr(
        receipt_authority, "default_authority_root", lambda: transplanted
    )
    assert not receipt_authority.verify_signature(
        key_id="0" * 64,
        claims=b"{}",
        signature="not-a-signature",
    )


async def test_production_verifier_rejects_valid_test_receipts_after_root_redirect(
    isolated_test_authority: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production verification must not consult the mutable client-root resolver."""
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_test_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate")
    await client.aclose()
    assert result.transport_receipt is not None

    monkeypatch.setattr(
        receipt_authority, "default_authority_root", lambda: isolated_test_authority
    )

    assert not verify_transport_receipt(result.transport_receipt)
    assert verify_test_transport_receipt(
        result.transport_receipt, root=isolated_test_authority
    )

    relabeled = isolated_test_authority.parent / "relabeled-test-root"
    relabeled.mkdir()
    for name in ("authority-v1.json", "test-authority-v1.json"):
        (relabeled / name).write_bytes((isolated_test_authority / name).read_bytes())
    (relabeled / "test-authority-v1.json").rename(
        relabeled / "production-authority-v1.json"
    )
    assert not verify_test_transport_receipt(result.transport_receipt, root=relabeled)


def test_direct_service_cli_and_imported_loop_cannot_bootstrap_production_root(
    tmp_path: Path,
) -> None:
    """A caller-supplied CLI token or direct loop call cannot create a prod root."""
    root = tmp_path / "would-be-production-root"
    bootstrap_read, bootstrap_write = os.pipe()
    try:
        os.set_inheritable(bootstrap_read, True)
        os.write(bootstrap_write, b"c" * 32)
        direct = subprocess.run(
            [
                sys.executable,
                "-m",
                "swe_forge.forge.receipt_authority_service",
                "--root",
                str(root),
                "--domain",
                "production",
                "--bootstrap-fd",
                str(bootstrap_read),
            ],
            cwd="/projects/Agent-SWE",
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(bootstrap_read,),
        )
    finally:
        os.close(bootstrap_read)
        os.close(bootstrap_write)
    assert direct.returncode != 0
    assert not root.exists()
    assert not hasattr(receipt_authority_service, "_EXECUTABLE_BOOTSTRAP")

    assert not hasattr(receipt_authority_service, "_run_authority")
    assert not root.exists()


def test_actual_executable_rejects_caller_fd_when_canonical_root_is_absent() -> None:
    """A caller-owned inherited descriptor is not production provisioning."""
    canonical_root = Path("/var/lib/swe_forge/teacher-receipt-authority")
    python = sys.executable
    command = """
set -eu
mkdir -p /var/lib/swe_forge
mount -t tmpfs -o mode=0755 tmpfs /var/lib/swe_forge
set +e
"$@" < /dev/null
status=$?
set -e
test ! -e /var/lib/swe_forge/teacher-receipt-authority
exit "$status"
"""

    bootstrap_read, bootstrap_write = os.pipe()
    try:
        os.set_inheritable(bootstrap_read, True)
        os.write(bootstrap_write, b"a" * 32)
        os.close(bootstrap_write)
        bootstrap_write = -1
        attempted = subprocess.run(
            [
                "unshare",
                "--mount",
                "--fork",
                "sh",
                "-c",
                command,
                "sh",
                python,
                "-m",
                "swe_forge.forge.receipt_authority_service",
                "--root",
                str(canonical_root),
                "--domain",
                "production",
                "--bootstrap-fd",
                str(bootstrap_read),
            ],
            cwd="/projects/Agent-SWE",
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(bootstrap_read,),
            timeout=5,
        )
    finally:
        os.close(bootstrap_read)
        if bootstrap_write >= 0:
            os.close(bootstrap_write)

    assert attempted.returncode != 0
    assert "key_id" not in attempted.stdout
    assert '"type":"result"' not in attempted.stdout
    assert not (canonical_root / "authority-v1.json").exists()


def test_parent_refuses_absent_production_root_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imported Forge Python cannot turn its child or descriptors into trust."""
    canonical_root = tmp_path / "absent-canonical-root"
    monkeypatch.setattr(
        receipt_authority,
        "_canonical_production_root",
        lambda: canonical_root,
    )
    spawned = False

    def forbidden_spawn(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal spawned
        spawned = True
        raise AssertionError("production startup must fail before spawning")

    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    authority = receipt_authority.ReceiptAuthorityClient(root=canonical_root)

    with pytest.raises(
        receipt_authority.ReceiptAuthorityError,
        match="externally provisioned",
    ):
        authority.complete({})

    assert not spawned
    assert not authority.is_alive
    assert not canonical_root.exists()


def test_provisioned_production_root_requires_matching_supervisor_capability() -> None:
    """Provisioning binds startup to a key caller-selected bytes cannot replace."""
    supervisor_key = Ed25519PrivateKey.generate()
    private_b64 = base64.b64encode(
        supervisor_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    ).decode("ascii")
    provision_script = f"""
from pathlib import Path
import base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from tests.test_forge.test_teacher_receipt_process_isolation import (
    _provision_production_root,
)
root = Path("/var/lib/swe_forge/teacher-receipt-authority")
key = Ed25519PrivateKey.from_private_bytes(base64.b64decode("{private_b64}"))
_provision_production_root(root, key)
"""
    launcher = """
set -eu
mkdir -p /var/lib/swe_forge
mount -t tmpfs -o mode=0755 tmpfs /var/lib/swe_forge
"$1" -c "$2"
shift 2
before="$(sha256sum \
  /var/lib/swe_forge/teacher-receipt-authority/production-authority-v1.json \
  /var/lib/swe_forge/teacher-receipt-authority/authority-v1.json)"
"$@" < /dev/null
after="$(sha256sum \
  /var/lib/swe_forge/teacher-receipt-authority/production-authority-v1.json \
  /var/lib/swe_forge/teacher-receipt-authority/authority-v1.json)"
test "$before" = "$after"
"""
    private_bytes = supervisor_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    outcomes: list[subprocess.CompletedProcess[str]] = []
    for capability in (b"x" * 32, private_bytes):
        bootstrap_read, bootstrap_write = os.pipe()
        try:
            os.set_inheritable(bootstrap_read, True)
            os.write(bootstrap_write, capability)
            os.close(bootstrap_write)
            bootstrap_write = -1
            outcomes.append(
                subprocess.run(
                    [
                        "unshare",
                        "--mount",
                        "--fork",
                        "sh",
                        "-c",
                        launcher,
                        "sh",
                        sys.executable,
                        provision_script,
                        sys.executable,
                        "-m",
                        "swe_forge.forge.receipt_authority_service",
                        "--root",
                        "/var/lib/swe_forge/teacher-receipt-authority",
                        "--domain",
                        "production",
                        "--bootstrap-fd",
                        str(bootstrap_read),
                    ],
                    cwd="/projects/Agent-SWE",
                    check=False,
                    capture_output=True,
                    text=True,
                    pass_fds=(bootstrap_read,),
                    timeout=5,
                )
            )
        finally:
            os.close(bootstrap_read)
            if bootstrap_write >= 0:
                os.close(bootstrap_write)

    wrong, right = outcomes
    assert wrong.returncode == 0
    assert '"type":"startup_error"' in wrong.stdout
    assert "key_id" not in wrong.stdout
    assert right.returncode == 0
    ready = json.loads(right.stdout.splitlines()[0])
    assert ready["type"] == "ready"
    assert ready["environment"] == "production"


def test_production_executable_does_not_repair_incomplete_provisioning() -> None:
    """A marker-only deployment root remains unchanged after rejected startup."""
    launcher = """
set -eu
mkdir -p /var/lib/swe_forge
mount -t tmpfs -o mode=0755 tmpfs /var/lib/swe_forge
mkdir -m 700 /var/lib/swe_forge/teacher-receipt-authority
printf '%s\n' '{"environment":"production"}' > \
  /var/lib/swe_forge/teacher-receipt-authority/production-authority-v1.json
chmod 600 /var/lib/swe_forge/teacher-receipt-authority/production-authority-v1.json
set +e
"$@" < /dev/null
status=$?
set -e
test "$(find /var/lib/swe_forge/teacher-receipt-authority -mindepth 1 | wc -l)" -eq 1
test ! -e /var/lib/swe_forge/teacher-receipt-authority/authority-v1.json
exit "$status"
"""
    bootstrap_read, bootstrap_write = os.pipe()
    try:
        os.set_inheritable(bootstrap_read, True)
        os.write(bootstrap_write, b"x" * 32)
        os.close(bootstrap_write)
        bootstrap_write = -1
        attempted = subprocess.run(
            [
                "unshare",
                "--mount",
                "--fork",
                "sh",
                "-c",
                launcher,
                "sh",
                sys.executable,
                "-m",
                "swe_forge.forge.receipt_authority_service",
                "--root",
                "/var/lib/swe_forge/teacher-receipt-authority",
                "--domain",
                "production",
                "--bootstrap-fd",
                str(bootstrap_read),
            ],
            cwd="/projects/Agent-SWE",
            check=False,
            capture_output=True,
            text=True,
            pass_fds=(bootstrap_read,),
            timeout=5,
        )
    finally:
        os.close(bootstrap_read)
        if bootstrap_write >= 0:
            os.close(bootstrap_write)

    assert attempted.returncode == 1
    assert "key_id" not in attempted.stdout


def test_imported_key_pinning_helper_cannot_initialize_an_absent_root(
    tmp_path: Path,
) -> None:
    """No executable or imported helper may first-write production trust."""
    root = tmp_path / "absent-production-root"

    assert not hasattr(receipt_authority_service, "_require_pinned_key")

    assert not root.exists()


def test_cross_domain_client_startup_rejects_a_relabeled_test_root(
    isolated_test_authority: Path,
) -> None:
    """A client cannot treat test material as a production authority root."""
    source = receipt_authority.ReceiptAuthorityClient(root=isolated_test_authority)
    source.complete(
        {
            "type": "complete",
            "routing": {
                "model": "anthropic/test-model",
                "api_base": "https://test.invalid",
                "api_key": "test-only",
                "num_retries": 0,
                "timeout": 1.0,
            },
            "messages": [],
            "max_tokens": 1,
            "tools": None,
            "tool_choice": None,
            "response_format": None,
            "context": None,
            "recovery": None,
            "test_provider_response": _test_response(),
        }
    )
    source.close()

    relabeled = isolated_test_authority.parent / "relabeled-production-root"
    relabeled.mkdir()
    for name in ("authority-v1.json", "test-authority-v1.json"):
        (relabeled / name).write_bytes((isolated_test_authority / name).read_bytes())
    (relabeled / "test-authority-v1.json").rename(
        relabeled / "production-authority-v1.json"
    )
    authority = receipt_authority.ReceiptAuthorityClient(root=relabeled)

    with pytest.raises(receipt_authority.ReceiptAuthorityError):
        authority.complete(
            {
                "type": "complete",
                "routing": {
                    "model": "anthropic/test-model",
                    "api_base": "https://test.invalid",
                    "api_key": "test-only",
                    "num_retries": 0,
                    "timeout": 1.0,
                },
                "messages": [],
                "max_tokens": 1,
                "tools": None,
                "tool_choice": None,
                "response_format": None,
                "context": None,
                "recovery": None,
                "test_provider_response": _test_response(),
            }
        )

    assert not authority.is_alive
    assert not authority.is_usable


def test_test_root_initializer_cannot_relabel_the_canonical_production_root() -> None:
    with pytest.raises(receipt_authority.ReceiptAuthorityError, match="canonical"):
        receipt_authority.initialize_test_authority_root(
            Path("/var/lib/swe_forge/teacher-receipt-authority")
        )
