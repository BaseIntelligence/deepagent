"""Adversarial tests for the isolated teacher transport receipt authority."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from swe_forge.forge import receipt_authority
from swe_forge.forge import receipt_authority_service
from swe_forge.forge import teacher as teacher_module
from swe_forge.forge.teacher import (
    LLMResult,
    TeacherClient,
    TransportReceipt,
    Usage,
    transport_receipt_context,
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
    assert verify_transport_receipt(receipt)

    root_payload = json.loads(
        (isolated_test_authority / "authority-v1.json").read_text(encoding="utf-8")
    )
    assert set(root_payload) == {
        "algorithm",
        "environment",
        "key_id",
        "public_key",
        "version",
    }
    assert root_payload["environment"] == "test"
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
from swe_forge.forge.teacher import TransportReceipt, verify_transport_receipt
receipt_authority.default_authority_root = lambda: Path(sys.argv[1])
receipt = TransportReceipt.from_private_dict(json.loads(sys.argv[2]))
raise SystemExit(0 if verify_transport_receipt(receipt) else 1)
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
