"""Tests for the public-root verifier and isolated receipt child."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path


from swe_forge.forge import receipt_authority
from swe_forge.forge.teacher import (
    TeacherClient,
    TransportReceipt,
    transport_receipt_context,
    verify_transport_receipt,
)


class _Candidate:
    def to_dict(self) -> dict[str, str]:
        return {"candidate": "receipt-authority"}


def _response() -> dict[str, object]:
    return {
        "text": "receipt",
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "cost": 0.0125,
        "request_id": "test-authority-request",
    }


async def test_root_contains_only_public_metadata_and_stays_stable_for_child(
    isolated_teacher_receipt_authority: Path,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate a proposal")
    await client.aclose()

    assert result.transport_receipt is not None
    root = isolated_teacher_receipt_authority
    metadata = json.loads((root / "authority-v1.json").read_text(encoding="utf-8"))
    assert set(metadata) == {
        "algorithm",
        "environment",
        "key_id",
        "public_key",
        "root_id",
        "version",
    }
    assert metadata["algorithm"] == "Ed25519"
    assert metadata["environment"] == "test"
    assert metadata["root_id"]
    assert stat.S_IMODE((root / "authority-v1.json").stat().st_mode) == 0o600
    assert not (root / "issuer-v1.key").exists()
    assert verify_transport_receipt(result.transport_receipt)


async def test_receipt_reloads_in_fresh_verifier_process(
    isolated_teacher_receipt_authority: Path,
) -> None:
    client = TeacherClient(
        base_url="https://teacher.test",
        api_key="sk-test",
        model="anthropic/test-model",
        authority_test_responses=[_response()],
    )
    with transport_receipt_context(
        _Candidate(), gate="differential", call_kind="proposal"
    ):
        result = await client.complete_text("generate a proposal")
    await client.aclose()
    assert result.transport_receipt is not None

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
    restarted = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(isolated_teacher_receipt_authority),
            json.dumps(result.transport_receipt.to_private_dict(), sort_keys=True),
        ],
        cwd="/projects/Agent-SWE",
        check=False,
        capture_output=True,
        text=True,
    )
    assert restarted.returncode == 0, restarted.stderr


def test_forged_wrong_root_or_altered_receipts_reject(
    isolated_teacher_receipt_authority: Path,
) -> None:
    # There is no parent-side issuer. A matching object cannot pass verification.
    forged = TransportReceipt(
        call_id="f" * 32,
        candidate_fingerprint="a" * 64,
        gate="differential",
        call_kind="proposal",
        model="anthropic/test-model",
        usage=__import__("swe_forge.forge.teacher", fromlist=["Usage"]).Usage(3, 2, 5),
        cost=0.0125,
        provider_request_id="forged-request",
        response_commitment="b" * 64,
        ledger_linkage="not_applicable",
        issuer_key_id="0" * 64,
        authority_domain="test",
        authority_root_id="0" * 64,
        signature="not-a-signature",
    )
    assert not verify_transport_receipt(forged)

    wrong_root = isolated_teacher_receipt_authority.parent / "wrong"
    receipt_authority.initialize_test_authority_root(wrong_root)
    assert not receipt_authority.verify_signature(
        key_id="0" * 64, claims=b"{}", signature="not-a-signature"
    )


def test_parent_exposes_no_production_signing_or_key_loader() -> None:
    assert not {
        "_sign_claims",
        "issuer_key_id",
        "_load_or_create_key",
        "_load_existing_key",
    } & set(dir(receipt_authority))
