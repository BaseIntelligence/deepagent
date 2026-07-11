"""CLI persistence and validation for protected teacher transport receipts."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import typer

from swe_forge.forge.cli import _load_oracle_report, _write_oracle_report
from swe_forge.forge.models import OracleReport
from swe_forge.forge.teacher import Usage
from tests.test_forge.receipt_helpers import signed_transport_receipt


def _forged_teacher_report() -> OracleReport:
    return OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        details={
            "teacher_gates": {
                "differential": {
                    "calls": [
                        {
                            "gate": "differential",
                            "call_kind": "proposal",
                            "real_teacher": True,
                            "status": "success",
                            "response_kind": "proposals",
                            "model": "anthropic/test",
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                            "cost": 0.01,
                            "finish_reason": "stop",
                            "requested_proposals": 1,
                            "received_proposals": 1,
                            "parsed_proposals": 1,
                            "identical_proposals": 0,
                            "invalid_proposals": 0,
                            "discarded_proposals": 0,
                            "execution_attempted": 1,
                            "execution_completed": 1,
                            "execution_errors": 0,
                            "executable_proposals": 1,
                            "error_type": "",
                            "recovery_accounting": None,
                            "call_id": "f" * 32,
                            "receipt_commitment": "e" * 64,
                        }
                    ]
                }
            }
        },
    )


def test_cli_rejects_forged_public_teacher_evidence_without_receipt(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "oracle.json"
    report_path.write_text(
        json.dumps(_forged_teacher_report().to_dict()), encoding="utf-8"
    )

    with pytest.raises(typer.Exit):
        _load_oracle_report(str(report_path))


def test_cli_rejects_altered_signed_receipt_sidecar(tmp_path: Path) -> None:
    report = _forged_teacher_report()
    calls = report.details["teacher_gates"]["differential"]["calls"]  # type: ignore[index]
    assert isinstance(calls, list) and isinstance(calls[0], dict)
    call = calls[0]
    call["response_kind"] = "content"
    call["recovery_accounting"] = None
    receipt = signed_transport_receipt(
        call_id="f" * 32,
        candidate_fingerprint="a" * 64,
        gate="differential",
        call_kind="proposal",
        model="anthropic/test",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        cost=0.01,
    )
    call["call_id"] = receipt.call_id
    call["receipt_commitment"] = receipt.commitment
    report.protected_teacher_transport_receipts = [receipt.to_private_dict()]
    _write_oracle_report(str(tmp_path), report)

    sidecar = tmp_path / "oracle_report.transport-receipts.json"
    receipt_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    receipt_payload[0]["signature"] = "0" * 64
    sidecar.write_text(json.dumps(receipt_payload), encoding="utf-8")

    with pytest.raises(typer.Exit):
        _load_oracle_report(str(tmp_path / "oracle_report.json"))


def test_cli_writes_receipts_only_to_mode_0600_sidecar(tmp_path: Path) -> None:
    report = OracleReport(
        language="python",
        generator="ast_mutation",
        verdict="pass",
        protected_teacher_transport_receipts=[
            {
                "call_id": "f" * 32,
                "candidate_fingerprint": "a" * 64,
                "gate": "differential",
                "call_kind": "proposal",
                "model": "anthropic/test",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "cost": 0.01,
                "version": 2,
                "issuer_key_id": "a" * 32,
                "signature": "e" * 64,
            }
        ],
    )

    _write_oracle_report(str(tmp_path), report)

    public = tmp_path / "oracle_report.json"
    protected = tmp_path / "oracle_report.transport-receipts.json"
    assert "issuer_key_id" not in public.read_text(encoding="utf-8")
    assert '"issuer_key_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"' in protected.read_text(
        encoding="utf-8"
    )
    assert stat.S_IMODE(protected.stat().st_mode) == 0o600
