"""CLI persistence and validation for protected teacher transport receipts."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import typer

from swe_forge.forge.cli import _load_oracle_report, _write_oracle_report
from swe_forge.forge.models import OracleReport


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
                "receipt_secret": "sensitive-receipt-secret",
                "commitment": "e" * 64,
            }
        ],
    )

    _write_oracle_report(str(tmp_path), report)

    public = tmp_path / "oracle_report.json"
    protected = tmp_path / "oracle_report.transport-receipts.json"
    assert "sensitive-receipt-secret" not in public.read_text(encoding="utf-8")
    assert "sensitive-receipt-secret" in protected.read_text(encoding="utf-8")
    assert stat.S_IMODE(protected.stat().st_mode) == 0o600
