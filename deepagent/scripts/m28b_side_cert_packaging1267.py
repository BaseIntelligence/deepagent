#!/usr/bin/env python3
"""Side Docker re-cert for packaging-1267 with unit-only F2P (drop flaky hypothesis property)."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for k in list(os.environ):
    ku = k.upper()
    if ku in {"ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "OXYLABS_PROXY_URL"} or (
        ku.endswith("_PROXY") and not ku.startswith("GITHUB")
    ):
        os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"

from swe_factory.harbor.harbor_oracle import HarborDockerVerifier  # noqa: E402
from swe_factory.harbor.real_oracle_cert import certify_real_pr_pack  # noqa: E402


def main() -> int:
    src = Path("datasets/_m28b_work_cert4/gen_work/staging/realpr-packaging-1267")
    if not src.is_dir():
        print("missing staging packaging pack", src)
        return 2
    dual_path = Path("datasets/_m28b_evidence2/realpr-packaging-1267.dual.json")
    dual = json.loads(dual_path.read_text())
    f2p_all = list(dual.get("f2p") or [])
    unit = [
        n
        for n in f2p_all
        if "property" not in n.lower() and "hypothesis" not in n.lower()
    ]
    if len(unit) < 5:
        print("unit f2p too thin", len(unit))
        return 2
    print(f"f2p unit keep {len(unit)} of {len(f2p_all)}")

    out_root = Path("datasets/_m28b_work_cert4/side_packaging1267")
    if out_root.exists():
        shutil.rmtree(out_root)
    pack = out_root / "realpr-packaging-1267"
    shutil.copytree(src, pack)

    cfg_path = pack / "tests" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["f2p_node_ids"] = unit
    # Empty P2P: binary reward = (f2p>0 and ff==0 and pf==0); empty p2p => pf==0.
    cfg["p2p_node_ids"] = []
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    backend = HarborDockerVerifier(run_id="m28b_packaging1267_side")
    result = certify_real_pr_pack(
        pack_dir=pack,
        backend=backend,
        dest_hint="datasets/_m28b_work_cert4/side_packaging1267_out",
        evidence_dir=out_root / "evidence",
        cleanup=True,
        require_real_pr_track=True,
    )
    d = result.to_dict() if hasattr(result, "to_dict") else dict(vars(result))
    Path("datasets/_m28b_evidence2/packaging1267_side_cert.json").write_text(
        json.dumps(d, indent=2, default=str)
    )
    print(json.dumps(d, indent=2, default=str)[:4000])
    sol = d.get("solution_reward")
    null = d.get("null_reward")
    cert = d.get("certified")
    print("FINAL certified", cert, "sol", sol, "null", null)
    return 0 if cert and sol == 1 and null == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
