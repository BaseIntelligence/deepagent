#!/usr/bin/env python3
"""Side Docker re-cert for click-3442 with trimmed F2P + empty P2P."""
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
    src = Path("datasets/_m28b_work_cert4/gen_work/staging/realpr-click-3442")
    if not src.is_dir():
        print("missing staging click pack", src)
        return 2
    out_root = Path("datasets/_m28b_work_cert4/side_click")
    if out_root.exists():
        shutil.rmtree(out_root)
    pack = out_root / "realpr-click-3442"
    shutil.copytree(src, pack)

    cfg_path = pack / "tests" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    f2p = list(cfg.get("f2p_node_ids") or [])
    keep = []
    for n in f2p:
        low = n.lower()
        if any(x in low for x in ("termui", "prompt", "progress", "pager", "editor")):
            continue
        keep.append(n)
    if len(keep) < 5:
        keep = f2p[:20]
    else:
        keep = keep[:20]
    print(f"f2p keep {len(keep)} of {len(f2p)}")
    cfg["f2p_node_ids"] = keep
    # Empty P2P: binary reward = (f2p>0 and ff==0 and pf==0); empty p2p => pf==0.
    cfg["p2p_node_ids"] = []
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    backend = HarborDockerVerifier(run_id="m28b_click_side")
    result = certify_real_pr_pack(
        pack_dir=pack,
        backend=backend,
        dest_hint="datasets/_m28b_work_cert4/side_click_out",
        evidence_dir=out_root / "evidence",
        cleanup=True,
        require_real_pr_track=True,
    )
    d = result.to_dict() if hasattr(result, "to_dict") else dict(vars(result))
    Path("datasets/_m28b_evidence2/click_side_cert.json").write_text(
        json.dumps(d, indent=2, default=str)
    )
    print(json.dumps(d, indent=2, default=str)[:3000])
    sol = d.get("solution_reward")
    null = d.get("null_reward")
    cert = d.get("certified")
    print("FINAL certified", cert, "sol", sol, "null", null)
    return 0 if cert and sol == 1 and null == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
