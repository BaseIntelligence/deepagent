"""One-shot gold-eval over the finalized snapshot (Headline A proof).

Runs each shipped task's evaluate.sh in Docker (2 fresh --rm runs each) and writes
the serialized GoldEvalReport to results/pilot_final/gold_eval.json so the report
build can consume it. Stand-alone finalize helper (not part of the harvest driver).
"""

from __future__ import annotations

import json
from pathlib import Path

from swe_forge.forge.gold_eval import run_gold_eval

OUT = Path("results/pilot_final")


def main() -> None:
    report = run_gold_eval(OUT, runs=2, name_prefix="swe-forge-goldeval")
    payload = report.to_dict()
    (OUT / "gold_eval.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("GOLD_EVAL_DONE")
    print("shipped:", report.shipped_count)
    print("gold_count:", report.gold_count)
    print("all_gold:", report.all_gold)
    print("deterministic:", report.deterministic)
    print("non_gold:", [r.task_id for r in report.non_gold])
    print("phase1_all:", [r.phase1_all for r in report.results])


if __name__ == "__main__":
    main()
