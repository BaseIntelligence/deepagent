"""One-shot gold-eval over an exported pilot set (Headline A proof).

Runs each shipped task's evaluate.sh in Docker (2 fresh --rm runs each) and writes
the serialized GoldEvalReport beside that set so the report build can consume it.
The canonical pilot output is the default; an explicit snapshot directory remains
available for ad-hoc verification. Stand-alone finalize helper, not part of the
harvest driver.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from swe_forge.forge.gold_eval import run_gold_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path("results/pilot_keeps")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Export directory containing tasks/ (default: results/pilot_keeps).",
    )
    return parser.parse_args(argv)


def resolve_out_dir(out_dir: Path) -> Path:
    """Resolve the canonical default against this repository, not the caller CWD."""
    return REPO_ROOT / out_dir if out_dir == DEFAULT_OUT_DIR else out_dir


def display_tasks_dir(tasks_dir: Path) -> str:
    """Keep canonical reports portable by storing a repository-relative path."""
    try:
        return str(tasks_dir.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(tasks_dir)


def main(argv: Sequence[str] | None = None) -> None:
    out_dir = resolve_out_dir(parse_args(argv).out_dir)
    report = run_gold_eval(out_dir, runs=2, name_prefix="swe-forge-goldeval")
    payload = report.to_dict()
    payload["tasks_dir"] = display_tasks_dir(report.tasks_dir)
    (out_dir / "gold_eval.json").write_text(
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
