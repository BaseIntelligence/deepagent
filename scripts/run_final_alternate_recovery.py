"""Run the single authorized final alternate recovery attempt."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from swe_forge.forge.alternate_recovery import run_final_alternate_recovery


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "pilot_final",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=REPO_ROOT / "results" / "final_alternate_recovery",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(
        run_final_alternate_recovery(
            out_dir=args.out_dir,
            work_root=args.work_root,
            repository_root=REPO_ROOT,
        )
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "reason": result.reason,
                "task_id": result.task_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
