"""Inert A1 plan; capture is intentionally unavailable in this source candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "capture"))
    args = parser.parse_args()
    from epsbench.diagnostics.a1_action_contrast import CaptureHeldError, plan

    if args.action == "capture":
        raise CaptureHeldError("A1 native launch held; finite polling does not satisfy hard caps")
    print(json.dumps(plan(SOURCE), allow_nan=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
