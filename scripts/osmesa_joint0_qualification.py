"""Initialise or inspect the bounded OSMesa joint0 candidate.

Execution is intentionally absent pending independent review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from epsbench.diagnostics.osmesa_joint0_qualification import (
    canonical,
    digest_bytes,
    plan,
)


def main() -> None:
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command", required=True)
    i = s.add_parser("init")
    i.add_argument("--output-root", type=Path, required=True)
    s.add_parser("plan")
    a = p.parse_args()
    if a.command == "plan":
        print(json.dumps(plan(), sort_keys=True))
        return
    out = a.output_root
    out.mkdir(parents=True, exist_ok=False)
    payload = plan()
    (out / "qualification-plan.json").write_bytes(canonical(payload))
    (out / "qualification-plan.sha256").write_text(digest_bytes(canonical(payload)) + "\n")
    print(
        json.dumps(
            {"status": "initialised", "plan": str(out / "qualification-plan.json")}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
