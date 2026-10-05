"""Prospective fixed-study controller. Explicit exact-binding phase authorization required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "continuation"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--expected-history")
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    from epsbench.diagnostics.a1_execution import (
        DockerController,
        HostReceipts,
        initialize,
        source_binding,
    )

    binding = source_binding(SOURCE, args.image)
    authorization = json.loads(args.authorization.read_text())
    if (
        authorization.get("binding") != vars(binding)
        or authorization.get("archive_path") != str(args.archive.resolve())
        or authorization.get("receipts_path") != str(args.receipts.resolve())
        or authorization.get("phase") != args.phase
        or authorization.get("expected_history") != args.expected_history
        or authorization.get("phase_gate_effect") != "NONE"
        or not isinstance(authorization.get("decision"), str)
        or not authorization["decision"].strip()
        or not isinstance(authorization.get("dummy_qualification_sha256"), str)
        or len(authorization["dummy_qualification_sha256"]) != 64
    ):
        raise ValueError(
            "exact coordinator decision and prior dummy qualification receipt required"
        )
    from epsbench.diagnostics.a1_retention import _digest

    _digest(authorization["dummy_qualification_sha256"])
    initial = args.expected_history is None
    if initial and (args.archive.exists() or args.phase != "development"):
        raise ValueError("initial creation only for a missing archive in development phase")
    receipts = HostReceipts(args.receipts, binding, initial=initial)
    try:
        expected = initialize(args.archive, binding, receipts) if initial else args.expected_history
        result = DockerController(args.docker, args.archive, binding, receipts).phase(
            args.phase, authorization["decision"], expected
        )
        print(json.dumps({"history_sha256": result, "binding": vars(binding), "phase": args.phase}))
    finally:
        receipts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
