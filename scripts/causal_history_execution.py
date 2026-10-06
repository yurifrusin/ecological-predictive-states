"""Held host launcher: exact source-bound external decision, one task, no automatic retry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--purpose", choices=("causal_history_native_v1", "causal_history_dummy_v1"), required=True
    )
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-history")
    parser.add_argument("--expected-receipt-history")
    args = parser.parse_args()
    from epsbench.diagnostics.causal_history_execution import (
        DockerController,
        HostReceipts,
        initialize,
        source_binding,
    )
    from epsbench.diagnostics.causal_history_sequence import canonical, parse

    binding = source_binding(
        SOURCE,
        args.image,
        args.purpose,
        args.task if args.purpose == "causal_history_dummy_v1" else None,
    )
    with args.authorization.open("rb") as stream:
        auth = parse(stream.read(16385), 16384)
    required = {
        "binding": vars(binding),
        "task": args.task,
        "archive": str(args.archive.resolve()),
        "receipts": str(args.receipts.resolve()),
        "expected_history": args.expected_history,
        "expected_receipt_history": args.expected_receipt_history,
        "phase_gate_effect": "NONE",
        "decision": auth.get("decision"),
        "native_admission": auth.get("native_admission"),
        "dummy_qualification_sha256": auth.get("dummy_qualification_sha256"),
    }
    if auth != required or type(auth["decision"]) is not str or not auth["decision"].strip():
        raise ValueError("exact externally recorded source/task/root decision required")
    if args.purpose == "causal_history_native_v1":
        import re

        if (
            auth["native_admission"] is not True
            or type(auth["dummy_qualification_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", auth["dummy_qualification_sha256"]) is None
        ):
            raise ValueError("separate source/image/numeric and dummy admission required")
    initial = args.expected_history is None
    if initial != (args.expected_receipt_history is None):
        raise ValueError("both exact external witnesses required")
    receipts = HostReceipts(args.receipts, binding, args.archive, initial=initial)
    try:
        expected = initialize(args.archive, binding, receipts) if initial else args.expected_history
        root = receipts.root if initial else args.expected_receipt_history
        assert expected is not None and root is not None
        result = DockerController(args.docker, args.archive, binding, receipts).launch(
            args.task, expected, root
        )
        print(
            canonical(
                {
                    "history": result,
                    "receipt_history": receipts.root,
                    "binding": vars(binding),
                    "task": args.task,
                }
            ).decode()
        )
    finally:
        receipts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
