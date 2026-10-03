"""Inert plan or explicitly preflight-bound, supervised one-shot study execution.

Do not invoke capture without the separately completed exact-head human receipt.
No init/next/resume or environment fallback exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="read-only prospective binding; no namespace or graphics")
    for name in ("capture", "_worker"):
        command = sub.add_parser(
            name, help="one-shot execution" if name == "capture" else argparse.SUPPRESS
        )
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--human-preflight", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "_worker":
        parent = os.getppid()
        if os.environ.get("EPS_TOPOLOGY_SUPERVISOR_PID") != str(parent):
            raise RuntimeError("internal worker requires its live resource supervisor")
        arguments = (Path("/proc") / str(parent) / "cmdline").read_bytes().split(b"\0")
        expected = str(Path(__file__).resolve()).encode()
        if expected not in arguments or b"capture" not in arguments:
            raise RuntimeError("internal worker parent is not the capture supervisor")
        # Resource limits precede NumPy, schema, producer and graphics imports.
        from epsbench.diagnostics.topology_eight_resources import child_apply_limits

        child_apply_limits()
        from epsbench.diagnostics.topology_eight_executor import worker

        result = worker(SOURCE, args.root, args.human_preflight)
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    from epsbench.diagnostics.topology_eight_executor import prepare_plan, validate_preflight

    if args.action == "plan":
        print(json.dumps(prepare_plan(SOURCE), allow_nan=False, sort_keys=True, indent=2))
        return 0
    from epsbench.diagnostics.topology_eight_resources import require_linux_capabilities, supervise

    require_linux_capabilities()
    validate_preflight(SOURCE, args.root, args.human_preflight)
    log = args.root.parent / (args.root.name + ".supervisor.json")
    if log.exists() or log.is_symlink():
        raise RuntimeError("supervisor receipt already exists; no replacement invocation")
    # Environment was checked, never silently repaired. Worker receives the same
    # declared runtime and imports this exact script/source, without module search.
    return supervise(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_worker",
            "--root",
            str(args.root),
            "--human-preflight",
            str(args.human_preflight),
        ],
        args.root,
        log,
    )


if __name__ == "__main__":
    # A new invocation never adopts an existing study root, including direct worker
    # invocation. Neither parser visibility nor an environment variable is authority.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(main())
