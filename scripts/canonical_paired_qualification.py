"""Operate one reviewed finite canonical paired qualification namespace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from epsbench.diagnostics.canonical_paired_qualification import (
    CanonicalQualificationFailure,
    bind_pristine_sdk,
    fixed_attempts,
    git_binding,
    initialise_ledger,
    plan,
    require_supported_runtime,
    run_attempt,
    validate_ledger,
    validate_saved,
    validate_source_linkage,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    for name in ("init", "next", "validate"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(), sort_keys=True))
        return
    source = validate_source_linkage(args.source_root.resolve())
    output = args.output_root.absolute()
    if output == source or source in output.parents:
        raise CanonicalQualificationFailure(
            "qualification output must be outside the source checkout"
        )
    if args.command == "init":
        require_supported_runtime(os.environ.get("MUJOCO_GL", ""))
        import mujoco

        binding = git_binding(source)
        runtime = bind_pristine_sdk(mujoco)
        path = initialise_ledger(output, binding, runtime)
        print(json.dumps({"status": "initialised", "ledger": str(path)}, sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate_saved(output, source), sort_keys=True))
    else:
        records = validate_ledger(output)
        completed = sum(record["event"] == "complete" for record in records)
        if completed >= len(fixed_attempts()):
            raise CanonicalQualificationFailure(
                "canonical paired qualification is already complete"
            )
        attempt = fixed_attempts()[completed]
        path = run_attempt(output, source, attempt)
        print(
            json.dumps(
                {"status": "complete", "attempt": attempt.name, "ledger": str(path)}, sort_keys=True
            )
        )


if __name__ == "__main__":
    main()
