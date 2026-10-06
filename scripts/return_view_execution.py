"""Explicit local launcher; requires a separately recorded exact-source decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from epsbench.diagnostics.return_view_execution import launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(launch(args.output, args.decision))


if __name__ == "__main__":
    main()
