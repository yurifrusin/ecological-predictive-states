"""Only synthetic paired-appearance source contracts; native imports denied before collection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from check_a1_source import Guard, check_loaded


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("no selector arguments accepted")
    check_loaded()
    sys.meta_path.insert(0, Guard())
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    import pytest

    result = pytest.main(
        ["--noconftest", "-o", "addopts=", "tests/test_paired_appearance.py", "-q"]
    )
    check_loaded()
    print("Synthetic serialization/validation/inspection and source asset checks only; native HELD")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
