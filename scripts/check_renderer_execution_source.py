"""Only guarded renderer adapter/core and old-domain source checks; no native or actual dummy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from check_a1_source import Guard, check_loaded


def main() -> int:
    if sys.argv[1:] not in ([], ["--adapter-only"]):
        raise ValueError("closed source-check selector")
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
        [
            "--noconftest",
            "-o",
            "addopts=",
            "tests/test_renderer_discriminator_execution.py",
            *(
                []
                if sys.argv[1:]
                else [
                    "tests/test_renderer_discriminator.py",
                    "tests/test_paired_appearance.py",
                    "tests/test_paired_appearance_execution.py",
                    "tests/test_appearance_contrast_calibration.py",
                ]
            ),
            "-q",
        ]
    )
    check_loaded()
    print("Fake commands and synthetic replay only; Docker/SDK/actual dummy HELD")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
