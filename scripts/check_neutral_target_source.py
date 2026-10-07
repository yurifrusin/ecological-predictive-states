"""Only neutral-target handwritten CPU checks; native/model imports denied."""

from __future__ import annotations

import importlib.abc
import os
import sys
from pathlib import Path

FORBIDDEN = (
    "mujoco",
    "OpenGL",
    "glfw",
    "torch",
    "tensorflow",
    "jax",
    "transformers",
    "epsbench.sim",
    "epsbench.data.generate",
    "epsbench.annotations.boundary_events",
    "epsbench.annotations.optical_transport",
    "epsbench.diagnostics.restricted_models",
    "epsbench.diagnostics.restricted_health",
    "epsbench.diagnostics.occupancy_collection",
    "epsbench.diagnostics.occupancy_producer",
    "epsbench.diagnostics.restricted_learning_collection",
)


def forbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if forbidden(fullname):
            raise RuntimeError("forbidden neutral-target check import: " + fullname)
        return None


def loaded() -> None:
    if any(forbidden(name) for name in sys.modules):
        raise RuntimeError("native/model modules already loaded")


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("no extra source-check arguments admitted")
    loaded()
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
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            "tests/test_neutral_observation_target.py",
        ]
    )
    loaded()
    print("Handwritten target serialization/replay validation only; native/optical causes HELD")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
