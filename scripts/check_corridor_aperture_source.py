"""Bounded public source check; deny native/model/producer imports before collection."""

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
    "epsbench.sim",
    "epsbench.data.generate",
    "epsbench.diagnostics.causal_history_fixture",
    "epsbench.diagnostics.occupancy_collection",
    "epsbench.diagnostics.occupancy_producer",
    "epsbench.diagnostics.restricted_learning_collection",
    "epsbench.diagnostics.restricted_models",
    "epsbench.diagnostics.restricted_health",
)


def forbidden(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if forbidden(fullname):
            raise RuntimeError("forbidden aperture source-check import: " + fullname)
        return None


def loaded() -> None:
    if any(forbidden(n) for n in sys.modules):
        raise RuntimeError("forbidden native/model/producer module already loaded")


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("no selector or arbitrary source-check argument permitted")
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
            "tests/test_corridor_aperture.py",
            "tests/test_corridor_aperture_camera.py",
            "tests/test_corridor_aperture_runtime.py",
            "tests/test_corridor_aperture_observer.py",
        ]
    )
    loaded()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
