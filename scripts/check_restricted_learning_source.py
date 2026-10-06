"""Guarded synthetic WP1 smoke/validation/inspection; actual collection held."""

from __future__ import annotations

import importlib
import importlib.abc
import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

FORBIDDEN = (
    "mujoco",
    "OpenGL",
    "glfw",
    "torch",
    "tensorflow",
    "jax",
    "sklearn",
    "epsbench.data",
    "epsbench.sim",
    "epsbench.diagnostics.causal_history_fixture",
    "epsbench.diagnostics.occupancy_producer",
    "epsbench.diagnostics.occupancy_collection",
    "epsbench.diagnostics.restricted_learning_producer",
    "epsbench.diagnostics.causal_history_lifecycle",
    "epsbench.diagnostics.causal_history_execution",
    "epsbench.diagnostics.causal_history_native",
    "epsbench.diagnostics.causal_history_runtime",
    "epsbench.diagnostics.paired_appearance_execution",
    "epsbench.diagnostics.a1_execution",
    "epsbench.diagnostics.return_view_execution",
    "epsbench.diagnostics.renderer_discriminator",
    "epsbench.diagnostics.shared_raster_capture",
)


def denied(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        if denied(fullname):
            raise RuntimeError("forbidden restricted source import: " + fullname)
        return None


def loaded() -> None:
    if any(denied(n) for n in sys.modules):
        raise RuntimeError("forbidden module already loaded")


def main() -> int:
    global FORBIDDEN
    if sys.argv[1:]:
        raise ValueError("no selectors or collection allowed")
    loaded()
    sys.meta_path.insert(0, Guard())
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    for name in FORBIDDEN:
        try:
            importlib.import_module(name)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("guard failed: " + name)
    import pytest

    reference = "epsbench.diagnostics.occupancy_reference"
    original = FORBIDDEN
    FORBIDDEN = (*original, reference)
    result = pytest.main(
        ["--noconftest", "-o", "addopts=", "-q", "tests/test_restricted_exact_raster.py"]
    )
    loaded()
    if result:
        return int(result)
    FORBIDDEN = original

    result = pytest.main(
        [
            "--noconftest",
            "-o",
            "addopts=",
            "-q",
            "-s",
            "tests/test_restricted_learning_source.py",
            "tests/test_restricted_learning_collection.py",
        ]
    )
    loaded()
    print("SYNTHETIC_SOURCE_ONLY smoke/validation/inspection; allocation/collection/models HELD")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
