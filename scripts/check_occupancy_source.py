"""Guarded noncandidate geometry/provider checks; prospective collection held."""

from __future__ import annotations

import importlib.abc
import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

BASE_FORBIDDEN = (
    "mujoco",
    "OpenGL",
    "glfw",
    "epsbench.data.generate",
    "epsbench.sim.renderer",
    "epsbench.sim.canonical_paired",
)

FORBIDDEN = (
    *BASE_FORBIDDEN,
    "epsbench.diagnostics.causal_history_fixture",
    "epsbench.diagnostics.occupancy_producer",
    "epsbench.diagnostics.causal_history_execution",
    "epsbench.diagnostics.causal_history_native",
    "epsbench.diagnostics.causal_history_runtime",
    "epsbench.diagnostics.paired_appearance_execution",
    "epsbench.diagnostics.a1_execution",
    "epsbench.diagnostics.return_view_execution",
    "epsbench.diagnostics.renderer_discriminator",
)


def denied(name: str) -> bool:
    return any(name == p or name.startswith(p + ".") for p in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        if denied(fullname):
            raise RuntimeError("forbidden occupancy source import: " + fullname)
        return None


def loaded() -> None:
    if any(denied(name) for name in sys.modules):
        raise RuntimeError("forbidden source-check module already loaded")


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("no selectors or actual collection permitted")
    loaded()
    sys.meta_path.insert(0, Guard())
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    # Demonstrate denial before collection, not after an actual producer import.
    import importlib

    for name in FORBIDDEN:
        try:
            importlib.import_module(name)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("guard failed for " + name)
    import pytest

    result = pytest.main(
        [
            "--noconftest",
            "-o",
            "addopts=",
            "-q",
            "-s",
            "tests/test_occupancy_qualification.py",
            "tests/test_visible_forecast_contract.py",
        ]
    )
    loaded()
    print(
        "Noncandidate synthetic geometry/lifecycle/validation/inspection only; "
        "actual qualification HELD"
    )
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
