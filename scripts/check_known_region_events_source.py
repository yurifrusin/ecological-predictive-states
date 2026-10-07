"""Guarded handwritten event checks, denying all native/model/collector dependencies."""

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
    "epsbench.data",
    "epsbench.audit",
    "epsbench.diagnostics.restricted_exact_raster",
    "epsbench.diagnostics.occupancy_reference",
    "epsbench.diagnostics.occupancy_collection",
    "epsbench.diagnostics.occupancy_producer",
    "epsbench.diagnostics.restricted_learning",
    "epsbench.diagnostics.restricted_model",
    "epsbench.diagnostics.restricted_health",
    "epsbench.diagnostics.restricted_mask_sampler",
    "epsbench.diagnostics.restricted_mask_training",
    "epsbench.diagnostics.fraction_finite_collector",
    "epsbench.diagnostics.finite_mask_motion",
    "epsbench.diagnostics.causal_history_native",
    "epsbench.diagnostics.restricted_mask_objective_development",
)


def forbidden(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if forbidden(fullname):
            raise RuntimeError("forbidden known-region source-check import: " + fullname)
        return None


def loaded() -> None:
    if any(forbidden(name) for name in sys.modules):
        raise RuntimeError("forbidden dependency already loaded")


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("source-only checker accepts no execution arguments")
    loaded()
    guard = Guard()
    sys.meta_path.insert(0, guard)
    for name in FORBIDDEN:
        try:
            guard.find_spec(name)
        except RuntimeError:
            pass
        else:
            raise AssertionError("forbidden import guard did not deny " + name)
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
            "-s",
            "tests/test_known_region_events.py",
        ]
    )
    loaded()
    print("Only handwritten rasters checked; no model, generation, fitting or gate action")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
