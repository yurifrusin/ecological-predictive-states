"""Guarded public software checks only; no study, model or native entrypoints."""

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
    "epsbench.model",
    "epsbench.models",
    "epsbench.training",
    "epsbench.experiments",
    "epsbench.sim",
    "epsbench.renderer",
    "epsbench.data",
    "epsbench.audit",
    "epsbench.diagnostics.occupancy_collection",
    "epsbench.diagnostics.occupancy_producer",
    "epsbench.diagnostics.finite_mask_motion",
    "epsbench.diagnostics.causal_history_native",
    "epsbench.diagnostics.causal_history_fixture",
    "epsbench.diagnostics.fraction_finite_collector",
    "epsbench.diagnostics.restricted_mask_sampler",
    "epsbench.diagnostics.restricted_learning_producer",
    "epsbench.diagnostics.restricted_mask_training",
    "epsbench.diagnostics.restricted_mask_models",
)


def forbidden(name: str) -> bool:
    return any(name.startswith(p) for p in FORBIDDEN) or (
        name.startswith("epsbench.")
        and any(
            part in name
            for part in ("collector", "collection", "private_operation", "study_launch")
        )
    )


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if forbidden(fullname):
            raise RuntimeError("forbidden disclosure source-check import: " + fullname)
        return None


def loaded() -> None:
    if any(forbidden(name) for name in sys.modules):
        raise RuntimeError("forbidden dependency already loaded")


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("public source checker accepts no study or execution arguments")
    loaded()
    guard = Guard()
    sys.meta_path.insert(0, guard)
    for name in (*FORBIDDEN, "epsbench.diagnostics.private_operation", "epsbench.study_launch"):
        try:
            guard.find_spec(name)
        except RuntimeError:
            pass
        else:
            raise AssertionError("guard failed to deny " + name)
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
            "tests/test_disclosure_geometry.py",
            "tests/test_disclosure_controls.py",
            "tests/test_point_sample_qualification.py",
        ]
    )
    loaded()
    print("Public hand-computed ray/mask/numeric checks only; no study membership or launch")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
