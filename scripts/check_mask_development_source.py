"""Only mask-development handwritten CPU checks; native/model imports denied."""

from __future__ import annotations

import importlib.abc
import os
import sys
from pathlib import Path
from typing import NoReturn

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
    "epsbench.diagnostics.fraction_finite_collector",
    "epsbench.diagnostics.causal_history_native",
)


def forbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if forbidden(fullname):
            raise RuntimeError("forbidden mask-development check import: " + fullname)
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
    # The scope permits fake control-flow checks, never actual geometry computation.
    from epsbench.diagnostics import occupancy_reference, restricted_exact_raster

    def denied(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("real producer/reference forbidden in source-only checks")

    restricted_exact_raster.raster = denied
    occupancy_reference.audit = denied
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
            "tests/test_mask_development_qualification.py",
        ]
    )
    loaded()
    print("Mask development public fake checks only; actual study remains unexecuted")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
