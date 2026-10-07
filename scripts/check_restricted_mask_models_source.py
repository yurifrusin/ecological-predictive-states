"""Restricted reader handwritten checks; native/old models and ALL fit/sampling denied."""

from __future__ import annotations

import builtins
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

FORBIDDEN = (
    "mujoco",
    "OpenGL",
    "glfw",
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
    "epsbench.diagnostics.restricted_exact_raster",
    "epsbench.diagnostics.occupancy_reference",
    "epsbench.diagnostics.fraction_finite_collector",
    "epsbench.diagnostics.causal_history_native",
)


def forbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if forbidden(fullname):
            raise RuntimeError("forbidden restricted-mask-projection check import: " + fullname)
        return None


def loaded() -> None:
    if any(forbidden(name) for name in sys.modules):
        raise RuntimeError("native/model modules already loaded")


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("no extra source-check arguments admitted")
    loaded()
    sys.meta_path.insert(0, Guard())
    original_find_spec = importlib.util.find_spec

    def metadata_spec(name: str, package: str | None = None) -> Any:
        if forbidden(name):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = metadata_spec
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if forbidden(name):
            raise RuntimeError("forbidden actual source-check import: " + name)
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from epsbench.diagnostics import restricted_mask_sampler as sampler
    from epsbench.diagnostics import restricted_mask_training as training

    def deny(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fit denied in source checks")

    training.public_readiness = deny  # type: ignore[assignment]
    training.fit_comparative = deny  # type: ignore[assignment]
    training.update = deny  # type: ignore[assignment]
    sampler.build_membership = deny  # type: ignore[assignment]
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
            "tests/test_restricted_mask_models.py",
            "tests/test_restricted_mask_sampler.py",
            "tests/test_restricted_mask_projection.py",
        ]
    )
    loaded()
    print(
        "Restricted readers: handwritten source checks only; "
        "no fit, sampling, experiment or physical qualification"
    )
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
