"""Guarded fake mapping checks; geometry and historical collectors cannot execute."""

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
    "epsbench.diagnostics.event_cell_mapping",
    "epsbench.sim",
    "epsbench.data",
    "epsbench.audit",
    "epsbench.diagnostics.occupancy_collection",
    "epsbench.diagnostics.occupancy_producer",
    "epsbench.diagnostics.restricted_learning",
    "epsbench.diagnostics.restricted_model",
    "epsbench.diagnostics.restricted_health",
    "epsbench.diagnostics.restricted_mask_sampler",
    "epsbench.diagnostics.restricted_mask_training",
    "epsbench.diagnostics.finite_mask_motion",
    "epsbench.diagnostics.fraction_finite_collector",
    "epsbench.diagnostics.causal_history_native",
    "epsbench.diagnostics.restricted_mask_objective_development",
)


def forbidden(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if forbidden(fullname):
            raise RuntimeError("forbidden area-census check import: " + fullname)
        return None


def loaded() -> None:
    if any(forbidden(name) for name in sys.modules):
        raise RuntimeError("forbidden dependency already loaded")


def source_closure() -> None:
    """All guarded imported local runtime modules must belong to the new source pin."""
    from epsbench.diagnostics.known_region_area_census import SOURCES

    root = Path(__file__).resolve().parents[1]
    imported = set()
    for name, module in sys.modules.items():
        filename = getattr(module, "__file__", None)
        if name.startswith("epsbench") and filename:
            source = Path(filename).resolve()
            if source.is_relative_to(root / "src"):
                imported.add(source.relative_to(root).as_posix())
    missing = imported - set(SOURCES)
    if missing:
        raise RuntimeError(
            "guarded imported source bytes not pinned: " + ", ".join(sorted(missing))
        )


def denied(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("real geometry/collector execution forbidden in fake source checks")


def install() -> None:
    loaded()
    sys.meta_path.insert(0, Guard())
    from epsbench.diagnostics import (
        mask_development_qualification as old,
    )
    from epsbench.diagnostics import (
        occupancy_reference as reference,
    )
    from epsbench.diagnostics import (
        restricted_exact_raster as producer,
    )

    for name in ("raster", "box_hit", "support_hit", "nearest"):
        setattr(producer, name, denied)
    for name in ("ray",):
        setattr(producer.Calibration, name, denied)
    for name in ("audit", "face_hit", "support_hit", "footprint", "status", "continuous_cover"):
        setattr(reference, name, denied)
    for name in ("collect", "inspect", "before_records", "parse"):
        setattr(old, name, denied)
    for name in ("produce", "audit"):
        setattr(old.ExactAdapter, name, denied)


def main() -> int:
    if sys.argv[1:]:
        raise ValueError("source checker accepts no operation arguments")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    install()
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
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
            "tests/test_known_region_area_census.py",
        ]
    )
    loaded()
    source_closure()
    print("Handwritten census/ZIP fixtures only; physical census remains unexecuted")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
