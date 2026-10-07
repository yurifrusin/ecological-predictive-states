"""Guarded synthetic health + model checks; every actual launch held."""

from __future__ import annotations

import importlib
import importlib.abc
import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

FORBIDDEN: tuple[str, ...] = (
    "mujoco",
    "OpenGL",
    "glfw",
    "tensorflow",
    "jax",
    "sklearn",
    "epsbench.diagnostics.occupancy_reference",
    "epsbench.diagnostics.restricted_learning_collection",
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


class DeniedLoader(importlib.abc.Loader):
    def create_module(self, spec: ModuleSpec) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        raise RuntimeError("forbidden restricted source import: " + module.__name__)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> ModuleSpec | None:
        if denied(fullname):
            return ModuleSpec(fullname, DeniedLoader())
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

    from epsbench.diagnostics import restricted_model_training
    from epsbench.diagnostics.restricted_model_export import ReadOnlyExport
    from epsbench.diagnostics.restricted_model_forecasts import ForecastSession

    def held(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("actual export/fit/forecast launch HELD in source checks")

    from epsbench.diagnostics import restricted_model_health

    restricted_model_health.run = held
    restricted_model_health.learner = held
    restricted_model_health.child_process = held
    restricted_model_training.train = held
    ReadOnlyExport.__init__ = held  # type: ignore[method-assign]
    ForecastSession.from_export = held  # type: ignore[method-assign]

    result = pytest.main(
        [
            "--noconftest",
            "-o",
            "addopts=",
            "-q",
            "-s",
            "tests/test_restricted_models_source.py",
            "tests/test_restricted_health_source.py",
            "tests/test_restricted_health_runtime.py",
        ]
    )
    loaded()
    print("SYNTHETIC_SOURCE_ONLY health/model checks; actual export/health/training/forecast HELD")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
