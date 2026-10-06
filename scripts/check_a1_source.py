"""Run only synthetic A1 checks with native/generation imports denied before collection."""

from __future__ import annotations

import importlib.abc
import os
import runpy
import sys
from pathlib import Path

FORBIDDEN = (
    "mujoco",
    "OpenGL",
    "glfw",
    "epsbench.data.generate",
    "epsbench.sim.renderer",
    "epsbench.sim.canonical_paired",
)


def forbidden(fullname: str) -> bool:
    return any(fullname == name or fullname.startswith(name + ".") for name in FORBIDDEN)


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise RuntimeError("forbidden native/generation import: " + fullname)
        return None


def check_loaded() -> None:
    loaded = sorted(name for name in sys.modules if forbidden(name))
    if loaded:
        raise RuntimeError("forbidden modules already loaded: " + ", ".join(loaded))


def main() -> int:
    check_loaded()
    sys.meta_path.insert(0, Guard())
    # Prevent ambient pytest plugins/options from broadening this bounded invocation.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    import pytest

    if sys.argv[1:] not in ([], ["--boundary-observation-only"]):
        raise ValueError("unsupported source-check selection")
    observation_only = sys.argv[1:] == ["--boundary-observation-only"]
    result = pytest.main(
        [
            "--noconftest",
            "-o",
            "addopts=",
            *(
                []
                if observation_only
                else [
                    "tests/test_a1_action_contrast.py",
                    "tests/test_a1_controls.py",
                    "tests/test_a1_files.py",
                    "tests/test_a1_retention.py",
                    "tests/test_a1_execution.py",
                    "tests/test_causal_history_fixture.py",
                    "tests/test_causal_history_core.py",
                    "tests/test_causal_history_sequence.py",
                    "tests/test_causal_history_execution.py",
                ]
            ),
            "tests/test_boundary_observation.py",
            "-q",
        ]
    )
    check_loaded()
    if result:
        return int(result)
    import json

    if observation_only:
        import numpy as np

        from epsbench.diagnostics.boundary_observation import BoundaryObservationView, VisibleRaster
        from epsbench.schema import Modality, ModalityPermissionSet

        class ExampleProvider:
            def raster(self, sequence_index):
                return VisibleRaster(
                    sequence_index,
                    np.array([[0, 1]], dtype=np.int32),
                    ((1, "surface-0000000000000001"),),
                )

        example = BoundaryObservationView(
            ExampleProvider(),
            ModalityPermissionSet(allowed=frozenset({Modality.SURFACE_REGIONS})),
            0,
        ).observe(0)
        payload = example.canonical_bytes()
        assert json.loads(payload)["edges"][0]["ownership"] == "unknown"
        print(payload.decode())
        check_loaded()
        return 0

    from epsbench.diagnostics.causal_history_fixture import check_candidate

    print(
        json.dumps(
            check_candidate(root / "configs/development/causal_history_fixture_design_v1.json"),
            sort_keys=True,
        )
    )

    from epsbench.diagnostics.a1_action_contrast import CaptureHeldError
    from epsbench.diagnostics.causal_history_execution import maximum_bytes

    print(json.dumps({"causal_maximum_bytes": maximum_bytes()}, sort_keys=True))

    sys.argv = ["a1_action_contrast.py", "capture"]
    try:
        runpy.run_path(str(root / "scripts/a1_action_contrast.py"), run_name="__main__")
    except CaptureHeldError:
        check_loaded()
        print("CLI capture fails closed before producer import; native guard stayed active")
    else:
        raise RuntimeError("capture unexpectedly enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
