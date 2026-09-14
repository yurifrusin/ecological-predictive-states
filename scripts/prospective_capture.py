"""Operator-only commands for the finite prospective EPS capture diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

from epsbench.diagnostics.capture import (
    CaptureCell,
    CaptureResult,
    DiagnosticFailure,
    init_capture_ledger,
    run_next_capture_cell,
)


def _paths(arguments: argparse.Namespace) -> dict[str, Path]:
    return {
        "preflight_wgl": Path(arguments.preflight_wgl),
        "generation_wgl": Path(arguments.generation_wgl),
        "preflight_osmesa": Path(arguments.preflight_osmesa),
        "generation_osmesa": Path(arguments.generation_osmesa),
        "wgl_manifest": Path(arguments.wgl_manifest),
        "osmesa_manifest": Path(arguments.osmesa_manifest),
        "wgl_config": Path(arguments.wgl_config),
        "osmesa_config": Path(arguments.osmesa_config),
        "wgl_instrumentation": Path(arguments.wgl_instrumentation),
        "osmesa_instrumentation": Path(arguments.osmesa_instrumentation),
        "wgl_before": Path(arguments.wgl_before),
        "wgl_after": Path(arguments.wgl_after),
        "osmesa_before": Path(arguments.osmesa_before),
        "osmesa_after": Path(arguments.osmesa_after),
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticFailure(f"cannot read verified JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticFailure(f"verified JSON input is not an object: {path}")
    return value


def _tuples(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value


def _configure_backend(backend: str) -> None:
    if backend == "wgl":
        if platform.system() != "Windows":
            raise DiagnosticFailure("WGL cell must run in native Windows")
        if os.environ.get("MUJOCO_GL"):
            raise DiagnosticFailure("WGL cell requires no MUJOCO_GL override")
        return
    if backend == "osmesa":
        if platform.system() == "Windows":
            raise DiagnosticFailure("OSMesa cell must run under WSL/Linux")
        if os.environ.get("MUJOCO_GL") not in (None, "osmesa"):
            raise DiagnosticFailure("OSMesa cell requires MUJOCO_GL=osmesa")
        if os.environ.get("PYOPENGL_PLATFORM") not in (None, "osmesa"):
            raise DiagnosticFailure("OSMesa cell requires PYOPENGL_PLATFORM=osmesa")
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        return
    raise DiagnosticFailure(f"unsupported fixed backend: {backend}")


def _checked_capture(arguments: argparse.Namespace) -> Callable[[CaptureCell], CaptureResult]:
    _configure_backend(arguments.backend)
    from epsbench.appearance import configured_appearance_render_plan
    from epsbench.config import load_config
    from epsbench.diagnostics.mujoco_runner import capture_corridor_transition
    from epsbench.schema import CorridorSampledGeometry
    from epsbench.sim.corridor import CORRIDOR_SURFACE_NAMES

    config = load_config(Path(getattr(arguments, f"{arguments.backend}_config")))
    instrumentation = _read_object(Path(getattr(arguments, f"{arguments.backend}_instrumentation")))
    manifest = _read_object(Path(getattr(arguments, f"{arguments.backend}_manifest")))
    if (
        config.scene_family.value != "corridor"
        or config.render.width != 160
        or config.render.height != 120
        or config.camera.field_of_view_degrees != 55.0
        or config.appearance.profile_id != "legacy_solid_base_v1"
        or int(manifest.get("root_seed", -1)) != 1729
        or int(instrumentation.get("generation_seeds", {}).get("episode_seed", -1))
        != 1703363364368450807
    ):
        raise DiagnosticFailure(
            "config, seed, appearance, or episode binding differs from fixed diagnostic"
        )
    geometry = CorridorSampledGeometry.model_validate(instrumentation["sampled_geometry"])
    appearance = configured_appearance_render_plan(
        config.appearance.profile_id, "corridor", CORRIDOR_SURFACE_NAMES, config.seed
    )
    expected_compiled = _tuples(
        {
            "raw_geom_ids": instrumentation["raw_geom_ids"],
            "raw_geom_world_positions": instrumentation["raw_geom_world_positions"],
            "raw_geom_compiled_sizes": instrumentation["raw_geom_compiled_sizes"],
            "raw_geom_types": instrumentation["raw_geom_types"],
            "raw_geom_world_rotations_row_major": instrumentation[
                "raw_geom_world_rotations_row_major"
            ],
            "camera_field_of_view_degrees": geometry.field_of_view_degrees,
            "camera_world_position": instrumentation["camera_before"]["camera_world_position"],
            "camera_world_rotation_row_major": instrumentation["camera_before"][
                "camera_world_rotation_row_major"
            ],
        }
    )
    before_camera = _tuples(
        {
            "camera_world_position": instrumentation["camera_before"]["camera_world_position"],
            "camera_world_rotation_row_major": instrumentation["camera_before"][
                "camera_world_rotation_row_major"
            ],
        }
    )
    after_camera = _tuples(
        {
            "camera_world_position": instrumentation["camera_after"]["camera_world_position"],
            "camera_world_rotation_row_major": instrumentation["camera_after"][
                "camera_world_rotation_row_major"
            ],
        }
    )

    def capture(cell: CaptureCell) -> CaptureResult:
        if cell.backend != arguments.backend:
            raise DiagnosticFailure(f"next reserved cell is {cell.backend}; use its matching host")
        return capture_corridor_transition(
            config,
            geometry,
            appearance,
            expected_compiled,
            before_camera,
            after_camera,
            cell.requested_offsamples,
            cell.backend,
        )

    return capture


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-wgl", required=True)
    parser.add_argument("--generation-wgl", required=True)
    parser.add_argument("--preflight-osmesa", required=True)
    parser.add_argument("--generation-osmesa", required=True)
    parser.add_argument("--wgl-manifest", required=True)
    parser.add_argument("--osmesa-manifest", required=True)
    parser.add_argument("--wgl-config", required=True)
    parser.add_argument("--osmesa-config", required=True)
    parser.add_argument("--wgl-instrumentation", required=True)
    parser.add_argument("--osmesa-instrumentation", required=True)
    parser.add_argument("--wgl-before", required=True)
    parser.add_argument("--wgl-after", required=True)
    parser.add_argument("--osmesa-before", required=True)
    parser.add_argument("--osmesa-after", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="bind fixed inputs and create an empty ledger")
    _arguments(initialize)
    next_cell = commands.add_parser("next", help="reserve and execute exactly one fixed cell")
    _arguments(next_cell)
    next_cell.add_argument("--backend", required=True, choices=("wgl", "osmesa"))
    arguments = parser.parse_args()
    try:
        output = Path(arguments.output)
        paths = _paths(arguments)
        if arguments.command == "init":
            init_capture_ledger(output, inputs=paths)
            print(f"initialized finite prospective ledger: {output}")
        else:
            cell = run_next_capture_cell(
                output,
                inputs=paths,
                capture=_checked_capture(arguments),
                geom_objtype=5,
            )
            print(f"completed exactly one cell: {cell.name}")
    except DiagnosticFailure as exc:
        parser.exit(1, f"diagnostic stopped: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
