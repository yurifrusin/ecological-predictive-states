"""Read-only facts extracted from a compiled MuJoCo scene."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class CompiledSceneContract:
    raw_geom_ids: dict[str, int]
    raw_geom_world_positions: dict[str, tuple[float, float, float]]
    raw_geom_compiled_sizes: dict[str, tuple[float, float, float]]
    camera_field_of_view_degrees: float
    camera_world_position: tuple[float, float, float]
    camera_world_rotation_row_major: tuple[float, ...]


def extract_compiled_scene_contract(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    surface_names: tuple[str, ...],
    camera_name: str,
) -> CompiledSceneContract:
    """Bind semantic names to the IDs and metric facts in one compiled model."""

    mujoco.mj_forward(model, data)
    raw_geom_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in surface_names
    }
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    return CompiledSceneContract(
        raw_geom_ids=raw_geom_ids,
        raw_geom_world_positions={
            name: (
                float(data.geom_xpos[raw_id, 0]),
                float(data.geom_xpos[raw_id, 1]),
                float(data.geom_xpos[raw_id, 2]),
            )
            for name, raw_id in raw_geom_ids.items()
        },
        raw_geom_compiled_sizes={
            name: (
                float(model.geom_size[raw_id, 0]),
                float(model.geom_size[raw_id, 1]),
                float(model.geom_size[raw_id, 2]),
            )
            for name, raw_id in raw_geom_ids.items()
        },
        camera_field_of_view_degrees=float(model.cam_fovy[camera_id]),
        camera_world_position=(
            float(data.cam_xpos[camera_id, 0]),
            float(data.cam_xpos[camera_id, 1]),
            float(data.cam_xpos[camera_id, 2]),
        ),
        camera_world_rotation_row_major=tuple(
            float(value) for value in data.cam_xmat[camera_id].reshape(-1)
        ),
    )
