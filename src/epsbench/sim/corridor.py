"""Deterministic four-surface corridor scene and privileged renderer evidence."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from epsbench.config import CorridorConfig
from epsbench.schema import CorridorSampledGeometry
from epsbench.sim.compiled import CompiledSceneContract, extract_compiled_scene_contract
from epsbench.utils.seeding import derive_seed

RGBArray = npt.NDArray[np.uint8]
DepthArray = npt.NDArray[np.float32]
RawSegmentationArray = npt.NDArray[np.int32]

CORRIDOR_SURFACE_NAMES = (
    "corridor_floor",
    "corridor_left_surface",
    "corridor_right_surface",
    "corridor_end_surface",
)


@dataclass(frozen=True)
class CorridorRenderedFrame:
    rgb: RGBArray
    depth: DepthArray
    raw_geom_segmentation: RawSegmentationArray
    camera_world_position: tuple[float, float, float]
    camera_world_rotation_row_major: tuple[float, ...]


@dataclass(frozen=True)
class CorridorRenderedTransition:
    before: CorridorRenderedFrame
    after: CorridorRenderedFrame
    raw_geom_ids: dict[str, int]
    raw_geom_positions: dict[str, tuple[float, float, float]]
    raw_geom_compiled_sizes: dict[str, tuple[float, float, float]]


def corridor_generation_seeds(episode_seed: int) -> tuple[int, int, int]:
    """Return independent geometry, remapping, and appearance namespaces."""

    return (
        derive_seed(episode_seed, "geometry-sampling"),
        derive_seed(episode_seed, "surface-remapping"),
        derive_seed(episode_seed, "appearance"),
    )


def sample_corridor_geometry(
    config: CorridorConfig,
    episode_seed: int,
) -> CorridorSampledGeometry:
    """Sample metric corridor geometry only from the geometry namespace."""

    geometry_seed, _, _ = corridor_generation_seeds(episode_seed)
    rng = np.random.default_rng(geometry_seed)
    width = float(rng.uniform(config.geometry.width.minimum, config.geometry.width.maximum))
    length = float(rng.uniform(config.geometry.length.minimum, config.geometry.length.maximum))
    before_forward = config.camera.starting_forward_position
    return CorridorSampledGeometry(
        width=width,
        length=length,
        wall_height=config.geometry.wall_height,
        camera_lateral_position=config.camera.lateral_position,
        camera_before_forward_position=before_forward,
        camera_after_forward_position=before_forward + config.action.delta_forward,
        camera_height=config.camera.height,
        field_of_view_degrees=config.camera.field_of_view_degrees,
    )


def _appearance_colours(variant: str, appearance_seed: int) -> dict[str, str]:
    if variant == "alternate":
        palette = {
            "floor": (0.18, 0.42, 0.30),
            "left": (0.70, 0.25, 0.55),
            "right": (0.23, 0.55, 0.72),
            "end": (0.82, 0.68, 0.18),
        }
    else:
        palette = {
            "floor": (0.30, 0.32, 0.36),
            "left": (0.65, 0.22, 0.18),
            "right": (0.16, 0.34, 0.68),
            "end": (0.32, 0.58, 0.28),
        }
    rng = np.random.default_rng(appearance_seed)
    colours: dict[str, str] = {}
    for name, rgb in palette.items():
        scale = float(rng.uniform(0.9, 1.05))
        varied = tuple(min(1.0, channel * scale) for channel in rgb)
        colours[name] = f"{varied[0]} {varied[1]} {varied[2]} 1"
    return colours


def build_corridor_scene_xml(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance_seed: int,
) -> str:
    """Build a parametric open-top corridor whose walls cover the camera optical field."""

    colours = _appearance_colours(config.appearance.variant, appearance_seed)
    half_width = geometry.width / 2.0
    half_length = geometry.length / 2.0
    half_height = geometry.wall_height / 2.0
    wall_thickness = 0.05
    camera_position = (
        f"{geometry.camera_lateral_position} "
        f"{geometry.camera_before_forward_position} {geometry.camera_height}"
    )
    return f"""
<mujoco model="epsbench_corridor_v0">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual>
    <global offwidth="{config.render.width}" offheight="{config.render.height}"/>
    <quality shadowsize="0"/>
    <map znear="0.01" zfar="30"/>
  </visual>
  <worldbody>
    <light name="key" directional="true" pos="0 -1 6" dir="0 0.25 -1"
           diffuse="0.95 0.95 0.95"/>
    <geom name="corridor_floor" type="box" pos="0 {half_length} -0.05"
          size="{half_width} {half_length} 0.05" rgba="{colours["floor"]}"/>
    <geom name="corridor_left_surface" type="box"
          pos="{-half_width} {half_length} {half_height}"
          size="{wall_thickness} {half_length} {half_height}" rgba="{colours["left"]}"/>
    <geom name="corridor_right_surface" type="box"
          pos="{half_width} {half_length} {half_height}"
          size="{wall_thickness} {half_length} {half_height}" rgba="{colours["right"]}"/>
    <geom name="corridor_end_surface" type="box"
          pos="0 {geometry.length} {half_height}"
          size="{half_width} {wall_thickness} {half_height}" rgba="{colours["end"]}"/>
    <camera name="monocular_camera"
            pos="{camera_position}"
            xyaxes="1 0 0 0 0 1" fovy="{geometry.field_of_view_degrees}"/>
  </worldbody>
</mujoco>
""".strip()


def compile_corridor_scene_contract(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance_seed: int,
) -> CompiledSceneContract:
    """Compile the sampled corridor without rendering and extract its exact facts."""

    model = mujoco.MjModel.from_xml_string(
        build_corridor_scene_xml(config, geometry, appearance_seed)
    )
    data = mujoco.MjData(model)
    return extract_compiled_scene_contract(
        model,
        data,
        CORRIDOR_SURFACE_NAMES,
        "monocular_camera",
    )


def _render_raw_segmentation(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_id: int,
) -> RawSegmentationArray:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_id)
    segmentation = np.asarray(renderer.render(), dtype=np.int32).copy()
    renderer.disable_segmentation_rendering()
    if segmentation.ndim != 3 or segmentation.shape[2] != 2:
        raise RuntimeError("MuJoCo returned an unexpected segmentation buffer")
    object_ids = segmentation[:, :, 0]
    object_types = segmentation[:, :, 1]
    return np.where(object_types == int(mujoco.mjtObj.mjOBJ_GEOM), object_ids, -1).astype(np.int32)


def _render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_id: int,
    forward_position: float,
) -> CorridorRenderedFrame:
    model.cam_pos[camera_id, 1] = forward_position
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_id)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    depth = np.asarray(renderer.render(), dtype=np.float32).copy()
    renderer.disable_depth_rendering()

    raw_geom_segmentation = _render_raw_segmentation(renderer, data, camera_id)
    return CorridorRenderedFrame(
        rgb=rgb,
        depth=depth,
        raw_geom_segmentation=raw_geom_segmentation,
        camera_world_position=(
            float(data.cam_xpos[camera_id, 0]),
            float(data.cam_xpos[camera_id, 1]),
            float(data.cam_xpos[camera_id, 2]),
        ),
        camera_world_rotation_row_major=tuple(
            float(value) for value in data.cam_xmat[camera_id].reshape(-1)
        ),
    )


def render_corridor_transition(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance_seed: int,
) -> CorridorRenderedTransition:
    """Render one prescribed forward transition in the sampled corridor."""

    model = mujoco.MjModel.from_xml_string(
        build_corridor_scene_xml(config, geometry, appearance_seed)
    )
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    compiled = extract_compiled_scene_contract(
        model,
        data,
        CORRIDOR_SURFACE_NAMES,
        "monocular_camera",
    )
    renderer = mujoco.Renderer(
        model,
        height=config.render.height,
        width=config.render.width,
    )
    try:
        before = _render_frame(
            model,
            data,
            renderer,
            camera_id,
            geometry.camera_before_forward_position,
        )
        after = _render_frame(
            model,
            data,
            renderer,
            camera_id,
            geometry.camera_after_forward_position,
        )
    finally:
        renderer.close()
    return CorridorRenderedTransition(
        before=before,
        after=after,
        raw_geom_ids=compiled.raw_geom_ids,
        raw_geom_positions=compiled.raw_geom_world_positions,
        raw_geom_compiled_sizes=compiled.raw_geom_compiled_sizes,
    )
