"""Deterministic four-surface corridor scene and privileged renderer evidence."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from epsbench.annotations import (
    AnalyticCamera,
    AnalyticTransportArrays,
    RawBoundaryVisibilityAnalysis,
    compute_analytic_transport,
    compute_raw_boundary_visibility_analysis,
)
from epsbench.appearance import AppearanceRenderPlan, configured_appearance_render_plan
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
CORRIDOR_ATTACHMENT_PAIRS = (
    ("corridor_floor", "corridor_left_surface"),
    ("corridor_floor", "corridor_right_surface"),
    ("corridor_floor", "corridor_end_surface"),
    ("corridor_left_surface", "corridor_end_surface"),
    ("corridor_right_surface", "corridor_end_surface"),
)


def _appearance(config: CorridorConfig, plan: AppearanceRenderPlan | None) -> AppearanceRenderPlan:
    if plan is not None:
        return plan
    return configured_appearance_render_plan(
        config.appearance.profile_id,
        "corridor",
        CORRIDOR_SURFACE_NAMES,
        config.seed,
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
    raw_geom_types: dict[str, str]
    raw_geom_world_rotations_row_major: dict[str, tuple[float, ...]]
    analytic_transport: AnalyticTransportArrays
    boundary_visibility: RawBoundaryVisibilityAnalysis


def corridor_generation_seeds(episode_seed: int) -> tuple[int, int, int]:
    """Return independent geometry, remapping, and appearance namespaces."""

    return (
        derive_seed(episode_seed, "geometry-sampling"),
        derive_seed(episode_seed, "surface-remapping"),
        derive_seed(episode_seed, "appearance-base"),
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


def build_corridor_scene_xml(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance: AppearanceRenderPlan | None = None,
) -> str:
    """Build a parametric open-top corridor whose walls cover the camera optical field."""

    appearance = _appearance(config, appearance)
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
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0" active="0"/>
  </visual>
  <asset>
    {appearance.asset_xml}
  </asset>
  <worldbody>
    <light name="key" directional="true" castshadow="false" pos="0 -1 6"
           dir="{appearance.light_direction}" diffuse="{appearance.light_diffuse}"
           ambient="{appearance.light_ambient}"
           specular="0 0 0"/>
    <geom name="corridor_floor" type="box" pos="0 {half_length} -0.05"
          size="{half_width} {half_length} 0.05"
          rgba="{appearance.rgba_by_surface["corridor_floor"]}"
          {appearance.material_by_surface["corridor_floor"]}/>
    <geom name="corridor_left_surface" type="box"
          pos="{-half_width} {half_length} {half_height}"
          size="{wall_thickness} {half_length} {half_height}"
          rgba="{appearance.rgba_by_surface["corridor_left_surface"]}"
          {appearance.material_by_surface["corridor_left_surface"]}/>
    <geom name="corridor_right_surface" type="box"
          pos="{half_width} {half_length} {half_height}"
          size="{wall_thickness} {half_length} {half_height}"
          rgba="{appearance.rgba_by_surface["corridor_right_surface"]}"
          {appearance.material_by_surface["corridor_right_surface"]}/>
    <geom name="corridor_end_surface" type="box"
          pos="0 {geometry.length} {half_height}"
          size="{half_width} {wall_thickness} {half_height}"
          rgba="{appearance.rgba_by_surface["corridor_end_surface"]}"
          {appearance.material_by_surface["corridor_end_surface"]}/>
    <camera name="monocular_camera"
            pos="{camera_position}"
            xyaxes="1 0 0 0 0 1" fovy="{geometry.field_of_view_degrees}"/>
  </worldbody>
</mujoco>
""".strip()


def compile_corridor_scene_contract(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance: AppearanceRenderPlan | None = None,
) -> CompiledSceneContract:
    """Compile the sampled corridor without rendering and extract its exact facts."""

    appearance = _appearance(config, appearance)
    model = mujoco.MjModel.from_xml_string(
        build_corridor_scene_xml(config, geometry, appearance), appearance.asset_bytes
    )
    data = mujoco.MjData(model)
    return extract_compiled_scene_contract(
        model,
        data,
        CORRIDOR_SURFACE_NAMES,
        "monocular_camera",
    )


def _analytic_camera(
    position: tuple[float, float, float],
    rotation: tuple[float, ...],
    vertical_field_of_view_degrees: float,
) -> AnalyticCamera:
    return AnalyticCamera(
        world_position=position,
        world_rotation_row_major=rotation,
        vertical_field_of_view_degrees=vertical_field_of_view_degrees,
    )


def compute_corridor_analytic_transport(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance: AppearanceRenderPlan | None = None,
) -> AnalyticTransportArrays:
    """Independently compile and recompute transport for whole-dataset validation."""

    appearance = _appearance(config, appearance)
    model = mujoco.MjModel.from_xml_string(
        build_corridor_scene_xml(config, geometry, appearance), appearance.asset_bytes
    )
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    model.cam_pos[camera_id, 1] = geometry.camera_before_forward_position
    mujoco.mj_forward(model, data)
    before_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    model.cam_pos[camera_id, 1] = geometry.camera_after_forward_position
    mujoco.mj_forward(model, data)
    after_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    controlled_geom_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in CORRIDOR_SURFACE_NAMES
    )
    return compute_analytic_transport(
        model,
        data,
        controlled_geom_ids,
        config.render.width,
        config.render.height,
        before_camera,
        after_camera,
    )


def compute_corridor_boundary_visibility(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance: AppearanceRenderPlan | None = None,
) -> RawBoundaryVisibilityAnalysis:
    """Independently compile and recompute the Slice 4 oracle for validation."""

    appearance = _appearance(config, appearance)
    model = mujoco.MjModel.from_xml_string(
        build_corridor_scene_xml(config, geometry, appearance), appearance.asset_bytes
    )
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    model.cam_pos[camera_id, 1] = geometry.camera_before_forward_position
    mujoco.mj_forward(model, data)
    before_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    model.cam_pos[camera_id, 1] = geometry.camera_after_forward_position
    mujoco.mj_forward(model, data)
    after_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    raw_geom_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in CORRIDOR_SURFACE_NAMES
    }
    transport = compute_analytic_transport(
        model,
        data,
        tuple(raw_geom_ids.values()),
        config.render.width,
        config.render.height,
        before_camera,
        after_camera,
    )
    return compute_raw_boundary_visibility_analysis(
        model,
        data,
        raw_geom_ids,
        CORRIDOR_ATTACHMENT_PAIRS,
        config.render.width,
        config.render.height,
        before_camera,
        after_camera,
        transport,
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
    appearance: AppearanceRenderPlan,
) -> CorridorRenderedTransition:
    """Render one prescribed forward transition in the sampled corridor."""

    model = mujoco.MjModel.from_xml_string(
        build_corridor_scene_xml(config, geometry, appearance), appearance.asset_bytes
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
    analytic_transport = compute_analytic_transport(
        model,
        data,
        tuple(compiled.raw_geom_ids[name] for name in CORRIDOR_SURFACE_NAMES),
        config.render.width,
        config.render.height,
        _analytic_camera(
            before.camera_world_position,
            before.camera_world_rotation_row_major,
            compiled.camera_field_of_view_degrees,
        ),
        _analytic_camera(
            after.camera_world_position,
            after.camera_world_rotation_row_major,
            compiled.camera_field_of_view_degrees,
        ),
    )
    before_analytic_camera = _analytic_camera(
        before.camera_world_position,
        before.camera_world_rotation_row_major,
        compiled.camera_field_of_view_degrees,
    )
    after_analytic_camera = _analytic_camera(
        after.camera_world_position,
        after.camera_world_rotation_row_major,
        compiled.camera_field_of_view_degrees,
    )
    boundary_visibility = compute_raw_boundary_visibility_analysis(
        model,
        data,
        compiled.raw_geom_ids,
        CORRIDOR_ATTACHMENT_PAIRS,
        config.render.width,
        config.render.height,
        before_analytic_camera,
        after_analytic_camera,
        analytic_transport,
    )
    return CorridorRenderedTransition(
        before=before,
        after=after,
        raw_geom_ids=compiled.raw_geom_ids,
        raw_geom_positions=compiled.raw_geom_world_positions,
        raw_geom_compiled_sizes=compiled.raw_geom_compiled_sizes,
        raw_geom_types=compiled.raw_geom_types,
        raw_geom_world_rotations_row_major=(compiled.raw_geom_world_rotations_row_major),
        analytic_transport=analytic_transport,
        boundary_visibility=boundary_visibility,
    )
