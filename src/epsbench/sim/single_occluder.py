"""Official-MuJoCo deterministic single-occluder scene instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
from epsbench.config import SingleOccluderConfig
from epsbench.sim.canonical_paired import CanonicalPairedRenderer, CanonicalPairedResult
from epsbench.sim.compiled import CompiledSceneContract, extract_compiled_scene_contract

RGBArray = npt.NDArray[np.uint8]
DepthArray = npt.NDArray[np.float32]
RawSegmentationArray = npt.NDArray[np.int32]

SINGLE_OCCLUDER_SURFACE_NAMES = (
    "support_surface",
    "occluding_surface",
    "background_surface",
)
SINGLE_OCCLUDER_ATTACHMENT_PAIRS = (
    ("support_surface", "occluding_surface"),
    ("support_surface", "background_surface"),
)


def _appearance(
    config: SingleOccluderConfig, plan: AppearanceRenderPlan | None
) -> AppearanceRenderPlan:
    if plan is not None:
        return plan
    return configured_appearance_render_plan(
        config.appearance.profile_id,
        "single_occluder",
        SINGLE_OCCLUDER_SURFACE_NAMES,
        config.seed,
    )


@dataclass(frozen=True)
class RenderedFrame:
    rgb: RGBArray
    depth: DepthArray
    raw_geom_segmentation: RawSegmentationArray
    counterfactual_raw_geom_segmentation: RawSegmentationArray
    camera_world_position: tuple[float, float, float]
    camera_world_rotation_row_major: tuple[float, ...]
    canonical_pair: CanonicalPairedResult | None = None


@dataclass(frozen=True)
class RenderedTransition:
    before: RenderedFrame
    after: RenderedFrame
    raw_geom_ids: dict[str, int]
    raw_geom_positions: dict[str, tuple[float, float, float]]
    raw_geom_compiled_sizes: dict[str, tuple[float, float, float]]
    raw_geom_types: dict[str, str]
    raw_geom_world_rotations_row_major: dict[str, tuple[float, ...]]
    analytic_transport: AnalyticTransportArrays
    boundary_visibility: RawBoundaryVisibilityAnalysis


def build_scene_xml(
    config: SingleOccluderConfig, appearance: AppearanceRenderPlan | None = None
) -> str:
    """Generate the compact scene rather than loading an external model asset."""

    appearance = _appearance(config, appearance)
    camera = config.camera
    support_colour = appearance.rgba_by_surface["support_surface"]
    background_colour = appearance.rgba_by_surface["background_surface"]
    occluder_colour = appearance.rgba_by_surface["occluding_surface"]
    support_material = appearance.material_by_surface["support_surface"]
    background_material = appearance.material_by_surface["background_surface"]
    occluder_material = appearance.material_by_surface["occluding_surface"]
    return f"""
<mujoco model="epsbench_single_occluder">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual>
    <global offwidth="{config.render.width}" offheight="{config.render.height}"/>
    <quality shadowsize="0"/>
    <map znear="0.01" zfar="20"/>
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0" active="0"/>
  </visual>
  <asset>
    {appearance.asset_xml}
  </asset>
  <worldbody>
    <light name="key" directional="true" castshadow="false" pos="-1 -2 5"
           dir="{appearance.light_direction}" diffuse="{appearance.light_diffuse}"
           ambient="{appearance.light_ambient}"
           specular="0 0 0"/>
    <geom name="support_surface" type="plane" size="4 7 0.1"
          rgba="{support_colour}"{support_material}/>
    <geom name="background_surface" type="box" pos="0 2.5 1.05"
          size="2.2 0.05 1.05" rgba="{background_colour}"{background_material}/>
    <geom name="occluding_surface" type="box" group="1" pos="0 0.8 0.9"
          size="0.55 0.05 0.9" rgba="{occluder_colour}"{occluder_material}/>
    <camera name="monocular_camera" pos="{camera.before_lateral} {camera.forward} {camera.height}"
            xyaxes="1 0 0 0 0.16 1" fovy="{camera.field_of_view_degrees}"/>
  </worldbody>
</mujoco>
""".strip()


def compile_single_occluder_scene_contract(
    config: SingleOccluderConfig,
    appearance: AppearanceRenderPlan | None = None,
) -> CompiledSceneContract:
    """Compile the configured apparatus without rendering and extract its exact facts."""

    appearance = _appearance(config, appearance)
    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(config, appearance), appearance.asset_bytes
    )
    data = mujoco.MjData(model)
    return extract_compiled_scene_contract(
        model,
        data,
        SINGLE_OCCLUDER_SURFACE_NAMES,
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


def compute_single_occluder_analytic_transport(
    config: SingleOccluderConfig,
    appearance: AppearanceRenderPlan | None = None,
) -> AnalyticTransportArrays:
    """Independently compile and recompute transport for whole-dataset validation."""

    appearance = _appearance(config, appearance)
    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(config, appearance), appearance.asset_bytes
    )
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    model.cam_pos[camera_id, 0] = config.camera.before_lateral
    mujoco.mj_forward(model, data)
    before_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    model.cam_pos[camera_id, 0] = config.camera.after_lateral
    mujoco.mj_forward(model, data)
    after_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    controlled_geom_ids = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in SINGLE_OCCLUDER_SURFACE_NAMES
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


def compute_single_occluder_boundary_visibility(
    config: SingleOccluderConfig,
    appearance: AppearanceRenderPlan | None = None,
) -> RawBoundaryVisibilityAnalysis:
    """Independently compile and recompute the Slice 4 oracle for validation."""

    appearance = _appearance(config, appearance)
    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(config, appearance), appearance.asset_bytes
    )
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    model.cam_pos[camera_id, 0] = config.camera.before_lateral
    mujoco.mj_forward(model, data)
    before_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    model.cam_pos[camera_id, 0] = config.camera.after_lateral
    mujoco.mj_forward(model, data)
    after_camera = _analytic_camera(
        tuple(float(value) for value in data.cam_xpos[camera_id]),  # type: ignore[arg-type]
        tuple(float(value) for value in data.cam_xmat[camera_id].reshape(-1)),
        float(model.cam_fovy[camera_id]),
    )
    raw_geom_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in SINGLE_OCCLUDER_SURFACE_NAMES
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
        SINGLE_OCCLUDER_ATTACHMENT_PAIRS,
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
    scene_option: mujoco.MjvOption | None = None,
) -> RawSegmentationArray:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
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
    lateral_position: float,
    counterfactual_option: mujoco.MjvOption,
    capture_mode: Literal["legacy", "canonical_paired"],
) -> RenderedFrame:
    model.cam_pos[camera_id, 0] = lateral_position
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_id)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

    canonical_pair: CanonicalPairedResult | None = None
    if capture_mode == "canonical_paired":
        renderer.update_scene(data, camera=camera_id)
        canonical_pair = CanonicalPairedRenderer(renderer).capture()
        depth = canonical_pair.depth
        raw_geom_segmentation = canonical_pair.raw_geom_segmentation
    else:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_id)
        depth = np.asarray(renderer.render(), dtype=np.float32).copy()
        renderer.disable_depth_rendering()
        raw_geom_segmentation = _render_raw_segmentation(renderer, data, camera_id)
    counterfactual_raw_geom_segmentation = _render_raw_segmentation(
        renderer,
        data,
        camera_id,
        scene_option=counterfactual_option,
    )
    return RenderedFrame(
        rgb=rgb,
        depth=depth,
        raw_geom_segmentation=raw_geom_segmentation,
        counterfactual_raw_geom_segmentation=counterfactual_raw_geom_segmentation,
        camera_world_position=(
            float(data.cam_xpos[camera_id, 0]),
            float(data.cam_xpos[camera_id, 1]),
            float(data.cam_xpos[camera_id, 2]),
        ),
        camera_world_rotation_row_major=tuple(
            float(value) for value in data.cam_xmat[camera_id].reshape(-1)
        ),
        canonical_pair=canonical_pair,
    )


def render_transition(
    config: SingleOccluderConfig,
    appearance: AppearanceRenderPlan,
    *,
    capture_mode: Literal["legacy", "canonical_paired"] = "legacy",
) -> RenderedTransition:
    """Render one before/action/after camera transition with no dynamics or GPU requirement."""

    model = mujoco.MjModel.from_xml_string(
        build_scene_xml(config, appearance), appearance.asset_bytes
    )
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    compiled = extract_compiled_scene_contract(
        model,
        data,
        SINGLE_OCCLUDER_SURFACE_NAMES,
        "monocular_camera",
    )
    if capture_mode == "canonical_paired":
        model.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(
        model,
        height=config.render.height,
        width=config.render.width,
    )
    counterfactual_option = mujoco.MjvOption()
    counterfactual_option.geomgroup[:] = 1
    counterfactual_option.geomgroup[1] = 0
    try:
        before = _render_frame(
            model,
            data,
            renderer,
            camera_id,
            config.camera.before_lateral,
            counterfactual_option,
            capture_mode,
        )
        after = _render_frame(
            model,
            data,
            renderer,
            camera_id,
            config.camera.after_lateral,
            counterfactual_option,
            capture_mode,
        )
    finally:
        renderer.close()
    analytic_transport = compute_analytic_transport(
        model,
        data,
        tuple(compiled.raw_geom_ids[name] for name in SINGLE_OCCLUDER_SURFACE_NAMES),
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
        SINGLE_OCCLUDER_ATTACHMENT_PAIRS,
        config.render.width,
        config.render.height,
        before_analytic_camera,
        after_analytic_camera,
        analytic_transport,
    )
    return RenderedTransition(
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
