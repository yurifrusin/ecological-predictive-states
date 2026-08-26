"""Official-MuJoCo deterministic single-occluder scene instrumentation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from epsbench.config import SingleOccluderConfig
from epsbench.sim.compiled import CompiledSceneContract, extract_compiled_scene_contract

RGBArray = npt.NDArray[np.uint8]
DepthArray = npt.NDArray[np.float32]
RawSegmentationArray = npt.NDArray[np.int32]

SINGLE_OCCLUDER_SURFACE_NAMES = (
    "support_surface",
    "occluding_surface",
    "background_surface",
)


@dataclass(frozen=True)
class RenderedFrame:
    rgb: RGBArray
    depth: DepthArray
    raw_geom_segmentation: RawSegmentationArray
    counterfactual_raw_geom_segmentation: RawSegmentationArray
    camera_world_position: tuple[float, float, float]
    camera_world_rotation_row_major: tuple[float, ...]


@dataclass(frozen=True)
class RenderedTransition:
    before: RenderedFrame
    after: RenderedFrame
    raw_geom_ids: dict[str, int]
    raw_geom_positions: dict[str, tuple[float, float, float]]
    raw_geom_compiled_sizes: dict[str, tuple[float, float, float]]


def _appearance_colours(variant: str) -> dict[str, str]:
    if variant == "alternate":
        return {
            "support": "0.10 0.46 0.35 1",
            "occluder": "0.82 0.70 0.12 1",
            "background": "0.48 0.16 0.68 1",
        }
    return {
        "support": "0.32 0.36 0.42 1",
        "occluder": "0.78 0.18 0.12 1",
        "background": "0.12 0.36 0.76 1",
    }


def build_scene_xml(config: SingleOccluderConfig) -> str:
    """Generate the compact scene rather than loading an external model asset."""

    colours = _appearance_colours(config.appearance.variant)
    camera = config.camera
    support_colour = colours["support"]
    background_colour = colours["background"]
    occluder_colour = colours["occluder"]
    return f"""
<mujoco model="epsbench_single_occluder">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual>
    <global offwidth="{config.render.width}" offheight="{config.render.height}"/>
    <quality shadowsize="0"/>
    <map znear="0.01" zfar="20"/>
  </visual>
  <worldbody>
    <light name="key" directional="true" pos="-1 -2 5" dir="0.2 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="support_surface" type="plane" size="4 7 0.1"
          rgba="{support_colour}"/>
    <geom name="background_surface" type="box" pos="0 2.5 1.05"
          size="2.2 0.05 1.05" rgba="{background_colour}"/>
    <geom name="occluding_surface" type="box" group="1" pos="0 0.8 0.9"
          size="0.55 0.05 0.9" rgba="{occluder_colour}"/>
    <camera name="monocular_camera" pos="{camera.before_lateral} {camera.forward} {camera.height}"
            xyaxes="1 0 0 0 0.16 1" fovy="{camera.field_of_view_degrees}"/>
  </worldbody>
</mujoco>
""".strip()


def compile_single_occluder_scene_contract(
    config: SingleOccluderConfig,
) -> CompiledSceneContract:
    """Compile the configured apparatus without rendering and extract its exact facts."""

    model = mujoco.MjModel.from_xml_string(build_scene_xml(config))
    data = mujoco.MjData(model)
    return extract_compiled_scene_contract(
        model,
        data,
        SINGLE_OCCLUDER_SURFACE_NAMES,
        "monocular_camera",
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
) -> RenderedFrame:
    model.cam_pos[camera_id, 0] = lateral_position
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_id)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

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
    )


def render_transition(config: SingleOccluderConfig) -> RenderedTransition:
    """Render one before/action/after camera transition with no dynamics or GPU requirement."""

    model = mujoco.MjModel.from_xml_string(build_scene_xml(config))
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    compiled = extract_compiled_scene_contract(
        model,
        data,
        SINGLE_OCCLUDER_SURFACE_NAMES,
        "monocular_camera",
    )
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
        )
        after = _render_frame(
            model,
            data,
            renderer,
            camera_id,
            config.camera.after_lateral,
            counterfactual_option,
        )
    finally:
        renderer.close()
    return RenderedTransition(
        before=before,
        after=after,
        raw_geom_ids=compiled.raw_geom_ids,
        raw_geom_positions=compiled.raw_geom_world_positions,
        raw_geom_compiled_sizes=compiled.raw_geom_compiled_sizes,
    )
