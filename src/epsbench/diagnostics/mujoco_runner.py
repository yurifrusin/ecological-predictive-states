"""Execution adapter for the reviewed prospective corridor diagnostic.

Nothing in this module runs at import time. Call only after independent review
and owner-directed execution. It deliberately uses existing corridor builders.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from epsbench.appearance import AppearanceRenderPlan
from epsbench.config import CorridorConfig
from epsbench.diagnostics.capture import DiagnosticFailure
from epsbench.diagnostics.gl_provenance import inspect_mujoco_offscreen_attachments
from epsbench.schema import CorridorSampledGeometry
from epsbench.sim.compiled import extract_compiled_scene_contract
from epsbench.sim.corridor import CORRIDOR_SURFACE_NAMES, build_corridor_scene_xml

RGB = npt.NDArray[np.uint8]
Depth = npt.NDArray[np.float32]
RawIds = npt.NDArray[np.int32]
DecodedPairs = npt.NDArray[np.int32]


@dataclass(frozen=True)
class CapturedFrame:
    rgb: RGB
    depth: Depth
    encoded_segmentation_rgb: RGB
    decoded_object_pairs: DecodedPairs
    raw_geom_ids: RawIds


@dataclass(frozen=True)
class CapturedTransition:
    before: CapturedFrame
    after: CapturedFrame
    provenance: Mapping[str, object]


def _scene_geom_map(renderer: mujoco.Renderer) -> dict[int, tuple[int, int]]:
    scene = renderer.scene
    mapping: dict[int, tuple[int, int]] = {}
    for geom in scene.geoms[: scene.ngeom]:
        if int(geom.segid) >= 0:
            mapping[int(geom.segid)] = (int(geom.objid), int(geom.objtype))
    if not mapping:
        raise DiagnosticFailure("scene has no segid-to-object mapping")
    return mapping


def _decode_rgb_pairs(encoded: RGB, mapping: Mapping[int, tuple[int, int]]) -> DecodedPairs:
    packed = (
        encoded[..., 0].astype(np.int64)
        + 256 * encoded[..., 1].astype(np.int64)
        + 65536 * encoded[..., 2].astype(np.int64)
    )
    segids = packed - 1
    decoded = np.full((*segids.shape, 2), -1, dtype=np.int32)
    for segid, (objid, objtype) in mapping.items():
        decoded[segids == segid] = (objid, objtype)
    # MuJoCo flips the returned decoded array for its GL contexts; caller out is unflipped.
    return np.flipud(decoded)


def _render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    camera_id: int,
    forward_position: float,
) -> CapturedFrame:
    model.cam_pos[camera_id, 1] = forward_position
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_id)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    depth = np.asarray(renderer.render(), dtype=np.float32).copy()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera_id)
    encoded = np.empty((renderer.height, renderer.width, 3), dtype=np.uint8)
    returned_pairs = np.asarray(renderer.render(out=encoded), dtype=np.int32).copy()
    mapping = _scene_geom_map(renderer)
    renderer.disable_segmentation_rendering()
    if returned_pairs.ndim != 3 or returned_pairs.shape[2] != 2:
        raise DiagnosticFailure("MuJoCo returned unexpected decoded segmentation shape")
    if not np.array_equal(_decode_rgb_pairs(encoded, mapping), returned_pairs):
        raise DiagnosticFailure(
            "encoded segmentation RGB disagrees with returned object-id/type segmentation"
        )
    raw_geom_ids = np.where(
        returned_pairs[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM),
        returned_pairs[..., 0],
        -1,
    ).astype(np.int32)
    return CapturedFrame(
        rgb=rgb,
        depth=depth,
        encoded_segmentation_rgb=encoded.copy(),
        decoded_object_pairs=returned_pairs,
        raw_geom_ids=raw_geom_ids,
    )


def capture_corridor_transition(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
    appearance: AppearanceRenderPlan,
    expected_compiled_facts: Mapping[str, object],
    requested_offsamples: int,
    inspect_gl: Callable[
        [mujoco.Renderer], Mapping[str, object]
    ] = inspect_mujoco_offscreen_attachments,
) -> CapturedTransition:
    """Capture RGB/depth/segmentation in historic order after strict scene checks.

    The caller must verify original receipt/manifest/config/seed bindings before
    this call. inspect_gl must query the main offscreen FBO and restore read,
    draw, and renderbuffer bindings in finally; it must fail if material facts
    are unavailable. No fallback substitutes guessed provenance.
    """
    xml = build_corridor_scene_xml(config, geometry, appearance)
    model = mujoco.MjModel.from_xml_string(xml, appearance.asset_bytes)
    data = mujoco.MjData(model)
    compiled = extract_compiled_scene_contract(
        model, data, CORRIDOR_SURFACE_NAMES, "monocular_camera"
    )
    observed = {
        "raw_geom_ids": compiled.raw_geom_ids,
        "raw_geom_world_positions": compiled.raw_geom_world_positions,
        "raw_geom_compiled_sizes": compiled.raw_geom_compiled_sizes,
        "raw_geom_types": compiled.raw_geom_types,
        "camera_field_of_view_degrees": compiled.camera_field_of_view_degrees,
    }
    if observed != dict(expected_compiled_facts):
        raise DiagnosticFailure(
            "compiled corridor facts differ from verified original instrumentation"
        )
    model.vis.quality.offsamples = requested_offsamples
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    renderer = mujoco.Renderer(model, height=config.render.height, width=config.render.width)
    try:
        before = _render_frame(
            model, data, renderer, camera_id, geometry.camera_before_forward_position
        )
        after = _render_frame(
            model, data, renderer, camera_id, geometry.camera_after_forward_position
        )
        provenance = dict(inspect_gl(renderer))
        provenance["requested_offsamples"] = requested_offsamples
        provenance["actual_offsamples"] = int(renderer._mjr_context.offSamples)
        provenance["segid_to_object_map"] = _scene_geom_map(renderer)
        provenance["readback_requires_vertical_flip"] = True
    finally:
        renderer.close()
    return CapturedTransition(before=before, after=after, provenance=provenance)
