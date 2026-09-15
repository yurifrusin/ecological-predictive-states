"""Owned single-draw canonical ID/depth capture for qualified OSMesa generation.

This module deliberately does not replace :class:`mujoco.Renderer` or any
process-wide SDK entry point.  Its public object owns one explicit native draw
and readback operation on a renderer whose scene has already been updated.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import platform
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import mujoco
import numpy as np
import numpy.typing as npt

NativeIdRgb = npt.NDArray[np.uint8]
NativeDepth = npt.NDArray[np.float32]
RawGeomSegmentation = npt.NDArray[np.int32]

CAPTURE_SCHEMA = "canonical_paired_capture/v1"
ORIENTATION = "native_bottom_up_then_common_vertical_flip_to_image_top_down"
SUPPORTED_RUNTIME = {
    "python": "3.11.15",
    "mujoco": "3.12.0",
    "numpy": "2.4.6",
    "backend": "osmesa",
    "PyOpenGL": "3.1.10",
    "glfw": "2.10.2",
}


class CanonicalPairedCaptureError(RuntimeError):
    """The requested renderer cannot satisfy the canonical paired contract."""


@dataclass(frozen=True)
class SceneMapEntry:
    segid_plus_one: int
    objid: int
    objtype: int


@dataclass(frozen=True)
class CanonicalPairedResult:
    """One canonical pair plus retained privileged native terms."""

    raw_geom_segmentation: RawGeomSegmentation
    depth: NativeDepth
    native_id_rgb: NativeIdRgb
    native_depth_pre_metric: NativeDepth
    scene_map: tuple[SceneMapEntry, ...]
    near: float
    far: float
    stable_state: Mapping[str, object]
    operational_state: Mapping[str, object]
    orientation: str = ORIENTATION


class StateObserver(Protocol):
    def __call__(self, renderer: Any) -> Mapping[str, object]: ...


def decode_id_colors(
    rgb: NativeIdRgb,
    scene_map: Sequence[SceneMapEntry],
) -> RawGeomSegmentation:
    """Decode ID colours and admit only mapped MuJoCo geoms."""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise CanonicalPairedCaptureError("native ID colour buffer shape or dtype differs")
    entries: dict[int, SceneMapEntry] = {}
    for entry in scene_map:
        if (
            entry.segid_plus_one <= 0
            or entry.segid_plus_one > 0xFFFFFF
            or not 0 <= entry.objid <= np.iinfo(np.int32).max
            or entry.objtype != int(mujoco.mjtObj.mjOBJ_GEOM)
            or entry.segid_plus_one in entries
        ):
            raise CanonicalPairedCaptureError("scene ID map is invalid, duplicate, or non-geom")
        entries[entry.segid_plus_one] = entry
    image = rgb.astype(np.uint32)
    segids = image[:, :, 0] + image[:, :, 1] * 256 + image[:, :, 2] * 65536
    unknown = set(int(value) for value in np.unique(segids)) - {0} - set(entries)
    if unknown:
        raise CanonicalPairedCaptureError("native ID colour contains an unmapped nonzero ID")
    decoded = np.full(segids.shape, -1, dtype=np.int32)
    for segid, entry in entries.items():
        decoded[segids == segid] = np.int32(entry.objid)
    return np.ascontiguousarray(decoded)


def convert_native_depth(native_depth: NativeDepth, near: float, far: float) -> NativeDepth:
    """Reproduce MuJoCo 3.12 ``Renderer.render`` depth conversion exactly."""

    if native_depth.ndim != 2 or native_depth.dtype != np.float32:
        raise CanonicalPairedCaptureError("native depth buffer shape or dtype differs")
    if not np.isfinite(native_depth).all() or np.any((native_depth < 0) | (native_depth > 1)):
        raise CanonicalPairedCaptureError("native depth is outside finite SDK readback range")
    if not np.isfinite([near, far]).all() or near <= 0.0 or far <= near:
        raise CanonicalPairedCaptureError("invalid SDK near/far depth parameters")
    zfar, znear = np.float32(far), np.float32(near)
    c_coef = np.float32(-0.5) * (-(zfar + znear) / (zfar - znear)) - np.float32(0.5)
    d_coef = np.float32(-0.5) * (-(np.float32(2.0) * zfar * znear) / (zfar - znear))
    converted = d_coef / (native_depth.astype(np.float64) + c_coef)
    return np.ascontiguousarray(converted.astype(np.float32))


def _scene_map(renderer: Any) -> tuple[SceneMapEntry, ...]:
    scene = renderer._scene
    result = tuple(
        SceneMapEntry(int(geom.segid) + 1, int(geom.objid), int(geom.objtype))
        for geom in scene.geoms[: int(scene.ngeom)]
        if int(geom.segid) != -1
    )
    # Validate the complete map before a draw can occur.
    decode_id_colors(np.zeros((1, 1, 3), dtype=np.uint8), result)
    return result


# Values are GL/MuJoCo enums from the pinned native producer, not configurable tolerances.
_STABLE_KEYS = (
    "rect",
    "scene_flags",
    "ngeom",
    "scene_map",
    "scene_geometry",
    "scene_cameras",
    "projection_matrix_float32",
    "modelview_matrix_float32",
    "readPixelFormat",
    "readDepthMap",
    "read_buffer",
    "draw_buffer",
    "pack_alignment",
    "pack_row_length",
    "pack_skip_rows",
    "pack_skip_pixels",
    "pixel_pack_buffer_binding",
    "clip_origin",
    "clip_depth_mode",
    "framewidth",
    "stereo",
    "rnd_depth",
    "segment_enabled",
    "idcolor_enabled",
    "mjr_currentBuffer",
    "offSamples",
    "offWidth",
    "offHeight",
    "query_bindings_restored",
    "near",
    "far",
    "extent",
    "context_runtime",
)
# A native draw installs its projection/modelview and clip convention. Its input
# camera/scene and attachment identity must already be fixed before that draw.
_DRAW_INSTALLED_KEYS = {
    "projection_matrix_float32",
    "modelview_matrix_float32",
    "clip_origin",
    "clip_depth_mode",
}


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _scene_geometry(renderer: Any) -> list[dict[str, object]]:
    return [
        {
            **{
                name: int(getattr(geom, name))
                for name in ("type", "objid", "objtype", "segid", "category", "dataid")
            },
            **{
                name: np.asarray(getattr(geom, name), dtype=np.float32).reshape(-1).tolist()
                for name in ("pos", "mat", "size")
            },
        }
        for geom in renderer._scene.geoms[: int(renderer._scene.ngeom)]
    ]


def stable_state(state: Mapping[str, object]) -> dict[str, object]:
    """Retain deterministic facts; context/FBO/object names remain in run metadata."""
    if any(key not in state for key in _STABLE_KEYS):
        raise CanonicalPairedCaptureError("paired state observation is incomplete")
    result = {key: copy.deepcopy(state[key]) for key in _STABLE_KEYS}
    attachments = copy.deepcopy(state.get("offscreen_attachments"))
    if not isinstance(attachments, dict):
        raise CanonicalPairedCaptureError("paired attachment observation is incomplete")
    main = attachments.get("offFBO")
    if not isinstance(main, dict):
        raise CanonicalPairedCaptureError("paired offFBO observation is absent")
    main.pop("framebuffer", None)
    for role in ("color0", "depth"):
        attachment = main.get(role)
        if not isinstance(attachment, dict):
            raise CanonicalPairedCaptureError("paired attachment observation is absent")
        attachment.pop("object_name", None)
    result["offscreen_attachments"] = attachments
    return result


def validate_saved_state(state: Mapping[str, object], width: int, height: int) -> None:
    """Validate retained stable facts independently of a live graphics context."""
    if set(state) != {*_STABLE_KEYS, "offscreen_attachments"}:
        raise CanonicalPairedCaptureError("paired stable state fields differ")
    exact = {
        "rect": [0, 0, width, height],
        "readPixelFormat": 6407,
        "readDepthMap": 1,
        "read_buffer": 36064,
        "draw_buffer": 36064,
        "pack_alignment": 1,
        "pack_row_length": 0,
        "pack_skip_rows": 0,
        "pack_skip_pixels": 0,
        "pixel_pack_buffer_binding": 0,
        "framewidth": 0.0,
        "stereo": 0,
        "rnd_depth": False,
        "segment_enabled": True,
        "idcolor_enabled": True,
        "mjr_currentBuffer": 1,
        "offSamples": 0,
        "offWidth": width,
        "offHeight": height,
        "query_bindings_restored": True,
        "clip_origin": 36001,
        "clip_depth_mode": 37727,
    }
    if any(
        type(state.get(key)) is not type(value) or state[key] != value
        for key, value in exact.items()
    ):
        raise CanonicalPairedCaptureError("paired stable producer configuration differs")
    expected_attachments = {
        "offFBO": {
            "present": True,
            "draw_framebuffer_samples": 0,
            "color0": {
                "object_type": 36161,
                "component_type": 35863,
                "internal_format": 32856,
                "samples": 0,
            },
            "depth": {
                "object_type": 36161,
                "component_type": 5126,
                "internal_format": 36013,
                "samples": 0,
            },
        },
        "offFBO_r": {"present": False},
    }
    if state["offscreen_attachments"] != expected_attachments:
        raise CanonicalPairedCaptureError("paired attachment storage/format/sample facts differ")
    for key in ("projection_matrix_float32", "modelview_matrix_float32"):
        value = np.asarray(state[key])
        if value.shape != (16,) or not np.isfinite(value.astype(np.float64)).all():
            raise CanonicalPairedCaptureError("paired projection/modelview facts differ")
    cameras = state["scene_cameras"]
    if not isinstance(cameras, list) or len(cameras) != 2:
        raise CanonicalPairedCaptureError("paired scene cameras are absent")
    for camera in cameras:
        if not isinstance(camera, dict) or set(camera) != {
            "pos",
            "forward",
            "up",
            "frustum_near",
            "frustum_far",
            "frustum_top",
            "frustum_bottom",
            "frustum_center",
            "frustum_width",
            "orthographic",
        }:
            raise CanonicalPairedCaptureError("paired camera/frustum fields differ")
        for key in ("pos", "forward", "up"):
            values = np.asarray(camera[key], dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise CanonicalPairedCaptureError("paired camera vectors differ")
        for key in (
            "frustum_near",
            "frustum_far",
            "frustum_top",
            "frustum_bottom",
            "frustum_center",
            "frustum_width",
        ):
            if not np.isfinite(camera[key]):
                raise CanonicalPairedCaptureError("paired frustum is nonfinite")
        if camera["orthographic"] != 0 or not 0 < camera["frustum_near"] < camera["frustum_far"]:
            raise CanonicalPairedCaptureError("paired perspective convention differs")
    mapping, geometry = state["scene_map"], state["scene_geometry"]
    if not isinstance(mapping, list) or not isinstance(geometry, list) or not geometry:
        raise CanonicalPairedCaptureError("paired scene facts are absent")
    if state["ngeom"] != len(geometry):
        raise CanonicalPairedCaptureError("paired scene geometry count differs")
    for geom in geometry:
        if not isinstance(geom, dict) or set(geom) != {
            "type",
            "objid",
            "objtype",
            "segid",
            "category",
            "dataid",
            "pos",
            "mat",
            "size",
        }:
            raise CanonicalPairedCaptureError("paired scene geometry fields differ")
        if any(
            type(geom[key]) is not int
            for key in ("type", "objid", "objtype", "segid", "category", "dataid")
        ):
            raise CanonicalPairedCaptureError("paired geometry identifiers/types must be integers")
        for key, size in (("pos", 3), ("mat", 9), ("size", 3)):
            values = np.asarray(geom[key], dtype=np.float64)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise CanonicalPairedCaptureError("paired geometry array differs")
    scene_map = tuple(SceneMapEntry(**item) for item in mapping)
    decode_id_colors(np.zeros((1, 1, 3), dtype=np.uint8), scene_map)
    expected_map = [
        {"segid_plus_one": geom["segid"] + 1, "objid": geom["objid"], "objtype": geom["objtype"]}
        for geom in geometry
        if geom["segid"] != -1
    ]
    if mapping != expected_map:
        raise CanonicalPairedCaptureError("paired scene map and geometry differ")
    flags = state["scene_flags"]
    if not isinstance(flags, list) or len(flags) != int(mujoco.mjtRndFlag.mjNRNDFLAG):
        raise CanonicalPairedCaptureError("paired scene flag shape differs")
    for flag, expected in (
        (mujoco.mjtRndFlag.mjRND_SEGMENT, 1),
        (mujoco.mjtRndFlag.mjRND_IDCOLOR, 1),
        (mujoco.mjtRndFlag.mjRND_DEPTH, 0),
    ):
        if type(flags[int(flag)]) is not int or flags[int(flag)] != expected:
            raise CanonicalPairedCaptureError("paired scene flags differ")
    near, far, extent = (float(cast(float, state[key])) for key in ("near", "far", "extent"))
    if not np.isfinite([near, far, extent]).all() or not 0 < near < far or extent <= 0:
        raise CanonicalPairedCaptureError("paired near/far/extent facts differ")
    runtime = state["context_runtime"]
    if not isinstance(runtime, dict):
        raise CanonicalPairedCaptureError("paired context runtime is absent")
    runtime_exact = {
        "actual_backend": "osmesa",
        "context_module": "mujoco.osmesa",
        "requested_offsamples": 0,
        "actual_offsamples": 0,
        "model_offsamples": 0,
        "width": width,
        "height": height,
        "sample_buffers": 0,
        "samples": 0,
        "mjr_off_width": width,
        "mjr_off_height": height,
        "color_storage_dimensions": [width, height],
        "depth_storage_dimensions": [width, height],
        "attachment_format": 32856,
        "attachment_component_type": 35863,
        "gl_samples": 0,
    }
    if set(runtime) != {*runtime_exact, "gl_vendor", "gl_renderer", "gl_version"}:
        raise CanonicalPairedCaptureError("paired context runtime fields differ")
    if any(runtime[key] != value for key, value in runtime_exact.items()):
        raise CanonicalPairedCaptureError("paired context runtime values differ")
    if any(
        not isinstance(runtime[key], str) or not runtime[key]
        for key in ("gl_vendor", "gl_renderer", "gl_version")
    ):
        raise CanonicalPairedCaptureError("paired GL identity is absent")


def validate_state(state: Mapping[str, object], renderer: Any, *, after_draw: bool) -> None:
    """Validate absolute native state, including volatile attachment identities."""
    mjr = renderer._mjr_context
    current = state.get("actual_current_context")
    if not _positive_integer(current) or current != state.get("expected_current_context"):
        raise CanonicalPairedCaptureError("renderer context is not current")
    if not _positive_integer(int(mjr.offFBO)) or int(mjr.offFBO_r) != 0:
        raise CanonicalPairedCaptureError("offscreen or resolve framebuffer differs")
    for key in ("offFBO", "read_framebuffer_binding", "draw_framebuffer_binding"):
        if state.get(key) != int(mjr.offFBO):
            raise CanonicalPairedCaptureError(key + " is not the owned offscreen framebuffer")
    if state.get("offFBO_r") != 0 or int(mjr.offSamples) != 0:
        raise CanonicalPairedCaptureError("sampled or resolve framebuffer is unsupported")
    attachments = state.get("offscreen_attachments")
    if not isinstance(attachments, dict):
        raise CanonicalPairedCaptureError("paired attachments are absent")
    main = attachments.get("offFBO")
    if not isinstance(main, dict) or main.get("framebuffer") != int(mjr.offFBO):
        raise CanonicalPairedCaptureError("paired attachment framebuffer identity differs")
    for role in ("color0", "depth"):
        item = main.get(role)
        if not isinstance(item, dict) or not _positive_integer(item.get("object_name")):
            raise CanonicalPairedCaptureError("paired attachment identity is absent")
    retained = stable_state(state)
    if not after_draw:
        # Only the native draw installs these GL facts. Validate all other inputs now.
        retained.update({"clip_origin": 36001, "clip_depth_mode": 37727})
    validate_saved_state(retained, int(renderer.width), int(renderer.height))


class CanonicalPairedRenderer:
    """Own one joint draw/read; leave its renderer context current, as the SDK does.

    The scene must already be updated. No process-wide SDK entrypoint is replaced.
    Only the scene flags and owned Mjr buffer selection are changed and restored.
    """

    def __init__(
        self,
        renderer: Any,
        *,
        state_observer: StateObserver | None = None,
        native_render: Callable[[Any, Any, Any], object] | None = None,
        native_read_pixels: Callable[[Any, Any, Any, Any], object] | None = None,
        restore_buffer: Callable[[Any, Any], object] | None = None,
    ) -> None:
        if state_observer is None:
            require_supported_runtime(os.environ.get("MUJOCO_GL", ""))
        self._renderer = renderer
        self._observe = state_observer or observe_canonical_paired_state
        self._render = native_render or mujoco.mjr_render
        self._read = native_read_pixels or mujoco.mjr_readPixels
        self._restore_buffer = restore_buffer or mujoco.mjr_setBuffer
        self._thread = threading.get_ident()
        self._active = False

    def capture(self) -> CanonicalPairedResult:
        if self._active or threading.get_ident() != self._thread:
            raise CanonicalPairedCaptureError("paired capture thread/reentrancy differs")
        renderer = self._renderer
        gl_context = getattr(renderer, "_gl_context", None)
        if gl_context is None:
            raise CanonicalPairedCaptureError("renderer GL context is unavailable")
        if bool(renderer._depth_rendering) or bool(renderer._segmentation_rendering):
            raise CanonicalPairedCaptureError("paired capture requires ordinary SDK renderer mode")
        gl_context.make_current()
        scene = renderer._scene
        original_flags = np.asarray(scene.flags).copy()
        original_buffer = int(renderer._mjr_context.currentBuffer)
        mapping = _scene_map(renderer)
        native_id = np.empty((renderer.height, renderer.width, 3), dtype=np.uint8)
        native_depth = np.empty((renderer.height, renderer.width), dtype=np.float32)
        self._active = True
        try:
            scene.flags[int(mujoco.mjtRndFlag.mjRND_SEGMENT)] = True
            scene.flags[int(mujoco.mjtRndFlag.mjRND_IDCOLOR)] = True
            scene.flags[int(mujoco.mjtRndFlag.mjRND_DEPTH)] = False
            before = copy.deepcopy(dict(self._observe(renderer)))
            validate_state(before, renderer, after_draw=False)
            self._render(renderer._rect, scene, renderer._mjr_context)
            after = copy.deepcopy(dict(self._observe(renderer)))
            validate_state(after, renderer, after_draw=True)
            if {k: v for k, v in before.items() if k not in _DRAW_INSTALLED_KEYS} != {
                k: v for k, v in after.items() if k not in _DRAW_INSTALLED_KEYS
            }:
                raise CanonicalPairedCaptureError(
                    "scene/camera/attachment inputs drifted across draw"
                )
            read_before = copy.deepcopy(dict(self._observe(renderer)))
            if read_before != after:
                raise CanonicalPairedCaptureError("producer state drifted before readback")
            self._read(native_id, native_depth, renderer._rect, renderer._mjr_context)
            read_after = copy.deepcopy(dict(self._observe(renderer)))
            validate_state(read_after, renderer, after_draw=True)
            if read_after != read_before or _scene_map(renderer) != mapping:
                raise CanonicalPairedCaptureError("producer state drifted across readback")
            retained_id = np.ascontiguousarray(np.flipud(native_id.copy()))
            retained_depth = np.ascontiguousarray(np.flipud(native_depth.copy()))
            near, far = float(cast(float, after["near"])), float(cast(float, after["far"]))
            return CanonicalPairedResult(
                raw_geom_segmentation=decode_id_colors(retained_id, mapping),
                depth=convert_native_depth(retained_depth, near, far),
                native_id_rgb=retained_id,
                native_depth_pre_metric=retained_depth,
                scene_map=mapping,
                near=near,
                far=far,
                stable_state=stable_state(after),
                operational_state={
                    "draw_input": before,
                    "draw_output": after,
                    "read_input": read_before,
                    "read_output": read_after,
                },
            )
        finally:
            try:
                scene.flags[:] = original_flags
                self._restore_buffer(original_buffer, renderer._mjr_context)
            finally:
                self._active = False


def require_supported_runtime(backend: str) -> None:
    """Reject opt-in production use outside the qualified candidate runtime."""

    python_version = platform.python_version()
    if (
        platform.system() != "Linux"
        or backend != SUPPORTED_RUNTIME["backend"]
        or python_version != SUPPORTED_RUNTIME["python"]
        or mujoco.__version__ != SUPPORTED_RUNTIME["mujoco"]
        or np.__version__ != SUPPORTED_RUNTIME["numpy"]
        or os.environ.get("PYOPENGL_PLATFORM") != "osmesa"
        or not (os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"))
        or any(
            importlib.metadata.version(name) != SUPPORTED_RUNTIME[name]
            for name in ("PyOpenGL", "glfw")
        )
    ):
        raise CanonicalPairedCaptureError("canonical paired capture runtime is unsupported")


def observe_canonical_paired_state(renderer: Any) -> Mapping[str, object]:
    """Observe the qualified native producer without drawing or reading pixels.

    The established observer is reused as a read-only primitive.  The canonical
    path does not use its renderer replacement or capture-study machinery.
    """

    from epsbench.diagnostics.osmesa_joint0_qualification import (
        context_runtime_binding,
        observe_zero_sample_osmesa,
    )
    from epsbench.diagnostics.shared_raster_capture import observe_native_read_state

    # Capture native state first: the context query may not repair an invalid binding.
    state = observe_native_read_state(renderer)
    context_facts = observe_zero_sample_osmesa(
        renderer, renderer._model, int(renderer.width), int(renderer.height)
    )
    state["context_runtime"] = context_runtime_binding(context_facts)
    state["scene_geometry"] = _scene_geometry(renderer)
    for item, camera in zip(
        cast(list[dict[str, object]], state["scene_cameras"]), renderer._scene.camera, strict=True
    ):
        item["orthographic"] = int(camera.orthographic)
    return state
