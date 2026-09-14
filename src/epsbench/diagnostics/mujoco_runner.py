"""Concrete two-frame runner for the reviewed prospective corridor diagnostic.

This module never runs at import time. The CLI configures the backend before
importing it; importing MuJoCo here would make that ordering unsafe.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import platform
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.diagnostics.capture import (
    CapturedFrame,
    CaptureResult,
    DiagnosticFailure,
    PartialCapturedFrame,
    PartialCaptureFailure,
)
from epsbench.diagnostics.gl_provenance import inspect_mujoco_offscreen_attachments


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_hashes(mujoco: Any, xml: str) -> dict[str, object]:
    renderer_path = Path(mujoco.__file__).parent / "rendering" / "classic" / "renderer.py"
    binary = Path(mujoco.__file__).parent / (
        "mujoco.dll" if platform.system() == "Windows" else "libmujoco.so.3.12.0"
    )
    if not renderer_path.is_file() or not binary.is_file():
        raise DiagnosticFailure("MuJoCo runtime hash targets are unavailable")
    package_root = Path(mujoco.__file__).parent
    package_digest = hashlib.sha256()
    for path in sorted(package_root.glob("*.py")):
        package_digest.update(path.name.encode("utf-8"))
        package_digest.update(path.read_bytes())
    return {
        "xml_source_sha256": hashlib.sha256(xml.encode("utf-8")).hexdigest(),
        "renderer_py_sha256": _sha256_file(renderer_path),
        "binary_sha256": _sha256_file(binary),
        "package_sha256": package_digest.hexdigest(),
        "package_version": importlib.metadata.version("mujoco"),
        "host": platform.node(),
    }


def _model_sha256(mujoco: Any, model: Any) -> str:
    """Hash the compiled model bytes, not merely the source XML."""
    with tempfile.NamedTemporaryFile(suffix=".mjb", delete=False) as stream:
        path = Path(stream.name)
    try:
        mujoco.mj_saveModel(model, str(path), None)
        return _sha256_file(path)
    except Exception as exc:
        raise DiagnosticFailure("cannot persist compiled MuJoCo model for provenance") from exc
    finally:
        path.unlink(missing_ok=True)


def _scene_geom_map(renderer: Any) -> dict[int, tuple[int, int]]:
    mapping: dict[int, tuple[int, int]] = {}
    for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
        if int(geom.segid) >= 0:
            mapping[int(geom.segid)] = (int(geom.objid), int(geom.objtype))
    if not mapping:
        raise DiagnosticFailure("scene has no segid-to-object map")
    return mapping


class _FrameFailure(RuntimeError):
    """Failure after a known subset of a frame has been obtained."""

    def __init__(self, message: str, observations: PartialCapturedFrame, stage: str) -> None:
        super().__init__(message)
        self.observations = observations
        self.stage = stage


def _pointer_value(value: Any) -> int:
    try:
        return int(ctypes.cast(value, ctypes.c_void_p).value or 0)
    except (TypeError, ValueError):
        return int(value)


def _observed_backend(renderer: Any) -> str:
    """Derive and validate the live context backend, independently of CLI text."""
    context = getattr(renderer, "_gl_context", None)
    module = type(context).__module__.lower()
    if module == "mujoco.osmesa":
        return "osmesa"
    if platform.system() != "Windows" or module != "mujoco.glfw":
        raise DiagnosticFailure(f"unsupported or unobservable live renderer context: {module}")
    try:
        import glfw
    except ImportError as exc:
        raise DiagnosticFailure("GLFW is required to verify the live WGL context") from exc
    window = getattr(context, "_context", None)
    current = glfw.get_current_context()
    if window is None or current is None:
        raise DiagnosticFailure("MuJoCo GLFW context is not current")
    if glfw.get_window_attrib(window, glfw.CONTEXT_CREATION_API) != glfw.NATIVE_CONTEXT_API:
        raise DiagnosticFailure("MuJoCo GLFW context is not a native WGL context")
    if _pointer_value(glfw.get_wgl_context(window)) == 0 or _pointer_value(
        glfw.get_wgl_context(window)
    ) != _pointer_value(glfw.get_wgl_context(current)):
        raise DiagnosticFailure("MuJoCo GLFW WGL handle does not match the current context")
    return "wgl"


def _frame(
    mujoco: Any,
    model: Any,
    data: Any,
    renderer: Any,
    camera_id: int,
    forward: float,
) -> CapturedFrame:
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    encoded: np.ndarray | None = None
    pairs: np.ndarray | None = None
    mapping: dict[int, tuple[int, int]] | None = None
    stage = "camera_position"
    try:
        model.cam_pos[camera_id, 1] = forward
        stage = "mj_forward"
        mujoco.mj_forward(model, data)
        stage = "rgb_scene_update"
        renderer.update_scene(data, camera=camera_id)
        stage = "rgb_readback"
        rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
        stage = "depth_enable"
        renderer.enable_depth_rendering()
        stage = "depth_scene_update"
        renderer.update_scene(data, camera=camera_id)
        stage = "depth_readback"
        depth = np.asarray(renderer.render(), dtype=np.float32).copy()
        stage = "depth_disable"
        renderer.disable_depth_rendering()
        stage = "segmentation_enable"
        renderer.enable_segmentation_rendering()
        stage = "segmentation_scene_update"
        renderer.update_scene(data, camera=camera_id)
        stage = "segmentation_readback_before_decoded_return"
        encoded_buffer = np.zeros((renderer.height, renderer.width, 3), dtype=np.uint8)
        encoded = encoded_buffer
        pairs = np.asarray(renderer.render(out=encoded_buffer), dtype=np.int32).copy()
        encoded = encoded_buffer.copy()
        stage = "segmentation_scene_map"
        mapping = _scene_geom_map(renderer)
        stage = "segmentation_disable"
        renderer.disable_segmentation_rendering()
        stage = "raw_geom_ids"
        raw = np.where(pairs[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), pairs[..., 0], -1).astype(
            np.int32
        )
    except Exception as exc:
        raise _FrameFailure(
            str(exc),
            PartialCapturedFrame(
                rgb=rgb,
                depth=depth,
                encoded_rgb=encoded.copy() if encoded is not None else None,
                decoded_pairs=pairs,
                segid_to_object_map=mapping,
            ),
            stage,
        ) from exc
    assert rgb is not None
    assert depth is not None
    assert encoded is not None
    assert pairs is not None
    assert mapping is not None
    return CapturedFrame(
        rgb=rgb,
        depth=depth,
        encoded_rgb=encoded,
        decoded_pairs=pairs,
        raw_geom_ids=raw,
        segid_to_object_map=mapping,
    )


def _contract_mapping(contract: Any) -> dict[str, object]:
    return {
        "raw_geom_ids": contract.raw_geom_ids,
        "raw_geom_world_positions": contract.raw_geom_world_positions,
        "raw_geom_compiled_sizes": contract.raw_geom_compiled_sizes,
        "raw_geom_types": contract.raw_geom_types,
        "raw_geom_world_rotations_row_major": contract.raw_geom_world_rotations_row_major,
        "camera_field_of_view_degrees": contract.camera_field_of_view_degrees,
        "camera_world_position": contract.camera_world_position,
        "camera_world_rotation_row_major": contract.camera_world_rotation_row_major,
    }


def _camera_fact(data: Any, camera_id: int) -> dict[str, object]:
    return {
        "camera_world_position": tuple(float(value) for value in data.cam_xpos[camera_id]),
        "camera_world_rotation_row_major": tuple(
            float(value) for value in data.cam_xmat[camera_id].reshape(-1)
        ),
    }


def capture_corridor_transition(
    config: Any,
    geometry: Any,
    appearance: Any,
    expected_compiled_facts: Mapping[str, object],
    expected_before_camera: Mapping[str, object],
    expected_after_camera: Mapping[str, object],
    requested_offsamples: int,
    backend: str,
    inspect_gl: Callable[[Any], Mapping[str, object]] = inspect_mujoco_offscreen_attachments,
) -> CaptureResult:
    """Compile, validate all geometry/cameras, then render the prescribed pair once."""
    import mujoco

    from epsbench.sim.compiled import extract_compiled_scene_contract
    from epsbench.sim.corridor import CORRIDOR_SURFACE_NAMES, build_corridor_scene_xml

    xml = build_corridor_scene_xml(config, geometry, appearance)
    model = mujoco.MjModel.from_xml_string(xml, appearance.asset_bytes)
    data = mujoco.MjData(model)
    compiled = extract_compiled_scene_contract(
        model, data, CORRIDOR_SURFACE_NAMES, "monocular_camera"
    )
    observed = _contract_mapping(compiled)
    if observed != dict(expected_compiled_facts):
        raise DiagnosticFailure(
            "compiled geometry, rotations, or initial camera differ from verified input"
        )
    # All source, compiled-geometry, and both world-camera checks happen CPU-only.
    model.vis.quality.offsamples = requested_offsamples
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "monocular_camera")
    model.cam_pos[camera_id, 1] = geometry.camera_before_forward_position
    mujoco.mj_forward(model, data)
    if _camera_fact(data, camera_id) != dict(expected_before_camera):
        raise DiagnosticFailure("before camera pose differs from verified input")
    model.cam_pos[camera_id, 1] = geometry.camera_after_forward_position
    mujoco.mj_forward(model, data)
    if _camera_fact(data, camera_id) != dict(expected_after_camera):
        raise DiagnosticFailure("after camera pose differs from verified input")
    model.cam_pos[camera_id, 1] = geometry.camera_before_forward_position
    mujoco.mj_forward(model, data)
    model_hash = _model_sha256(mujoco, model)
    renderer = mujoco.Renderer(model, height=config.render.height, width=config.render.width)
    before: CapturedFrame | None = None
    after: CapturedFrame | None = None
    try:
        before = _frame(
            mujoco, model, data, renderer, camera_id, geometry.camera_before_forward_position
        )
        after = _frame(
            mujoco, model, data, renderer, camera_id, geometry.camera_after_forward_position
        )
        provenance = dict(inspect_gl(renderer))
        provenance.update(_runtime_hashes(mujoco, xml))
        provenance["model_sha256"] = model_hash
        provenance["requested_offsamples"] = requested_offsamples
        provenance["actual_offsamples"] = int(renderer._mjr_context.offSamples)
        provenance["backend"] = backend
        provenance["actual_backend"] = _observed_backend(renderer)
        return CaptureResult(before=before, after=after, provenance=provenance)
    except _FrameFailure as exc:
        raise PartialCaptureFailure(
            str(exc),
            before=before,
            partial_frame=exc.observations,
            partial_frame_name="before" if before is None else "after",
            stage=exc.stage,
        ) from exc
    except Exception as exc:
        if before is not None and after is not None:
            raise PartialCaptureFailure(
                str(exc), partial=CaptureResult(before, after, {}), stage="post_frame_capture"
            ) from exc
        if before is not None:
            raise PartialCaptureFailure(
                str(exc), before=before, stage="after_before_frame"
            ) from exc
        raise
    finally:
        renderer.close()
