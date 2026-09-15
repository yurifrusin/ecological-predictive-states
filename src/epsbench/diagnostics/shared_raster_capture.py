"""Finite shared-raster ID/depth candidate capture study.

The candidate changes only the color argument of an ordinary MuJoCo depth
``mjr_readPixels`` call.  The ID bytes and depth values therefore come from the
same completed depth-tested draw, but the two GL reads performed by the SDK call
remain sequential rather than hardware-atomic.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import sys
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from epsbench.diagnostics.osmesa_joint0_qualification import (
    LOCK_SHA256,
    PROFILES,
    SDK_RENDERER_SHA256,
    SDK_VERSION,
    Attempt,
    _raw_segmentation,
    context_runtime_binding,
    observe_zero_sample_osmesa,
    require_osmesa_linux,
    validate_child_source_provenance,
    validate_source_linkage,
)
from epsbench.diagnostics.osmesa_joint0_qualification import (
    bind_pristine_sdk as qualification_bind_pristine_sdk,
)
from epsbench.diagnostics.osmesa_joint0_qualification import (
    git_binding as qualification_git_binding,
)
from epsbench.diagnostics.osmesa_joint0_qualification import (
    permission_probes as canonical_permission_probes,
)
from epsbench.diagnostics.revision_capture import (
    AttemptLock,
    canonical_json_bytes,
    publish_bytes,
    sha256_file,
)

SCHEMA = "shared_raster_capture_candidate/v1"
LEDGER_SCHEMA = "shared_raster_capture_ledger/v1"
PAIR_SCHEMA = "shared_raster_pair/v1"
ROOT_SEED = 1729
STATUS = (
    "FINITE_SHARED_RASTER_CANDIDATE_EVIDENCE_ESTABLISHED_"
    "METRIC_AND_GEOMETRIC_ACCURACY_SEPARATE_UNQUALIFIED"
)
_HOOK_LOCK = threading.Lock()
_PRISTINE: dict[int, tuple[object, object, object]] = {}


class SharedRasterFailure(RuntimeError):
    """An integrity failure that permanently stops this study."""


def digest_file(path: Path) -> str:
    return sha256_file(path)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return canonical_json_bytes(value)


def fixed_attempts() -> tuple[Attempt, ...]:
    attempts: list[Attempt] = []
    for family in ("single_occluder", "corridor"):
        for profile in PROFILES:
            for repeat in (0, 1):
                attempts.append(Attempt(len(attempts), family, profile, repeat))
    return tuple(attempts)


def plan() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "review_profile": "DUAL_REVIEW",
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "phase_gate_effect": "NONE",
        "seed": ROOT_SEED,
        "episodes_per_attempt": 4,
        "resolution": [160, 120],
        "component_topology": False,
        "backend": "osmesa",
        "quality": {"offsamples": 0},
        "profiles": list(PROFILES),
        "attempts": [
            {
                **attempt.__dict__,
                "name": attempt.name,
                "expected_contexts": 4,
                "expected_native_render_calls": attempt.expected_native_calls,
                "expected_native_readbacks": attempt.expected_native_calls,
                "expected_paired_depth_readbacks": 8,
            }
            for attempt in fixed_attempts()
        ],
        "runtime": {
            "python": "3.11.15",
            "mujoco": "3.12.0",
            "numpy": "2.4.6",
            "PyOpenGL": "3.1.10",
            "glfw": "2.10.2",
        },
        "limits": {
            "batch_attempts": 8,
            "contexts": 32,
            "ordinary_pose_endpoints": 64,
            "ordinary_modality_renders": 192,
            "counterfactual_segmentation_renders": 32,
            "native_render_calls": 224,
            "native_readbacks": 224,
            "paired_readbacks": 64,
        },
        "depth_term": "native SDK readback before metric conversion",
        "readback_atomicity": "sequential color/depth GL reads; not hardware-atomic",
        "repeat_exclusion": "dataset-relative run.json only",
        "completion_interpretation": STATUS,
    }


def validate_plan(value: object) -> None:
    if value != plan():
        raise SharedRasterFailure("immutable shared-raster plan differs")


def git_binding(root: Path) -> dict[str, object]:
    binding = qualification_git_binding(
        root,
        root / "uv.lock",
        (root / "configs/benchmark_v0.yaml", root / "configs/corridor_v0.yaml"),
        root / "configs/appearance_candidates_v0.yaml",
        root / "configs/evaluation_seed_candidates_v0.yaml",
    )
    if binding["lock_sha256"] != LOCK_SHA256:
        raise SharedRasterFailure("dependency lock differs from fixed study lock")
    return binding


def validate_bindings(root: Path, recorded: Mapping[str, object]) -> None:
    if git_binding(root) != dict(recorded):
        raise SharedRasterFailure("source/config/registry binding changed")


@dataclass(frozen=True)
class PairedEvidencePermission:
    allow_depth: bool
    allow_privileged_raw_ids: bool
    allow_instrumentation_metadata: bool = False

    @classmethod
    def authorized(cls) -> PairedEvidencePermission:
        return cls(
            allow_depth=True, allow_privileged_raw_ids=True, allow_instrumentation_metadata=True
        )

    def require(self) -> None:
        if not self.allow_depth or not self.allow_privileged_raw_ids:
            raise PermissionError("paired evidence requires depth and privileged raw-ID authority")


class PairedArtifactAccess:
    """Fail-closed access; permission is checked before path or metadata I/O."""

    def __init__(self, root: Path, permission: PairedEvidencePermission):
        permission.require()
        self._root = root
        self._permission = permission

    def metadata(self, episode: int, frame: int) -> dict[str, object]:
        if not self._permission.allow_instrumentation_metadata:
            raise PermissionError("paired metadata requires instrumentation authority")
        path = self._root / f"episode-{episode:06d}" / f"frame-{frame}" / "pair.json"
        return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))

    def array(self, episode: int, frame: int, name: str) -> np.ndarray:
        allowed = {
            "native_id_rgb",
            "native_depth_pre_metric",
            "decoded_objid",
            "decoded_objtype",
            "converted_depth",
            "canonical_sdk_segmentation",
            "canonical_sdk_depth",
        }
        if name not in allowed:
            raise SharedRasterFailure("unknown paired artifact array")
        path = self._root / f"episode-{episode:06d}" / f"frame-{frame}" / f"{name}.npy"
        return cast(np.ndarray, np.load(path, allow_pickle=False))


def _array_ok(value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype[Any], role: str) -> None:
    if value.shape != shape or value.dtype != dtype or not value.flags.c_contiguous:
        raise SharedRasterFailure(f"{role} shape/dtype/contiguity differs")


def decode_id_colors(
    rgb: np.ndarray, scene_map: Sequence[Mapping[str, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode SDK little-channel-order 24-bit ID colors with strict map checks."""
    if rgb.ndim != 3:
        raise SharedRasterFailure("ID RGB rank differs")
    _array_ok(rgb, (rgb.shape[0], rgb.shape[1], 3), np.dtype(np.uint8), "ID RGB")
    entries: dict[int, tuple[int, int]] = {}
    for item in scene_map:
        segid, entry_objid, entry_objtype = (
            int(item["segid_plus_one"]),
            int(item["objid"]),
            int(item["objtype"]),
        )
        if (
            segid <= 0
            or segid > 0xFFFFFF
            or entry_objid < 0
            or entry_objtype < 0
            or segid in entries
        ):
            raise SharedRasterFailure("invalid, duplicate, or out-of-range scene ID map")
        entries[segid] = (entry_objid, entry_objtype)
    image = rgb.astype(np.uint32)
    segids = image[:, :, 0] + image[:, :, 1] * 256 + image[:, :, 2] * 65536
    unknown = set(int(x) for x in np.unique(segids)) - {0} - set(entries)
    if unknown:
        raise SharedRasterFailure("unmapped nonzero ID color")
    objid = np.full(segids.shape, -1, dtype=np.int32)
    objtype = np.full(segids.shape, -1, dtype=np.int32)
    for segid, pair in entries.items():
        mask = segids == segid
        objid[mask], objtype[mask] = pair
    return np.ascontiguousarray(segids), objid, objtype


def depth_coefficients(near: float, far: float) -> tuple[np.float32, np.float32]:
    if not np.isfinite([near, far]).all() or near <= 0 or far <= near:
        raise SharedRasterFailure("invalid near/far depth parameters")
    zfar, znear = np.float32(far), np.float32(near)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)
    return np.float32(-0.5) * c_coef - np.float32(0.5), np.float32(-0.5) * d_coef


def convert_native_depth(native_depth: np.ndarray, near: float, far: float) -> np.ndarray:
    if native_depth.ndim != 2:
        raise SharedRasterFailure("native depth rank differs")
    _array_ok(native_depth, native_depth.shape, np.dtype(np.float32), "native depth")
    c_coef, d_coef = depth_coefficients(near, far)
    converted64 = d_coef / (native_depth.astype(np.float64) + c_coef)
    return np.ascontiguousarray(converted64.astype(np.float32))


def _strict_gl_integer(value: object, name: str, *, allow_zero_padding: bool = False) -> int:
    array = np.asarray(value)
    if array.size == 1:
        return int(array.reshape(-1)[0])
    if allow_zero_padding and array.size == 2 and int(array.reshape(-1)[1]) == 0:
        return int(array.reshape(-1)[0])
    raise SharedRasterFailure(f"{name} has unexpected query shape or padding")


def observe_shared_context(
    renderer: object, model: Any, width: int, height: int
) -> Mapping[str, object]:
    from OpenGL import GL  # type: ignore[import-untyped]

    facts = dict(observe_zero_sample_osmesa(renderer, model, width, height))
    context = getattr(renderer, "_gl_context", None)
    if context is None:
        raise SharedRasterFailure("renderer context unavailable")
    context.make_current()
    facts["osmesa_context_identity"] = _pointer_identity(getattr(context, "_context", None))
    mjr = getattr(renderer, "_mjr_context", None)
    facts["readPixelFormat"] = int(getattr(mjr, "readPixelFormat", -1))
    facts["readDepthMap"] = int(getattr(mjr, "readDepthMap", -1))
    facts["gl_rgb"] = int(GL.GL_RGB)
    facts["clip_origin"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_CLIP_ORIGIN), "GL_CLIP_ORIGIN", allow_zero_padding=True
    )
    facts["clip_depth_mode"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_CLIP_DEPTH_MODE), "GL_CLIP_DEPTH_MODE", allow_zero_padding=True
    )
    facts["pack_alignment"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT), "GL_PACK_ALIGNMENT"
    )
    facts["pack_row_length"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_PACK_ROW_LENGTH), "GL_PACK_ROW_LENGTH"
    )
    facts["pack_skip_rows"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_PACK_SKIP_ROWS), "GL_PACK_SKIP_ROWS"
    )
    facts["pack_skip_pixels"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_PACK_SKIP_PIXELS), "GL_PACK_SKIP_PIXELS"
    )
    facts["pixel_pack_buffer_binding"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING), "GL_PIXEL_PACK_BUFFER_BINDING"
    )
    facts["read_framebuffer_binding"] = _strict_gl_integer(
        GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING), "GL_READ_FRAMEBUFFER_BINDING"
    )
    facts["read_buffer"] = _strict_gl_integer(GL.glGetIntegerv(GL.GL_READ_BUFFER), "GL_READ_BUFFER")
    if facts["readPixelFormat"] != facts["gl_rgb"]:
        raise SharedRasterFailure("SDK readPixelFormat is not GL_RGB")
    # MuJoCo enum mjDEPTH_ZEROFAR has integer value 1 in the bound SDK.
    if facts["readDepthMap"] != 1:
        raise SharedRasterFailure("SDK readDepthMap is not mjDEPTH_ZEROFAR")
    if (
        facts["pack_alignment"],
        facts["pack_row_length"],
        facts["pack_skip_rows"],
        facts["pack_skip_pixels"],
        facts["pixel_pack_buffer_binding"],
    ) != (1, 0, 0, 0, 0):
        raise SharedRasterFailure("GL pixel packing state differs")
    attachments = cast(Mapping[str, object], facts["offscreen_attachments"])
    main = cast(Mapping[str, object], attachments["offFBO"])
    resolve = cast(Mapping[str, object], attachments["offFBO_r"])
    if (
        facts["read_framebuffer_binding"] != main.get("object_name")
        or resolve.get("present") is not False
    ):
        raise SharedRasterFailure("read framebuffer is not the unresolved offFBO")
    return facts


def shared_context_binding(facts: Mapping[str, object]) -> dict[str, object]:
    value = context_runtime_binding(facts)
    for key in (
        "readPixelFormat",
        "readDepthMap",
        "gl_rgb",
        "pack_alignment",
        "pack_row_length",
        "pack_skip_rows",
        "pack_skip_pixels",
        "pixel_pack_buffer_binding",
        "read_buffer",
    ):
        if key not in facts:
            raise SharedRasterFailure("shared context binding is incomplete")
        value[key] = facts[key]
    return value


def _pointer_identity(value: object) -> int:
    try:
        pointer = ctypes.cast(cast(Any, value), ctypes.c_void_p).value
    except (TypeError, ValueError) as exc:
        raise SharedRasterFailure("OSMesa context pointer is unavailable") from exc
    if pointer is None or pointer <= 0:
        raise SharedRasterFailure("OSMesa context pointer is unavailable")
    return int(pointer)


def observe_native_read_state(renderer: Any) -> dict[str, object]:
    """Observe current draw/read state without making a context current."""
    import mujoco
    from OpenGL import GL
    from OpenGL import osmesa as GL_OSMESA

    from epsbench.diagnostics.gl_provenance import inspect_mujoco_offscreen_attachments
    from epsbench.diagnostics.revision_mujoco import _study_depth_attachments

    scene, model, rect, mjr = (
        renderer._scene,
        renderer._model,
        renderer._rect,
        renderer._mjr_context,
    )
    gl_context = getattr(renderer, "_gl_context", None)
    expected_pointer = getattr(gl_context, "_context", None)
    actual_pointer = GL_OSMESA.OSMesaGetCurrentContext()
    query_bindings_before = {
        "read_framebuffer": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING), "GL_READ_FRAMEBUFFER_BINDING"
        ),
        "draw_framebuffer": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING), "GL_DRAW_FRAMEBUFFER_BINDING"
        ),
        "renderbuffer": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_RENDERBUFFER_BINDING), "GL_RENDERBUFFER_BINDING"
        ),
    }
    attachments = dict(inspect_mujoco_offscreen_attachments(renderer))
    query_bindings_after = {
        "read_framebuffer": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING), "GL_READ_FRAMEBUFFER_BINDING"
        ),
        "draw_framebuffer": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING), "GL_DRAW_FRAMEBUFFER_BINDING"
        ),
        "renderbuffer": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_RENDERBUFFER_BINDING), "GL_RENDERBUFFER_BINDING"
        ),
    }
    if query_bindings_after != query_bindings_before:
        raise SharedRasterFailure("attachment query did not restore GL bindings")
    if _pointer_identity(GL_OSMESA.OSMesaGetCurrentContext()) != _pointer_identity(actual_pointer):
        raise SharedRasterFailure("attachment query changed the current OSMesa context")
    attachment_records = cast(dict[str, object], attachments["offscreen_attachments"])
    for name, depth in _study_depth_attachments(renderer).items():
        cast(dict[str, object], attachment_records[name])["depth"] = depth
    mapping = [
        {
            "segid_plus_one": int(g.segid) + 1,
            "objid": int(g.objid),
            "objtype": int(g.objtype),
        }
        for g in scene.geoms[: int(scene.ngeom)]
        if int(g.segid) != -1
    ]
    cameras = [
        {
            "pos": np.asarray(camera.pos, dtype=np.float32).tolist(),
            "forward": np.asarray(camera.forward, dtype=np.float32).tolist(),
            "up": np.asarray(camera.up, dtype=np.float32).tolist(),
            "frustum_near": float(camera.frustum_near),
            "frustum_far": float(camera.frustum_far),
            "frustum_top": float(camera.frustum_top),
            "frustum_bottom": float(camera.frustum_bottom),
            "frustum_center": float(camera.frustum_center),
            "frustum_width": float(camera.frustum_width),
        }
        for camera in scene.camera
    ]
    projection = np.asarray(GL.glGetFloatv(GL.GL_PROJECTION_MATRIX), dtype=np.float32)
    modelview = np.asarray(GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX), dtype=np.float32)
    if projection.size != 16 or modelview.size != 16:
        raise SharedRasterFailure("post-draw projection/modelview matrix shape differs")
    if not np.isfinite(projection).all() or not np.isfinite(modelview).all():
        raise SharedRasterFailure("post-draw projection/modelview matrix is nonfinite")
    return {
        "actual_current_context": _pointer_identity(actual_pointer),
        "expected_current_context": _pointer_identity(expected_pointer),
        "rect": [int(rect.left), int(rect.bottom), int(rect.width), int(rect.height)],
        "scene_flags": np.asarray(scene.flags, dtype=np.uint8).tolist(),
        "ngeom": int(scene.ngeom),
        "scene_map": mapping,
        "scene_cameras": cameras,
        "projection_matrix_float32": projection.reshape(-1).tolist(),
        "modelview_matrix_float32": modelview.reshape(-1).tolist(),
        "framewidth": float(scene.framewidth),
        "stereo": int(scene.stereo),
        "stereo_none": int(mujoco.mjtStereo.mjSTEREO_NONE),
        "rnd_depth": bool(scene.flags[int(mujoco.mjtRndFlag.mjRND_DEPTH)]),
        "segment_enabled": bool(scene.flags[int(mujoco.mjtRndFlag.mjRND_SEGMENT)]),
        "idcolor_enabled": bool(scene.flags[int(mujoco.mjtRndFlag.mjRND_IDCOLOR)]),
        "near": float(model.vis.map.znear * model.stat.extent),
        "far": float(model.vis.map.zfar * model.stat.extent),
        "extent": float(model.stat.extent),
        "readPixelFormat": int(mjr.readPixelFormat),
        "readDepthMap": int(mjr.readDepthMap),
        "gl_rgb": int(GL.GL_RGB),
        "depth_zerofar": int(mujoco.mjtDepthMap.mjDEPTH_ZEROFAR),
        "mjr_currentBuffer": int(mjr.currentBuffer),
        "framebuffer_offscreen": int(mujoco.mjtFramebuffer.mjFB_OFFSCREEN),
        "offSamples": int(mjr.offSamples),
        "offFBO": int(mjr.offFBO),
        "offFBO_r": int(mjr.offFBO_r),
        "offWidth": int(mjr.offWidth),
        "offHeight": int(mjr.offHeight),
        "offscreen_attachments": attachment_records,
        "query_bindings_restored": True,
        "read_framebuffer_binding": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING),
            "GL_READ_FRAMEBUFFER_BINDING",
        ),
        "draw_framebuffer_binding": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING),
            "GL_DRAW_FRAMEBUFFER_BINDING",
        ),
        "read_buffer": _strict_gl_integer(GL.glGetIntegerv(GL.GL_READ_BUFFER), "GL_READ_BUFFER"),
        "draw_buffer": _strict_gl_integer(GL.glGetIntegerv(GL.GL_DRAW_BUFFER), "GL_DRAW_BUFFER"),
        "clip_origin": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_CLIP_ORIGIN),
            "GL_CLIP_ORIGIN",
            allow_zero_padding=True,
        ),
        "clip_depth_mode": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_CLIP_DEPTH_MODE),
            "GL_CLIP_DEPTH_MODE",
            allow_zero_padding=True,
        ),
        "pack_alignment": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT), "GL_PACK_ALIGNMENT"
        ),
        "pack_row_length": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_PACK_ROW_LENGTH), "GL_PACK_ROW_LENGTH"
        ),
        "pack_skip_rows": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_PACK_SKIP_ROWS), "GL_PACK_SKIP_ROWS"
        ),
        "pack_skip_pixels": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_PACK_SKIP_PIXELS), "GL_PACK_SKIP_PIXELS"
        ),
        "pixel_pack_buffer_binding": _strict_gl_integer(
            GL.glGetIntegerv(GL.GL_PIXEL_PACK_BUFFER_BINDING),
            "GL_PIXEL_PACK_BUFFER_BINDING",
        ),
    }


def validate_paired_draw_state(
    state: Mapping[str, object],
    *,
    expected_context: int,
    expected_read_framebuffer: int,
    expected_read_buffer: int,
    expected_context_facts: Mapping[str, object],
) -> None:
    """Enforce absolute paired-draw invariants in live and retained evidence."""
    if state.get("actual_current_context") != expected_context:
        raise SharedRasterFailure("paired draw does not use the expected current context")
    if state.get("expected_current_context") != expected_context:
        raise SharedRasterFailure("renderer OSMesa context identity differs")
    if state.get("framewidth") != 0.0:
        raise SharedRasterFailure("paired draw framewidth is not zero")
    if state.get("stereo") != state.get("stereo_none"):
        raise SharedRasterFailure("paired draw stereo mode is not NONE")
    if state.get("rnd_depth") is not False:
        raise SharedRasterFailure("paired draw enables the native depth-redraw path")
    if state.get("segment_enabled") is not True or state.get("idcolor_enabled") is not True:
        raise SharedRasterFailure("paired draw lacks SEGMENT and IDCOLOR flags")
    if state.get("readPixelFormat") != state.get("gl_rgb"):
        raise SharedRasterFailure("paired draw readPixelFormat is not GL_RGB")
    if state.get("readDepthMap") != state.get("depth_zerofar"):
        raise SharedRasterFailure("paired draw readDepthMap is not mjDEPTH_ZEROFAR")
    packing = (
        state.get("pack_alignment"),
        state.get("pack_row_length"),
        state.get("pack_skip_rows"),
        state.get("pack_skip_pixels"),
        state.get("pixel_pack_buffer_binding"),
    )
    if packing != (1, 0, 0, 0, 0):
        raise SharedRasterFailure("paired draw pixel packing/PBO state differs")
    if state.get("query_bindings_restored") is not True:
        raise SharedRasterFailure("paired draw query bindings were not restored")
    if state.get("mjr_currentBuffer") != state.get("framebuffer_offscreen"):
        raise SharedRasterFailure("paired draw mjr currentBuffer is not offscreen")
    if state.get("offSamples") != 0 or state.get("offFBO_r") != 0:
        raise SharedRasterFailure("paired draw is sampled or has a resolve FBO")
    if state.get("offFBO") != expected_read_framebuffer:
        raise SharedRasterFailure("paired draw mjr offFBO identity differs")
    if state.get("read_framebuffer_binding") != expected_read_framebuffer:
        raise SharedRasterFailure("paired draw read framebuffer is not expected offFBO")
    if state.get("draw_framebuffer_binding") != expected_read_framebuffer:
        raise SharedRasterFailure("paired draw draw framebuffer is not expected offFBO")
    if state.get("read_buffer") != expected_read_buffer:
        raise SharedRasterFailure("paired draw read buffer differs from expected buffer")
    for key in ("offWidth", "offHeight"):
        expected_key = "mjr_off_width" if key == "offWidth" else "mjr_off_height"
        if state.get(key) != expected_context_facts.get(expected_key):
            raise SharedRasterFailure("paired draw framebuffer dimensions differ")
    if state.get("offscreen_attachments") != expected_context_facts.get("offscreen_attachments"):
        raise SharedRasterFailure("paired draw attachment identities/storage/samples differ")
    cameras = state.get("scene_cameras")
    if not isinstance(cameras, list) or not cameras:
        raise SharedRasterFailure("paired draw scene camera/frustum facts are absent")
    for key in ("projection_matrix_float32", "modelview_matrix_float32"):
        matrix = state.get(key)
        if not isinstance(matrix, list) or len(matrix) != 16:
            raise SharedRasterFailure("paired draw projection provenance differs")


def stable_paired_draw_state(state: Mapping[str, object]) -> dict[str, object]:
    """Project deterministic per-draw facts; volatile GL/context handles stay in receipts."""
    keys = (
        "rect",
        "scene_flags",
        "ngeom",
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
        "stereo_none",
        "rnd_depth",
        "segment_enabled",
        "idcolor_enabled",
        "gl_rgb",
        "depth_zerofar",
        "mjr_currentBuffer",
        "framebuffer_offscreen",
        "offSamples",
        "offWidth",
        "offHeight",
        "query_bindings_restored",
    )
    if any(key not in state for key in keys):
        raise SharedRasterFailure("paired stable draw-state projection is incomplete")
    return {key: state[key] for key in keys}


class Observer(Protocol):
    def __call__(
        self, renderer: object, model: object, width: int, height: int
    ) -> Mapping[str, object]: ...


class _RendererProxy:
    def __init__(self, owner: SharedRasterRendererAdapter, renderer: Any, context_index: int):
        self._owner, self._renderer, self._context_index = owner, renderer, context_index
        self._thread = threading.get_ident()
        self._last_update: dict[str, object] | None = None
        self._rendering = False
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._renderer, name)

    def update_scene(self, data: object, *args: object, **kwargs: object) -> object:
        option: Any = kwargs.get("scene_option")
        self._last_update = {
            "data_identity": id(data),
            "camera": kwargs.get("camera", args[0] if args else None),
            "scene_option_identity": None if option is None else id(option),
            "geomgroup": None
            if option is None
            else np.asarray(option.geomgroup).astype(int).tolist(),
        }
        return self._renderer.update_scene(data, *args, **kwargs)

    def _modality(self) -> str:
        if bool(getattr(self._renderer, "_depth_rendering", False)):
            return "depth"
        if bool(getattr(self._renderer, "_segmentation_rendering", False)):
            if self._last_update and self._last_update.get("geomgroup") is not None:
                group = cast(list[int], self._last_update["geomgroup"])
                if len(group) > 1 and group[1] == 0:
                    return "counterfactual_segmentation"
            return "segmentation"
        return "rgb"

    def render(self, *args: object, **kwargs: object) -> object:
        if threading.get_ident() != self._thread or self._rendering:
            raise SharedRasterFailure("renderer render thread/reentrancy differs")
        if self._last_update is None:
            raise SharedRasterFailure("render lacks preceding scene update")
        sequence = len(self._owner.native_events)
        expected = (
            self._owner.expected_modalities[sequence]
            if sequence < len(self._owner.expected_modalities)
            else None
        )
        modality = self._modality()
        if (
            modality != expected
            or self._context_index != sequence // self._owner.attempt.calls_per_episode
        ):
            raise SharedRasterFailure("renderer modality/context sequence differs")
        pristine = _PRISTINE.get(id(self._owner.mujoco))
        if pristine is None:
            raise SharedRasterFailure("pristine SDK entrypoints not bound")
        mujoco, original_render, original_read = self._owner.mujoco, pristine[1], pristine[2]
        if mujoco.mjr_render is not original_render or mujoco.mjr_readPixels is not original_read:
            raise SharedRasterFailure("native entrypoints differ before scoped call")
        pose = (sequence % self._owner.attempt.calls_per_episode) // (
            4 if self._owner.attempt.family == "single_occluder" else 3
        )
        event: dict[str, object] = {
            "sequence": sequence,
            "context_index": self._context_index,
            "frame": pose,
            "modality": modality,
            "thread": self._thread,
            "renderer_identity": id(self._renderer),
            "scene_update": dict(self._last_update),
            "render_calls": 0,
            "readback_calls": 0,
        }
        raw_pair: tuple[np.ndarray, np.ndarray, list[dict[str, int]], dict[str, object]] | None = (
            None
        )
        render_scene_map: list[dict[str, int]] | None = None
        update_snapshot = dict(self._last_update)

        def render_hook(rect: object, scene: object, context: object) -> object:
            nonlocal render_scene_map
            if event["render_calls"] != 0:
                raise SharedRasterFailure("duplicate native render rejected before call")
            if (
                threading.get_ident() != self._thread
                or rect is not self._renderer._rect
                or scene is not self._renderer._scene
                or context is not self._renderer._mjr_context
            ):
                raise SharedRasterFailure("native render identity/thread differs before call")
            flags = np.asarray(scene.flags, dtype=bool)
            segment = int(mujoco.mjtRndFlag.mjRND_SEGMENT)
            idcolor = int(mujoco.mjtRndFlag.mjRND_IDCOLOR)
            if modality == "depth" and (not flags[segment] or not flags[idcolor]):
                raise SharedRasterFailure("depth draw lacks SEGMENT and IDCOLOR flags")
            render_scene_map = [
                {
                    "segid_plus_one": int(g.segid) + 1,
                    "objid": int(g.objid),
                    "objtype": int(g.objtype),
                }
                for g in scene.geoms[: int(scene.ngeom)]
                if int(g.segid) != -1
            ]
            event.update(
                {
                    "render_calls": 1,
                    "rect_identity": id(rect),
                    "scene_identity": id(scene),
                    "context_identity": id(context),
                    "scene_flags": flags.astype(int).tolist(),
                    "draw_input_scene_flags": flags.astype(int).tolist(),
                    "draw_input_scene_map": render_scene_map,
                    "segment_enabled": bool(flags[segment]),
                    "idcolor_enabled": bool(flags[idcolor]),
                    "ngeom": int(scene.ngeom),
                }
            )
            result = cast(Callable[..., object], original_render)(rect, scene, context)
            draw_state = self._owner.native_state_observer(self._renderer)
            if modality == "depth":
                context_facts = self._owner.contexts[self._context_index]
                main = cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], context_facts["offscreen_attachments"])["offFBO"],
                )
                validate_paired_draw_state(
                    draw_state,
                    expected_context=int(cast(int, context_facts["osmesa_context_identity"])),
                    expected_read_framebuffer=int(cast(int, main["object_name"])),
                    expected_read_buffer=int(cast(int, context_facts["read_buffer"])),
                    expected_context_facts=context_facts,
                )
            event["draw_state"] = draw_state
            return result

        def read_hook(rgb: object, depth: object, rect: object, context: object) -> object:
            nonlocal raw_pair
            if event["readback_calls"] != 0 or event["render_calls"] != 1:
                raise SharedRasterFailure("native readback order/count differs")
            if (
                threading.get_ident() != self._thread
                or rect is not self._renderer._rect
                or context is not self._renderer._mjr_context
            ):
                raise SharedRasterFailure("native readback identity/thread differs")
            if self._last_update != update_snapshot:
                raise SharedRasterFailure("scene update state drifted between draw and read")
            current_map = [
                {
                    "segid_plus_one": int(g.segid) + 1,
                    "objid": int(g.objid),
                    "objtype": int(g.objtype),
                }
                for g in self._renderer._scene.geoms[: int(self._renderer._scene.ngeom)]
                if int(g.segid) != -1
            ]
            if render_scene_map is None or current_map != render_scene_map:
                raise SharedRasterFailure("scene ID map drifted between draw and read")
            read_state_before = self._owner.native_state_observer(self._renderer)
            event["read_state_before"] = read_state_before
            if read_state_before != event.get("draw_state"):
                raise SharedRasterFailure("native state drifted between draw and read")
            if modality == "depth":
                context_facts = self._owner.contexts[self._context_index]
                main = cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], context_facts["offscreen_attachments"])["offFBO"],
                )
                validate_paired_draw_state(
                    read_state_before,
                    expected_context=int(cast(int, context_facts["osmesa_context_identity"])),
                    expected_read_framebuffer=int(cast(int, main["object_name"])),
                    expected_read_buffer=int(cast(int, context_facts["read_buffer"])),
                    expected_context_facts=context_facts,
                )
            height, width = int(self._renderer.height), int(self._renderer.width)
            if modality == "depth":
                if rgb is not None or not isinstance(depth, np.ndarray):
                    raise SharedRasterFailure("ordinary depth readback roles differ")
                _array_ok(depth, (height, width), np.dtype(np.float32), "depth destination")
                candidate = np.empty((height, width, 3), dtype=np.uint8)
                event["readback_calls"] = 1
                event["readback_roles"] = {
                    "color": "candidate_native_id_rgb_uint8_hwc3",
                    "depth": "canonical_native_depth_float32_hw",
                }
                result = cast(Callable[..., object], original_read)(candidate, depth, rect, context)
                read_state_after = self._owner.native_state_observer(self._renderer)
                event["read_state_after"] = read_state_after
                if read_state_after != read_state_before:
                    raise SharedRasterFailure("native state drifted across paired readback")
                mapping = render_scene_map
                draw_state = cast(dict[str, object], event["draw_state"])
                raw_pair = (
                    np.ascontiguousarray(candidate.copy()),
                    np.ascontiguousarray(depth.copy()),
                    mapping,
                    dict(draw_state),
                )
                return result
            if not isinstance(rgb, np.ndarray) or depth is not None:
                raise SharedRasterFailure("RGB/segmentation readback roles differ")
            event["readback_calls"] = 1
            event["readback_roles"] = {"color": "canonical_uint8_hwc3", "depth": None}
            result = cast(Callable[..., object], original_read)(rgb, depth, rect, context)
            read_state_after = self._owner.native_state_observer(self._renderer)
            event["read_state_after"] = read_state_after
            if read_state_after != read_state_before:
                raise SharedRasterFailure("native state drifted across readback")
            return result

        self._rendering = True
        error: BaseException | None = None
        result: object = None
        try:
            mujoco.mjr_render, mujoco.mjr_readPixels = render_hook, read_hook
            result = self._renderer.render(*args, **kwargs)
            if modality == "depth":
                if raw_pair is None or not isinstance(result, np.ndarray):
                    raise SharedRasterFailure("paired buffers missing after depth render")
                self._owner.record_depth(self._context_index, pose, raw_pair, result)
            elif modality == "segmentation":
                if not isinstance(result, np.ndarray):
                    raise SharedRasterFailure("canonical segmentation output missing")
                self._owner.record_segmentation(self._context_index, pose, result)
        except BaseException as exc:
            error = exc
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            mujoco.mjr_render, mujoco.mjr_readPixels = original_render, original_read
            self._rendering = False
            event["restored"] = (
                mujoco.mjr_render is original_render and mujoco.mjr_readPixels is original_read
            )
            event["validation_failures"] = (
                []
                if event["render_calls"] == 1 and event["readback_calls"] == 1 and event["restored"]
                else ["native call/restoration validation failed"]
            )
            self._owner.native_events.append(event)
        if error is not None:
            raise error
        if event["validation_failures"]:
            raise SharedRasterFailure("native call/restoration validation failed")
        return result

    def close(self) -> None:
        if not self._closed:
            self._renderer.close()
            self._closed = True
            self._owner.contexts[self._context_index]["close"] = {"complete": True}


class SharedRasterRendererAdapter:
    """Scope the original constructor and pair ordinary depth readbacks."""

    def __init__(
        self,
        mujoco: Any,
        observe: Observer,
        attempt: Attempt,
        pair_root: Path,
        context_baseline: Mapping[str, object] | None = None,
        native_state_observer: Callable[[Any], dict[str, object]] = observe_native_read_state,
    ):
        self.mujoco, self.observe, self.attempt, self.pair_root = (
            mujoco,
            observe,
            attempt,
            pair_root,
        )
        self.context_baseline = dict(context_baseline) if context_baseline is not None else None
        self.native_state_observer = native_state_observer
        self.thread = threading.get_ident()
        self.contexts: list[dict[str, object]] = []
        self.native_events: list[dict[str, object]] = []
        self.proxies: list[_RendererProxy] = []
        self._pending: dict[tuple[int, int], dict[str, np.ndarray | object]] = {}
        self.restored = self.cleanup_complete = False
        self.constructor_failures: list[dict[str, object]] = []
        self._original: object | None = None
        self.expected_modalities = [
            m
            for _e in range(4)
            for _p in range(2)
            for m in (
                ("rgb", "depth", "segmentation", "counterfactual_segmentation")
                if attempt.family == "single_occluder"
                else ("rgb", "depth", "segmentation")
            )
        ]

    def __enter__(self) -> SharedRasterRendererAdapter:
        pristine = _PRISTINE.get(id(self.mujoco))
        if pristine is None or not _HOOK_LOCK.acquire(blocking=False):
            raise SharedRasterFailure("pristine binding absent or renderer hook active")
        self._original = pristine[0]
        if self.mujoco.Renderer is not self._original:
            _HOOK_LOCK.release()
            raise SharedRasterFailure("pre-existing Renderer replacement")

        def constructor(model: Any, *args: Any, **kwargs: Any) -> _RendererProxy:
            if threading.get_ident() != self.thread or len(self.contexts) >= 4:
                raise SharedRasterFailure("Renderer constructor thread/count differs")
            model.vis.quality.offsamples = 0
            renderer = cast(Callable[..., Any], self._original)(model, *args, **kwargs)
            try:
                facts = dict(
                    self.observe(renderer, model, int(renderer.width), int(renderer.height))
                )
                baseline = self.context_baseline or (
                    shared_context_binding(self.contexts[0]) if self.contexts else None
                )
                if baseline is not None and shared_context_binding(facts) != baseline:
                    raise SharedRasterFailure("OSMesa context runtime differs")
                facts.update({"context_index": len(self.contexts), "thread": self.thread})
                self.contexts.append(facts)
                proxy = _RendererProxy(self, renderer, len(self.contexts) - 1)
                self.proxies.append(proxy)
                return proxy
            except BaseException:
                renderer.close()
                raise

        self.mujoco.Renderer = constructor
        return self

    def __exit__(self, typ: object, value: object, tb: object) -> Literal[False]:
        close_error: BaseException | None = None
        try:
            self.mujoco.Renderer = self._original
            for proxy in reversed(self.proxies):
                if not proxy._closed:
                    try:
                        proxy.close()
                    except BaseException as exc:
                        self.constructor_failures.append(
                            {
                                "context_index": proxy._context_index,
                                "close_complete": False,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                        )
                        if close_error is None:
                            close_error = exc
            self.restored = self.mujoco.Renderer is self._original
            self.cleanup_complete = all(proxy._closed for proxy in self.proxies)
        finally:
            _HOOK_LOCK.release()
        if value is None and close_error is not None:
            raise close_error
        return False

    def record_depth(
        self,
        episode: int,
        frame: int,
        raw: tuple[np.ndarray, np.ndarray, list[dict[str, int]], dict[str, object]],
        canonical_depth: np.ndarray,
    ) -> None:
        raw_rgb, raw_native, mapping, draw_state = raw
        projection = draw_state
        rgb = np.ascontiguousarray(np.flipud(raw_rgb))
        native = np.ascontiguousarray(np.flipud(raw_native))
        _segids, objid, objtype = decode_id_colors(rgb, mapping)
        converted = convert_native_depth(
            native, float(cast(float, projection["near"])), float(cast(float, projection["far"]))
        )
        canonical_depth = np.ascontiguousarray(canonical_depth.copy())
        if not np.array_equal(converted, canonical_depth):
            raise SharedRasterFailure("paired converted depth differs from canonical SDK output")
        self._pending[(episode, frame)] = {
            "raw_rgb": raw_rgb,
            "raw_native": raw_native,
            "rgb": rgb,
            "native": native,
            "objid": objid,
            "objtype": objtype,
            "converted": converted,
            "canonical_depth": canonical_depth,
            "mapping": mapping,
            "projection": projection,
        }

    def record_segmentation(
        self, episode: int, frame: int, canonical_segmentation: np.ndarray
    ) -> None:
        key = (episode, frame)
        item = self._pending.get(key)
        if item is None:
            raise SharedRasterFailure("canonical segmentation lacks preceding paired depth")
        segmentation = np.ascontiguousarray(canonical_segmentation.copy())
        if segmentation.shape[-1:] != (2,) or segmentation.dtype != np.int32:
            raise SharedRasterFailure("canonical segmentation shape/dtype differs")
        if not np.array_equal(
            np.asarray(item["objid"]), segmentation[:, :, 0]
        ) or not np.array_equal(np.asarray(item["objtype"]), segmentation[:, :, 1]):
            raise SharedRasterFailure("paired IDs differ from canonical ordinary SDK segmentation")
        directory = self.pair_root / f"episode-{episode:06d}" / f"frame-{frame}"
        permission = PairedEvidencePermission.authorized()
        permission.require()
        directory.mkdir(parents=True, exist_ok=False)
        arrays = {
            "native_id_rgb": item["raw_rgb"],
            "native_depth_pre_metric": item["raw_native"],
            "decoded_objid": item["objid"],
            "decoded_objtype": item["objtype"],
            "converted_depth": item["converted"],
            "canonical_sdk_segmentation": segmentation,
            "canonical_sdk_depth": item["canonical_depth"],
        }
        hashes: dict[str, str] = {}
        for name, value in arrays.items():
            path = directory / f"{name}.npy"
            with path.open("xb") as handle:
                np.save(handle, cast(np.ndarray, value), allow_pickle=False)
            hashes[name] = digest_file(path)
        projection = cast(dict[str, object], item["projection"])
        c_coef, d_coef = depth_coefficients(
            float(cast(float, projection["near"])), float(cast(float, projection["far"]))
        )
        context = self.contexts[episode]
        modalities_per_pose = 4 if self.attempt.family == "single_occluder" else 3
        native_event_sequence = (
            episode * self.attempt.calls_per_episode + frame * modalities_per_pose + 1
        )
        metadata = {
            "schema": PAIR_SCHEMA,
            "episode": episode,
            "frame": frame,
            "context_logical": f"episode-{episode:06d}",
            "native_event_modality": "depth",
            "native_event_sequence": native_event_sequence,
            "framebuffer": {
                "role": "offFBO",
                "resolve_present": False,
                "actual_offsamples": context["actual_offsamples"],
                "sample_buffers": context["sample_buffers"],
                "samples": context["samples"],
                "color_storage_dimensions": context["color_storage_dimensions"],
                "depth_storage_dimensions": context["depth_storage_dimensions"],
                "attachment_format": context["attachment_format"],
                "attachment_component_type": context["attachment_component_type"],
            },
            "orientation": (
                "native arrays preserve readback orientation; one vertical flip is "
                "applied equally before ID decode and depth conversion"
            ),
            "readback_atomicity": "sequential color/depth GL reads; not hardware-atomic",
            "depth_term": "native SDK readback before metric conversion",
            "scene_id_map": item["mapping"],
            "draw_state": stable_paired_draw_state(projection),
            "projection": {
                **{key: projection[key] for key in ("near", "far", "extent")},
                "c_coef_float32": float(c_coef),
                "d_coef_float32": float(d_coef),
                "inverse_precision": "float64",
                "final_dtype": "float32",
            },
            "arrays_sha256": hashes,
            "compatibility": {"canonical_sdk_ids_exact": True, "canonical_sdk_depth_exact": True},
            "interpretation": (
                "shared raster correspondence candidate; geometric ownership and "
                "metric accuracy are separate and unqualified"
            ),
        }
        publish_bytes(directory / "pair.json", canonical(metadata))
        del self._pending[key]

    def validate_complete(self) -> None:
        if (
            len(self.contexts) != 4
            or [e["modality"] for e in self.native_events] != self.expected_modalities
            or self._pending
        ):
            raise SharedRasterFailure("context/render/pair completion differs")
        if (
            not self.restored
            or not self.cleanup_complete
            or any(e["validation_failures"] for e in self.native_events)
        ):
            raise SharedRasterFailure("renderer/native restoration differs")
        pairs = list(self.pair_root.glob("episode-*/frame-*/pair.json"))
        if len(pairs) != 8:
            raise SharedRasterFailure("paired endpoint membership differs")


def bind_pristine_sdk(mujoco: Any) -> dict[str, object]:
    runtime = qualification_bind_pristine_sdk(mujoco)
    module = __import__("mujoco.rendering.classic.renderer", fromlist=["Renderer"])
    native = __import__("mujoco._render", fromlist=["mjr_render", "mjr_readPixels"])
    if (
        str(mujoco.__version__) != SDK_VERSION
        or digest_file(Path(cast(str, module.__file__)).resolve()) != SDK_RENDERER_SHA256
    ):
        raise SharedRasterFailure("MuJoCo source/version differs")
    _PRISTINE[id(mujoco)] = (mujoco.Renderer, native.mjr_render, native.mjr_readPixels)
    return runtime


def pair_artifact_manifest(root: Path) -> list[dict[str, object]]:
    if not root.is_dir():
        raise SharedRasterFailure("paired artifact root missing")
    return [
        {
            "path": p.relative_to(root).as_posix(),
            "sha256": digest_file(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def dataset_artifact_manifest(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": p.relative_to(root).as_posix(),
            "sha256": digest_file(p),
            "bytes": p.stat().st_size,
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def _verify_manifest(root: Path, recorded: object) -> None:
    if recorded != dataset_artifact_manifest(root):
        raise SharedRasterFailure("artifact membership or hash differs")


def deterministic_pair_members(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in sorted(root.rglob("*")) if p.is_file()]


def compare_repeat_pairs(left: Path, right: Path) -> list[str]:
    left_members, right_members = (
        deterministic_pair_members(left),
        deterministic_pair_members(right),
    )
    if left_members != right_members:
        raise SharedRasterFailure("repeat paired member names differ")
    for name in left_members:
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise SharedRasterFailure(f"repeat paired artifact differs: {name}")
    return left_members


def validate_pair_tree(root: Path, dataset: Path | None = None) -> dict[str, object]:
    permission = PairedEvidencePermission.authorized()
    access = PairedArtifactAccess(root, permission)
    expected = {(episode, frame) for episode in range(4) for frame in range(2)}
    loader: Any | None = None
    if dataset is not None:
        from epsbench.data import DatasetLoader
        from epsbench.schema import ModalityPermissionSet

        loader = DatasetLoader(dataset, ModalityPermissionSet.all_modalities())
    observed: set[tuple[int, int]] = set()
    for episode, frame in sorted(expected):
        metadata = access.metadata(episode, frame)
        if (
            metadata.get("schema") != PAIR_SCHEMA
            or metadata.get("episode") != episode
            or metadata.get("frame") != frame
        ):
            raise SharedRasterFailure("paired metadata identity differs")
        arrays = {
            name: access.array(episode, frame, name)
            for name in cast(Mapping[str, str], metadata["arrays_sha256"])
        }
        directory = root / f"episode-{episode:06d}" / f"frame-{frame}"
        for name, expected_hash in cast(Mapping[str, str], metadata["arrays_sha256"]).items():
            if digest_file(directory / f"{name}.npy") != expected_hash:
                raise SharedRasterFailure("paired array hash differs")
        raw_rgb = arrays["native_id_rgb"]
        raw_native = arrays["native_depth_pre_metric"]
        rgb = np.ascontiguousarray(np.flipud(raw_rgb))
        native = np.ascontiguousarray(np.flipud(raw_native))
        _segids, objid, objtype = decode_id_colors(
            rgb, cast(Sequence[Mapping[str, int]], metadata["scene_id_map"])
        )
        projection = cast(Mapping[str, object], metadata["projection"])
        converted = convert_native_depth(
            native, float(cast(float, projection["near"])), float(cast(float, projection["far"]))
        )
        canonical_seg = arrays["canonical_sdk_segmentation"]
        if not np.array_equal(objid, arrays["decoded_objid"]) or not np.array_equal(
            objtype, arrays["decoded_objtype"]
        ):
            raise SharedRasterFailure("regenerated paired ID decode differs")
        if not np.array_equal(converted, arrays["converted_depth"]):
            raise SharedRasterFailure("regenerated paired depth conversion differs")
        if not np.array_equal(objid, canonical_seg[:, :, 0]) or not np.array_equal(
            objtype, canonical_seg[:, :, 1]
        ):
            raise SharedRasterFailure("stored paired/canonical IDs differ")
        if not np.array_equal(converted, arrays["canonical_sdk_depth"]):
            raise SharedRasterFailure("stored paired/canonical depth differs")
        if loader is not None:
            instrumentation = loader._instrumentation(episode)
            dataset_objid, _source = _raw_segmentation(loader, instrumentation, episode, frame)
            if not np.array_equal(objid, dataset_objid):
                raise SharedRasterFailure("paired IDs differ from generated dataset segmentation")
            if not np.array_equal(
                converted, np.asarray(loader.read_depth(episode, frame), dtype=np.float32)
            ):
                raise SharedRasterFailure("paired depth differs from generated dataset depth")
        observed.add((episode, frame))
    actual_metadata = {
        (int(p.parent.parent.name.split("-")[1]), int(p.parent.name.split("-")[1]))
        for p in root.glob("episode-*/frame-*/pair.json")
    }
    if observed != expected or actual_metadata != expected:
        raise SharedRasterFailure("paired endpoint membership differs")
    return {"endpoints": 8, "regenerated_exact": True, "compatibility_exact": True}


def paired_permission_probes(root: Path) -> dict[str, object]:
    denied: list[str] = []
    for permission in (
        PairedEvidencePermission(False, False),
        PairedEvidencePermission(True, False),
        PairedEvidencePermission(False, True),
    ):
        try:
            PairedArtifactAccess(root, permission)
        except PermissionError:
            denied.append(
                f"{int(permission.allow_depth)}{int(permission.allow_privileged_raw_ids)}"
            )
        else:
            raise SharedRasterFailure("paired permission denial failed")
    metadata_denied = PairedArtifactAccess(root, PairedEvidencePermission(True, True, False))
    try:
        metadata_denied.metadata(0, 0)
    except PermissionError:
        denied.append("110-metadata")
    else:
        raise SharedRasterFailure("paired metadata permission denial failed")
    access = PairedArtifactAccess(root, PairedEvidencePermission.authorized())
    metadata = access.metadata(0, 0)
    rgb = access.array(0, 0, "native_id_rgb")
    depth = access.array(0, 0, "native_depth_pre_metric")
    return {
        "denied_before_open": denied,
        "authorized_positive_probe_schema": metadata["schema"],
        "authorized_arrays": {"id_rgb": list(rgb.shape), "depth": list(depth.shape)},
    }


def _revision_path(root: Path, index: int) -> Path:
    return root / "shared-raster-ledger" / f"revision-{index:04d}.json"


def initialise_ledger(root: Path, binding: Mapping[str, object]) -> Path:
    if root.exists():
        raise SharedRasterFailure("shared-raster output root already exists")
    root.mkdir(parents=True)
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": 0,
        "predecessor_sha256": None,
        "event": "initialised",
        "plan": plan(),
        "binding": dict(binding),
    }
    publish_bytes(_revision_path(root, 0), canonical(record))
    return _revision_path(root, 0)


def validate_ledger(root: Path, *, allow_stopped: bool = False) -> list[dict[str, Any]]:
    directory = root / "shared-raster-ledger"
    paths = sorted(directory.glob("revision-*.json")) if directory.is_dir() else []
    if (
        not paths
        or set(directory.iterdir()) != set(paths)
        or paths != [_revision_path(root, i) for i in range(len(paths))]
    ):
        raise SharedRasterFailure("shared-raster ledger absent, foreign, or gapped")
    records: list[dict[str, Any]] = []
    predecessor: str | None = None
    next_ordinal, expecting_terminal = 0, False
    for index, path in enumerate(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema") != LEDGER_SCHEMA
            or record.get("revision") != index
            or record.get("predecessor_sha256") != predecessor
        ):
            raise SharedRasterFailure("shared-raster ledger hash chain invalid")
        predecessor = digest_file(path)
        records.append(record)
        if any(r.get("event") == "failed" for r in records[:-1]):
            raise SharedRasterFailure("failed shared-raster event is not terminal")
        if index == 0:
            if record.get("event") != "initialised":
                raise SharedRasterFailure("shared-raster initial event invalid")
            validate_plan(record.get("plan"))
        elif not expecting_terminal:
            if (
                next_ordinal >= len(fixed_attempts())
                or record.get("event") != "reserved"
                or record.get("attempt_ordinal") != next_ordinal
                or record.get("attempt_name") != fixed_attempts()[next_ordinal].name
            ):
                raise SharedRasterFailure("shared-raster reservation order invalid")
            expecting_terminal = True
        else:
            if (
                record.get("event") not in {"complete", "failed"}
                or record.get("attempt_ordinal") != next_ordinal
            ):
                raise SharedRasterFailure("shared-raster terminal event invalid")
            expecting_terminal = False
            next_ordinal += 1
    if not allow_stopped and records[-1]["event"] in {"reserved", "failed"}:
        raise SharedRasterFailure("reserved/failed shared-raster study permanently stops")
    return records


def append_revision(
    root: Path, event: str, attempt: Attempt, details: Mapping[str, object] | None = None
) -> Path:
    records = validate_ledger(root, allow_stopped=True)
    tail = records[-1]
    if event == "reserved":
        completed = [r["attempt_ordinal"] for r in records if r["event"] == "complete"]
        if tail["event"] in {"reserved", "failed"} or completed != list(range(attempt.ordinal)):
            raise SharedRasterFailure("study cannot reserve this attempt")
    elif event in {"complete", "failed"}:
        if tail.get("event") != "reserved" or tail.get("attempt_ordinal") != attempt.ordinal:
            raise SharedRasterFailure("terminal event lacks matching reservation")
    else:
        raise SharedRasterFailure("unknown shared-raster ledger event")
    index = len(records)
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": index,
        "predecessor_sha256": digest_file(_revision_path(root, index - 1)),
        "event": event,
        "attempt_ordinal": attempt.ordinal,
        "attempt_name": attempt.name,
        "details": dict(details or {}),
    }
    publish_bytes(_revision_path(root, index), canonical(record))
    return _revision_path(root, index)


def validate_recorded_attempt(
    output: Path,
    attempt: Attempt,
    *,
    context_baseline: Mapping[str, object] | None,
    expected_binding: Mapping[str, object],
) -> dict[str, object]:
    receipt_path = output / "shared-raster-receipts" / f"{attempt.name}.json"
    receipt = cast(dict[str, object], json.loads(receipt_path.read_text(encoding="utf-8")))
    validate_plan(receipt.get("plan"))
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != STATUS
        or receipt.get("attempt") != attempt.__dict__ | {"name": attempt.name}
        or receipt.get("source") != dict(expected_binding)
    ):
        raise SharedRasterFailure("recorded shared-raster receipt binding differs")
    dataset, pairs = output / "datasets" / attempt.name, output / "paired" / attempt.name
    _verify_manifest(dataset, receipt.get("dataset_artifacts"))
    _verify_manifest(pairs, receipt.get("paired_artifacts"))
    from epsbench.data import DatasetLoader
    from epsbench.schema import ModalityPermissionSet

    child = DatasetLoader(dataset, ModalityPermissionSet.all_modalities()).read_dataset_manifest()
    if receipt.get("dataset_manifest") != child.model_dump(mode="json"):
        raise SharedRasterFailure("receipt and child dataset manifest differ")
    validate_child_source_provenance(child, expected_binding)
    validate_pair_tree(pairs, dataset)
    contexts = cast(list[dict[str, object]], receipt.get("contexts"))
    if len(contexts) != 4:
        raise SharedRasterFailure("recorded context count differs")
    baseline = (
        dict(context_baseline)
        if context_baseline is not None
        else shared_context_binding(contexts[0])
    )
    if any(shared_context_binding(c) != baseline for c in contexts):
        raise SharedRasterFailure("recorded context binding differs")
    events = cast(list[dict[str, object]], receipt.get("native_events"))
    expected_modalities = [
        m
        for _e in range(4)
        for _p in range(2)
        for m in (
            ("rgb", "depth", "segmentation", "counterfactual_segmentation")
            if attempt.family == "single_occluder"
            else ("rgb", "depth", "segmentation")
        )
    ]
    if (
        len(events) != attempt.expected_native_calls
        or [e.get("modality") for e in events] != expected_modalities
        or sum(e.get("modality") == "depth" for e in events) != 8
    ):
        raise SharedRasterFailure("recorded native event counts/schedule differ")
    for index, event in enumerate(events):
        if (
            event.get("sequence") != index
            or event.get("context_index") != index // attempt.calls_per_episode
            or event.get("render_calls") != 1
            or event.get("readback_calls") != 1
            or event.get("restored") is not True
            or event.get("validation_failures") != []
        ):
            raise SharedRasterFailure("recorded native call fact differs")
        if event.get("modality") == "depth":
            draw_state = event.get("draw_state")
            if (
                not isinstance(draw_state, Mapping)
                or event.get("segment_enabled") is not True
                or event.get("idcolor_enabled") is not True
                or event.get("readback_roles")
                != {
                    "color": "candidate_native_id_rgb_uint8_hwc3",
                    "depth": "canonical_native_depth_float32_hw",
                }
            ):
                raise SharedRasterFailure("paired depth draw/read state missing")
            context = contexts[int(cast(int, event["context_index"]))]
            main = cast(
                Mapping[str, object],
                cast(Mapping[str, object], context["offscreen_attachments"])["offFBO"],
            )
            validate_paired_draw_state(
                draw_state,
                expected_context=int(cast(int, context["osmesa_context_identity"])),
                expected_read_framebuffer=int(cast(int, main["object_name"])),
                expected_read_buffer=int(cast(int, context["read_buffer"])),
                expected_context_facts=context,
            )
            if (
                event.get("read_state_before") != draw_state
                or event.get("read_state_after") != draw_state
            ):
                raise SharedRasterFailure("paired depth framebuffer/state association differs")
            if event.get("draw_input_scene_flags") != draw_state.get("scene_flags"):
                raise SharedRasterFailure("draw-input flags differ from post-draw paired state")
            if event.get("draw_input_scene_map") != draw_state.get("scene_map"):
                raise SharedRasterFailure("draw-input map differs from post-draw paired state")
            episode = int(cast(int, event["context_index"]))
            frame = int(cast(int, event["frame"]))
            pair = PairedArtifactAccess(pairs, PairedEvidencePermission.authorized()).metadata(
                episode, frame
            )
            if (
                pair.get("native_event_sequence") != index
                or pair.get("native_event_modality") != "depth"
                or pair.get("context_logical") != f"episode-{episode:06d}"
                or pair.get("frame") != frame
                or pair.get("scene_id_map") != draw_state.get("scene_map")
                or pair.get("draw_state") != stable_paired_draw_state(draw_state)
            ):
                raise SharedRasterFailure("paired sidecar/native-event association differs")
            projection = pair.get("projection")
            if not isinstance(projection, Mapping) or any(
                projection.get(key) != draw_state.get(key) for key in ("near", "far", "extent")
            ):
                raise SharedRasterFailure("paired sidecar/native projection association differs")
    if (
        receipt.get("constructor_restored") is not True
        or receipt.get("cleanup_complete") is not True
        or receipt.get("constructor_failures") != []
    ):
        raise SharedRasterFailure("recorded constructor cleanup/restoration differs")
    if receipt.get("permissions") != canonical_permission_probes(dataset):
        raise SharedRasterFailure("canonical permission evidence differs")
    if receipt.get("paired_permissions") != paired_permission_probes(pairs):
        raise SharedRasterFailure("paired permission evidence differs")
    assessment = receipt.get("assessment")
    expected_inspection = output / "inspections" / f"{attempt.name}-episode-000000.png"
    if not isinstance(assessment, Mapping):
        raise SharedRasterFailure("recorded assessment is absent")
    inspection = assessment.get("inspection")
    if (
        not isinstance(inspection, Mapping)
        or inspection.get("path") != expected_inspection.relative_to(output).as_posix()
        or inspection.get("sha256") != digest_file(expected_inspection)
    ):
        raise SharedRasterFailure("saved inspection evidence differs")
    return receipt


def validate_completed_prefix(
    output: Path, records: Sequence[Mapping[str, Any]], binding: Mapping[str, object]
) -> dict[str, object] | None:
    completed = [r for r in records if r.get("event") == "complete"]
    baseline: dict[str, object] | None = None
    for attempt, record in zip(fixed_attempts(), completed, strict=False):
        receipt_path = output / "shared-raster-receipts" / f"{attempt.name}.json"
        details = record.get("details")
        if (
            record.get("attempt_ordinal") != attempt.ordinal
            or not isinstance(details, Mapping)
            or details.get("receipt_sha256") != digest_file(receipt_path)
        ):
            raise SharedRasterFailure("completed prefix receipt binding differs")
        receipt = validate_recorded_attempt(
            output, attempt, context_baseline=baseline, expected_binding=binding
        )
        if baseline is None:
            baseline = shared_context_binding(cast(list[dict[str, object]], receipt["contexts"])[0])
    return baseline


def run_attempt(
    output: Path,
    attempt: Attempt,
    source: Path,
    binding: Mapping[str, object],
    generate: Callable[[Path, SharedRasterRendererAdapter], object],
    mujoco: Any,
    observer: Observer = observe_shared_context,
    assess: Callable[[Path, object], Mapping[str, object]] | None = None,
) -> Path:
    source = validate_source_linkage(source)
    lock = AttemptLock(output)
    lock.path = output / "shared-raster-attempt.lock"
    lock.acquire()
    terminal: Path | None = None
    try:
        records = validate_ledger(output)
        ledger_binding = records[0].get("binding")
        if not isinstance(ledger_binding, Mapping) or dict(binding) != dict(ledger_binding):
            raise SharedRasterFailure("attempt binding differs from initialized ledger")
        baseline = validate_completed_prefix(output, records, ledger_binding)
        append_revision(output, "reserved", attempt)
        dataset, pairs = output / "datasets" / attempt.name, output / "paired" / attempt.name
        receipt_path = output / "shared-raster-receipts" / f"{attempt.name}.json"
        receipt: dict[str, object] = {
            "schema": SCHEMA,
            "attempt": attempt.__dict__ | {"name": attempt.name},
            "status": "reserved",
            "plan": records[0]["plan"],
            "source": dict(ledger_binding),
            "invocation": {"argv": list(sys.argv), "cwd": os.getcwd(), "host": platform.node()},
        }
        adapter: SharedRasterRendererAdapter | None = None
        try:
            require_osmesa_linux()
            validate_bindings(source, ledger_binding)
            receipt["runtime"] = bind_pristine_sdk(mujoco)
            dataset.mkdir(parents=True, exist_ok=False)
            pairs.parent.mkdir(parents=True, exist_ok=True)
            with SharedRasterRendererAdapter(mujoco, observer, attempt, pairs, baseline) as active:
                adapter = active
                manifest = generate(dataset, active)
            adapter.validate_complete()
            validate_child_source_provenance(manifest, ledger_binding)
            receipt.update(
                {
                    "dataset_manifest": manifest.model_dump(mode="json")
                    if hasattr(manifest, "model_dump")
                    else manifest,
                    "contexts": adapter.contexts,
                    "native_events": adapter.native_events,
                    "constructor_restored": adapter.restored,
                    "cleanup_complete": adapter.cleanup_complete,
                    "constructor_failures": adapter.constructor_failures,
                    "dataset_artifacts": dataset_artifact_manifest(dataset),
                    "paired_artifacts": dataset_artifact_manifest(pairs),
                    "paired_validation": validate_pair_tree(pairs, dataset),
                    "paired_permissions": paired_permission_probes(pairs),
                    "permissions": canonical_permission_probes(dataset),
                    "assessment": dict(assess(dataset, manifest)) if assess else {},
                    "status": STATUS,
                }
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            publish_bytes(receipt_path, canonical(receipt))
        except BaseException as exc:
            if adapter is not None:
                receipt.update(
                    {
                        "contexts": adapter.contexts,
                        "native_events": adapter.native_events,
                        "constructor_restored": adapter.restored,
                        "cleanup_complete": adapter.cleanup_complete,
                        "constructor_failures": adapter.constructor_failures,
                    }
                )
            receipt["status"] = "failed"
            receipt["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            publish_bytes(receipt_path, canonical(receipt))
            terminal = append_revision(
                output,
                "failed",
                attempt,
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "partial_receipt_sha256": digest_file(receipt_path),
                },
            )
            raise
        terminal = append_revision(
            output,
            "complete",
            attempt,
            {"receipt_sha256": digest_file(receipt_path), "status": STATUS},
        )
        lock.release_after_success(terminal)
        return terminal
    except BaseException:
        raise
