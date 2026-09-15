"""Finite, failure-preserving renderer readback investigation.

The graphics hook is scoped to one already-required ``Renderer.render`` call.  It
does not construct a context, render, read pixels, or alter framebuffer state.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import threading
import traceback
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from epsbench.diagnostics.revision_capture import (
    ArtifactWriter,
    AttemptLock,
    RevisionCaptureFailure,
    StudyCell,
    canonical_json_bytes,
    detect_study_runtime,
    publish_bytes,
    sha256_file,
    translate_study_root,
)
from epsbench.diagnostics.revision_runner import (
    PreparedEpisode,
    RendererStack,
    StackFactory,
    StageFailure,
    capture_cell,
)

SCHEMA = "renderer_readback_cell/v1"
LEDGER_SCHEMA = "renderer_readback_ledger/v1"
ARCHIVE_SHA256 = "0be040f4f37ea8d49ac76c0d2a35baf8cd49b1f1d2dde6216b8582af0685962f"
ARCHIVE_BYTES = 12_012_790
MANIFEST_SHA256 = "c906420ee9a01cc80c02d2ad9900f7bf3233253de9d3322c324a2e3501d76a80"
SDK_RENDERER_SHA256 = "c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548"
SDK_VERSION = "3.12.0"
STUDY_HOST = "DESKTOP-TPUQMNG"

_ACTIVE_HOOK = threading.Lock()
_PRISTINE_ENTRYPOINTS: dict[int, tuple[object, object]] = {}


def bind_sdk_entrypoints(mujoco: Any, *, require_native: bool = True) -> None:
    """Bind pristine SDK callables before any scoped hook is installed."""
    render = mujoco.mjr_render
    read = mujoco.mjr_readPixels
    if require_native and (
        getattr(render, "__module__", None) != "mujoco._render"
        or getattr(read, "__module__", None) != "mujoco._render"
    ):
        raise RevisionCaptureFailure("SDK render entrypoints are not pristine native bindings")
    key = id(mujoco)
    bound = _PRISTINE_ENTRYPOINTS.get(key)
    if bound is not None and (bound[0] is not render or bound[1] is not read):
        raise RevisionCaptureFailure("SDK render entrypoints changed after pristine binding")
    _PRISTINE_ENTRYPOINTS[key] = (render, read)


# Fixed OpenGL 4.5 enum values.  There is deliberately no capability fallback.
GL_ENUMS = {
    "multisample": 0x809D,
    "sample_buffers": 0x80A8,
    "samples": 0x80A9,
    "read_framebuffer_binding": 0x8CAA,
    "draw_framebuffer_binding": 0x8CA6,
    "read_buffer": 0x0C02,
    "draw_buffer": 0x0C01,
    "clip_depth_mode": 0x935D,
    "clip_origin": 0x935C,
    "depth_range": 0x0B70,
    "depth_func": 0x0B74,
    "viewport": 0x0BA2,
    "subpixel_bits": 0x0D50,
    "projection_matrix": 0x0BA7,
    "modelview_matrix": 0x0BA6,
    "sample_position": 0x8E50,
}


def fixed_cells() -> tuple[StudyCell, ...]:
    """The eight cells, in their only permitted order."""
    specification = (
        ("corridor", 1),
        ("single_occluder", 0),
    )
    cells: list[StudyCell] = []
    for family, episode in specification:
        for backend, policy in (
            ("wgl", "joint4"),
            ("osmesa", "joint4"),
            ("wgl", "joint0"),
            ("osmesa", "joint0"),
        ):
            cells.append(StudyCell(len(cells), family, episode, backend, policy))
    return tuple(cells)


_REFERENCE_ORDINAL = {
    ("corridor", 1, "wgl", "joint4"): 6,
    ("corridor", 1, "osmesa", "joint4"): 7,
    ("corridor", 1, "wgl", "joint0"): 10,
    ("corridor", 1, "osmesa", "joint0"): 11,
    ("single_occluder", 0, "wgl", "joint4"): 24,
    ("single_occluder", 0, "osmesa", "joint4"): 25,
    ("single_occluder", 0, "wgl", "joint0"): 28,
    ("single_occluder", 0, "osmesa", "joint0"): 29,
}


def _safe_member(name: str) -> None:
    path = Path(name.replace("\\", "/"))
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise RevisionCaptureFailure(f"unsafe archive member path: {name!r}")


def verify_reference_archive(path: Path) -> dict[str, str]:
    """Bind the complete immutable PR26 archive and every manifest entry."""
    if not path.is_file() or path.stat().st_size != ARCHIVE_BYTES:
        raise RevisionCaptureFailure("fixed reference archive size differs")
    if sha256_file(path) != ARCHIVE_SHA256:
        raise RevisionCaptureFailure("fixed reference archive SHA-256 differs")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise RevisionCaptureFailure("duplicate reference archive member")
            for name in names:
                _safe_member(name)
            raw_manifest = archive.read("evidence-manifest.json")
            if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA256:
                raise RevisionCaptureFailure("fixed reference manifest SHA-256 differs")
            manifest = json.loads(raw_manifest)
            entries = manifest.get("entries")
            if not isinstance(entries, list):
                raise RevisionCaptureFailure("reference manifest entries are absent")
            declared: dict[str, str] = {}
            for item in entries:
                if not isinstance(item, dict):
                    raise RevisionCaptureFailure("invalid reference manifest entry")
                raw_name = item.get("path")
                digest = item.get("sha256")
                size = item.get("bytes")
                if (
                    not isinstance(raw_name, str)
                    or not isinstance(digest, str)
                    or type(size) is not int
                ):
                    raise RevisionCaptureFailure("invalid reference manifest fields")
                name = raw_name
                _safe_member(name)
                if name in declared:
                    raise RevisionCaptureFailure("duplicate reference manifest path")
                payload = archive.read(name)
                if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
                    raise RevisionCaptureFailure(f"reference manifest member differs: {name}")
                declared[name] = digest
            if set(names) != set(declared) | {"evidence-manifest.json"}:
                raise RevisionCaptureFailure("reference archive membership differs")
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise RevisionCaptureFailure("cannot verify fixed reference archive") from exc
    return declared


def _reference_cell_name(cell: StudyCell) -> str:
    key = (cell.family, cell.episode_index, cell.backend, cell.policy)
    ordinal = _REFERENCE_ORDINAL[key]
    return (
        f"{ordinal:02d}-{cell.family}-episode-{cell.episode_index:06d}-{cell.backend}-{cell.policy}"
    )


def exact_reference_comparison(
    archive_path: Path, cell: StudyCell, cell_directory: Path
) -> list[dict[str, object]]:
    """Compare every selected PR26 output exactly; a mismatch is terminal."""
    verify_reference_archive(archive_path)
    old_root = f"study/cells/{_reference_cell_name(cell)}"
    results: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for pose in ("before", "after"):
            for suffix in (
                "rgb",
                "depth",
                "encoded-rgb",
                "decoded-pairs",
                "raw-geom-ids",
            ):
                current = np.load(cell_directory / f"{pose}-{suffix}.npy", allow_pickle=False)
                expected = np.load(
                    io.BytesIO(archive.read(f"{old_root}/{pose}-{suffix}.npy")),
                    allow_pickle=False,
                )
                equal = (
                    current.dtype == expected.dtype
                    and current.shape == expected.shape
                    and np.array_equal(current, expected)
                )
                results.append({"pose": pose, "artifact": suffix, "exact": equal})
                if not equal:
                    raise RevisionCaptureFailure(
                        f"instrumented {pose} {suffix} differs from fixed PR26 output"
                    )
            current_map = json.loads((cell_directory / f"{pose}-segid-map.json").read_text())
            expected_map = json.loads(archive.read(f"{old_root}/{pose}-segid-map.json"))
            equal_map = current_map == expected_map
            results.append({"pose": pose, "artifact": "segid-map", "exact": equal_map})
            if not equal_map:
                raise RevisionCaptureFailure(
                    f"instrumented {pose} segid-map differs from fixed PR26 output"
                )
    verify_reference_archive(archive_path)
    return results


def _safe_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(child) for child in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {
            "renderer_readback_nonfinite": "nan"
            if np.isnan(value)
            else ("+inf" if value > 0 else "-inf")
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return {"renderer_readback_unsupported_type": type_name, "repr": repr(value)}


class GLReader(Protocol):
    def glIsEnabled(self, enum: int) -> object: ...
    def glGetIntegerv(self, enum: int) -> object: ...
    def glGetDoublev(self, enum: int) -> object: ...
    def glGetFloatv(self, enum: int) -> object: ...
    def glGetMultisamplefv(self, enum: int, index: int) -> object: ...
    def glGetError(self) -> object: ...


def _flat(value: object, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    return np.asarray(value, dtype=dtype).reshape(-1).copy()


def inverse_reverse_z(
    raw_depth: npt.NDArray[np.float32], near: float, far: float
) -> npt.NDArray[np.float32]:
    """Reproduce the bound SDK's float32-coefficient/float64 inverse projection."""
    zfar = np.float32(far)
    znear = np.float32(near)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef
    return (d_coef / (np.asarray(raw_depth).astype(np.float64) + c_coef)).astype(np.float32)


def _matrix_record(
    writer: ArtifactWriter, prefix: str, name: str, gl: GLReader, enum: int
) -> tuple[dict[str, object], bool]:
    f32 = _flat(gl.glGetFloatv(enum), np.float32)
    f64 = _flat(gl.glGetDoublev(enum), np.float64)
    f32_ref = writer.array(f"{prefix}-{name}-gl-float.npy", f32, allow_nonfinite=True)
    f64_ref = writer.array(f"{prefix}-{name}-gl-double.npy", f64, allow_nonfinite=True)
    valid = (
        f32.shape == (16,)
        and f64.shape == (16,)
        and np.isfinite(f32).all()
        and np.isfinite(f64).all()
        and np.array_equal(f32, f64.astype(np.float32))
    )
    math_matrix = f64.reshape((4, 4), order="F") if f64.size == 16 else f64
    math_ref = writer.array(
        f"{prefix}-{name}-math-row-column.npy", math_matrix, allow_nonfinite=True
    )
    return (
        {
            "layout": "GL column-major flat16",
            "gl_float": f32_ref,
            "gl_double": f64_ref,
            "math_conversion": "reshape 4x4 with order='F'; rows then columns",
            "math_row_column": math_ref,
            "finite_and_float_cast_consistent": valid,
        },
        valid,
    )


def observe_gl_state(
    gl: GLReader,
    writer: ArtifactWriter,
    prefix: str,
    renderer: object,
    role: str,
    thread_id: int,
) -> tuple[dict[str, object], list[str]]:
    """Read the fixed state set after mjr_render, without changing bindings."""
    failures: list[str] = []
    state: dict[str, object] = {
        "stage": "post_render",
        "stage_limit": "reported state after mjr_render returned; not draw-time state",
        "role": role,
        "thread_id": thread_id,
        "renderer_identity": id(renderer),
        "gl_errors_before_queries": [],
        "queries": {},
    }
    errors_before: list[int] = []
    try:
        first = int(cast(Any, gl.glGetError()))
        if first:
            errors_before.append(first)
    except BaseException as exc:
        failures.append(f"glGetError before queries: {type(exc).__name__}: {exc}")
    state["gl_errors_before_queries"] = errors_before
    queries = cast(dict[str, object], state["queries"])
    integer_queries = (
        "sample_buffers",
        "samples",
        "read_framebuffer_binding",
        "draw_framebuffer_binding",
        "read_buffer",
        "draw_buffer",
        "clip_depth_mode",
        "clip_origin",
        "depth_func",
        "viewport",
        "subpixel_bits",
    )
    try:
        queries["multisample"] = bool(gl.glIsEnabled(GL_ENUMS["multisample"]))
    except BaseException as exc:
        failures.append(f"multisample: {type(exc).__name__}: {exc}")
    for name in integer_queries:
        try:
            value = _flat(gl.glGetIntegerv(GL_ENUMS[name]), np.int64)
            queries[name] = value.tolist()
        except BaseException as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        queries["depth_range"] = _flat(
            gl.glGetDoublev(GL_ENUMS["depth_range"]), np.float64
        ).tolist()
    except BaseException as exc:
        failures.append(f"depth_range: {type(exc).__name__}: {exc}")
    for name in ("projection_matrix", "modelview_matrix"):
        try:
            record, valid = _matrix_record(writer, prefix, name, gl, GL_ENUMS[name])
            queries[name] = record
            if not valid:
                failures.append(f"{name} is nonfinite, malformed, or float-cast inconsistent")
        except BaseException as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    samples = queries.get("samples")
    if samples == [4]:
        positions: list[list[float]] = []
        try:
            for index in range(4):
                positions.append(
                    _flat(
                        gl.glGetMultisamplefv(GL_ENUMS["sample_position"], index), np.float32
                    ).tolist()
                )
            queries["sample_positions"] = {
                "values": positions,
                "interpretation": "shading sample positions; not exhaustive coverage proof",
            }
        except BaseException as exc:
            failures.append(f"sample_positions: {type(exc).__name__}: {exc}")
    context = getattr(renderer, "_mjr_context", None)
    scene = getattr(renderer, "scene", getattr(renderer, "_scene", None))
    state["read_depth_map"] = _safe_json(getattr(context, "readDepthMap", None))
    if scene is None:
        failures.append("renderer scene unavailable")
    else:
        state["scene"] = _safe_json(
            {
                "flags": np.asarray(scene.flags).copy(),
                "camera": [
                    {
                        **{
                            name: np.asarray(getattr(camera, name)).copy()
                            for name in ("pos", "forward", "up")
                        },
                        "frustum": {
                            name: getattr(camera, name)
                            for name in (
                                "frustum_near",
                                "frustum_far",
                                "frustum_top",
                                "frustum_bottom",
                                "frustum_center",
                                "frustum_width",
                            )
                        },
                    }
                    for camera in scene.camera
                ],
            }
        )
    errors_after: list[int] = []
    try:
        error = int(cast(Any, gl.glGetError()))
        if error:
            errors_after.append(error)
    except BaseException as exc:
        failures.append(f"glGetError after queries: {type(exc).__name__}: {exc}")
    state["gl_errors_after_queries"] = errors_after
    queries["sample_query_scope"] = (
        "GL_SAMPLE_BUFFERS and GL_SAMPLES describe the bound draw framebuffer at "
        "this post-render query stage"
    )
    if errors_before or errors_after:
        failures.append("OpenGL error observed")
    state["query_failures"] = failures
    return state, failures


class InstrumentedStack:
    """Delegate stack whose only intervention scopes two SDK function wrappers."""

    def __init__(
        self,
        base: RendererStack,
        writer: ArtifactWriter,
        gl: GLReader,
        expected_depth_map: int,
    ):
        self.base = base
        self.role = base.role
        self.width = base.width
        self.height = base.height
        self.writer = writer
        self.gl = gl
        self.expected_depth_map = expected_depth_map
        self.observations: list[dict[str, object]] = []
        self.pose_name = "unset"
        self._render_index = 0

    def make_current(self) -> None:
        self.base.make_current()

    def set_pose(self, pose_name: str) -> Mapping[str, object]:
        self.pose_name = pose_name
        return self.base.set_pose(pose_name)

    def update_scene(self) -> None:
        self.base.update_scene()

    def _render(self, modality: str, operation: Callable[[], npt.NDArray[Any]]) -> npt.NDArray[Any]:
        renderer = getattr(self.base, "renderer", None)
        mujoco = getattr(self.base, "mujoco", None)
        if renderer is None or mujoco is None:
            raise RevisionCaptureFailure("instrumented stack lacks SDK renderer/module")
        pristine = _PRISTINE_ENTRYPOINTS.get(id(mujoco))
        if pristine is None:
            raise RevisionCaptureFailure("SDK entrypoints were not bound before instrumentation")
        original_render, original_read = pristine
        if mujoco.mjr_render is not original_render or mujoco.mjr_readPixels is not original_read:
            raise RevisionCaptureFailure("SDK functions differ from pristine bindings")
        if not _ACTIVE_HOOK.acquire(blocking=False):
            raise RevisionCaptureFailure("another renderer hook is active in this process")
        thread_id = threading.get_ident()
        prefix = f"readback-{self.pose_name}-{modality}"
        event: dict[str, object] = {
            "sequence": self._render_index,
            "pose": self.pose_name,
            "modality": modality,
            "role": self.role,
            "thread_id": thread_id,
            "renderer_identity": id(renderer),
            "render_calls": 0,
            "readback_calls": 0,
        }
        self._render_index += 1
        failures: list[str] = []
        operation_error: BaseException | None = None
        result: npt.NDArray[Any] | None = None

        def reject_wrong_call(kind: str, count: int, rect: object, context: object) -> None:
            if count != 0:
                raise RevisionCaptureFailure(f"duplicate {kind} callback rejected")
            if threading.get_ident() != thread_id:
                raise RevisionCaptureFailure(f"{kind} callback thread differs")
            if rect is not renderer._rect or context is not renderer._mjr_context:
                raise RevisionCaptureFailure(f"{kind} context/viewport identity differs")

        def render_hook(rect: object, scene: object, context: object) -> object:
            count = int(cast(Any, event["render_calls"]))
            reject_wrong_call("mjr_render", count, rect, context)
            if scene is not renderer._scene:
                raise RevisionCaptureFailure("mjr_render scene identity differs")
            event["render_calls"] = count + 1
            value = cast(Callable[..., object], original_render)(rect, scene, context)
            state, state_failures = observe_gl_state(
                self.gl, self.writer, prefix, renderer, self.role, thread_id
            )
            event["post_render_state"] = state
            failures.extend(state_failures)
            queries = cast(Mapping[str, object], state.get("queries", {}))
            if queries.get("viewport") != [0, 0, self.width, self.height]:
                failures.append("post-render viewport differs from renderer dimensions")
            expected_samples = int(getattr(context, "offSamples", -1))
            if queries.get("samples") != [expected_samples]:
                failures.append("post-render draw-framebuffer samples differ from context")
            if int(getattr(context, "readDepthMap", -1)) != self.expected_depth_map:
                failures.append("readDepthMap differs from mjDEPTH_ZEROFAR")
            return value

        def retain_read_argument(argument_value: object, name: str, complete: bool) -> None:
            if argument_value is None:
                return
            argument = np.asarray(argument_value)
            before_copy = hashlib.sha256(argument.tobytes(order="C")).hexdigest()
            retained = argument.copy()
            after_copy = hashlib.sha256(argument.tobytes(order="C")).hexdigest()
            copy_hash = hashlib.sha256(retained.tobytes(order="C")).hexdigest()
            identical = before_copy == after_copy == copy_hash
            reference = self.writer.array(
                f"{prefix}-raw-{name}.npy",
                retained,
                complete=complete,
                validated=False,
                allow_nonfinite=True,
            )
            event[f"raw_{name}"] = reference
            event[f"raw_{name}_copy_identity"] = {
                "intercepted_argument_sha256_before_copy": before_copy,
                "intercepted_argument_sha256_after_copy": after_copy,
                "retained_copy_sha256": copy_hash,
                "bitwise_identical_and_argument_unmodified": identical,
                "timing": "after original mjr_readPixels; before SDK postprocessing",
            }
            if not identical:
                failures.append(f"raw {name} copy is not bitwise identical")
            expected_shape = (
                (self.height, self.width)
                if name == "depth_window"
                else (self.height, self.width, 3)
            )
            expected_dtype = np.float32 if name == "depth_window" else np.uint8
            if retained.dtype != expected_dtype or retained.shape != expected_shape:
                failures.append(f"raw {name} readback schema differs")
            elif np.issubdtype(retained.dtype, np.floating) and not np.isfinite(retained).all():
                failures.append(f"raw {name} readback is nonfinite")
            elif complete:
                reference["validated"] = True

        def read_hook(rgb: object, depth: object, rect: object, context: object) -> object:
            count = int(cast(Any, event["readback_calls"]))
            reject_wrong_call("mjr_readPixels", count, rect, context)
            event["readback_calls"] = count + 1
            native_error: BaseException | None = None
            native_traceback: object | None = None
            value: object = None
            try:
                value = cast(Callable[..., object], original_read)(rgb, depth, rect, context)
            except BaseException as exc:
                native_error = exc
                native_traceback = exc.__traceback__
            try:
                retain_read_argument(rgb, "color", native_error is None)
                retain_read_argument(depth, "depth_window", native_error is None)
            except BaseException as exc:
                event["readback_retention_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                if native_error is None:
                    raise
            if native_error is not None:
                raise native_error.with_traceback(cast(Any, native_traceback))
            return value

        try:
            mujoco.mjr_render = render_hook
            mujoco.mjr_readPixels = read_hook
            result = operation()
        except BaseException as exc:
            operation_error = exc
            event["operation_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
        finally:
            mujoco.mjr_render = original_render
            mujoco.mjr_readPixels = original_read
            restored = (
                mujoco.mjr_render is original_render and mujoco.mjr_readPixels is original_read
            )
            event["originals_restored"] = restored
            if event["render_calls"] != 1 or event["readback_calls"] != 1:
                failures.append("expected exactly one SDK render and one SDK readback")
            if not restored:
                failures.append("SDK functions were not restored")
            event["validation_failures"] = failures
            self.observations.append(event)
            try:
                publish_bytes(
                    self.writer.root / f"{prefix}-observation.json",
                    canonical_json_bytes(_safe_json(event)),
                )
            except BaseException as exc:
                if operation_error is None:
                    operation_error = exc
            finally:
                _ACTIVE_HOOK.release()
        if operation_error is not None:
            raise operation_error
        if failures:
            raise RevisionCaptureFailure("; ".join(failures))
        if result is None:
            raise RevisionCaptureFailure("SDK renderer returned no array")
        return np.asarray(result)

    def render_rgb(self) -> npt.NDArray[np.uint8]:
        return cast(npt.NDArray[np.uint8], self._render("rgb", self.base.render_rgb))

    def enable_depth(self) -> None:
        self.base.enable_depth()

    def render_depth(self) -> npt.NDArray[np.float32]:
        return cast(npt.NDArray[np.float32], self._render("depth", self.base.render_depth))

    def disable_depth(self) -> None:
        self.base.disable_depth()

    def enable_segmentation(self) -> None:
        self.base.enable_segmentation()

    def scene_map(self) -> Mapping[int, tuple[int, int]]:
        return self.base.scene_map()

    def render_segmentation(self, out: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
        return cast(
            npt.NDArray[np.int32],
            self._render("segmentation", lambda: self.base.render_segmentation(out)),
        )

    def disable_segmentation(self) -> None:
        self.base.disable_segmentation()

    def provenance(self) -> Mapping[str, object]:
        return self.base.provenance()

    def close(self) -> None:
        self.base.close()


def capture_instrumented_cell(
    cell: StudyCell,
    prepared: PreparedEpisode,
    factory: StackFactory,
    cell_directory: Path,
    reference_archive: Path,
    *,
    gl: GLReader,
    expected_depth_map: int,
    geom_objtype: int,
) -> dict[str, Any]:
    """Reuse the accepted capture path and add a distinct immutable receipt."""
    writer = ArtifactWriter(cell_directory)
    stacks: list[InstrumentedStack] = []

    def wrapped_factory(samples: int, role: str) -> InstrumentedStack:
        stack = InstrumentedStack(factory(samples, role), writer, gl, expected_depth_map)
        stacks.append(stack)
        return stack

    comparisons: list[dict[str, object]] = []

    def final_validate(_receipt: dict[str, Any], directory: Path) -> None:
        comparisons.extend(exact_reference_comparison(reference_archive, cell, directory))

    base_receipt: dict[str, Any]
    terminal: BaseException | None = None
    try:
        base_receipt = capture_cell(
            cell,
            prepared,
            wrapped_factory,
            cell_directory,
            geom_objtype=geom_objtype,
            final_validate=final_validate,
        )
    except StageFailure as exc:
        base_receipt = exc.receipt
        terminal = exc
    receipt = {
        "schema": SCHEMA,
        "identity": base_receipt.get("identity"),
        "source": base_receipt.get("source"),
        "base_capture_receipt": "receipt.json",
        "renderer_calls": [event for stack in stacks for event in stack.observations],
        "reference_comparisons": comparisons,
        "state": "failed" if terminal else "complete",
        "failure": _safe_json(base_receipt.get("failure")) if terminal else None,
    }
    publish_bytes(
        cell_directory / "renderer-readback-receipt.json",
        canonical_json_bytes(_safe_json(receipt)),
    )
    if terminal is not None:
        raise terminal
    expected_modalities = [
        (pose, modality)
        for pose in ("before", "after")
        for modality in ("rgb", "depth", "segmentation")
    ]
    renderer_calls = cast(list[Mapping[str, object]], receipt["renderer_calls"])
    observed = [(event["pose"], event["modality"]) for event in renderer_calls]
    if observed != expected_modalities:
        raise RevisionCaptureFailure("fixed six-call modality order differs")
    return cast(dict[str, Any], receipt)


def _revision_path(root: Path, index: int) -> Path:
    return root / "ledger" / f"revision-{index:04d}.json"


def _validate_plan(plan: object) -> Mapping[str, object]:
    if not isinstance(plan, Mapping):
        raise RevisionCaptureFailure("renderer-readback plan binding is absent")
    exact = {
        "review_profile": "DUAL_REVIEW",
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "phase_gate_effect": "NONE",
        "archive_sha256": ARCHIVE_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "sdk_version": SDK_VERSION,
        "sdk_renderer_sha256": SDK_RENDERER_SHA256,
        "root_seed": 1729,
        "render": {"width": 160, "height": 120},
        "cells": [cell.name for cell in fixed_cells()],
        "maximum_renderer_contexts": 8,
        "maximum_pose_endpoints": 16,
        "maximum_modality_render_calls": 48,
        "maximum_sdk_readbacks": 48,
        "host": STUDY_HOST,
        "runtimes": ["windows", "wsl"],
    }
    for key, value in exact.items():
        if plan.get(key) != value:
            raise RevisionCaptureFailure(f"fixed renderer-readback plan differs: {key}")
    for key in ("source_head", "source_tree", "dependency_lock_sha256"):
        value = plan.get(key)
        if not isinstance(value, str) or len(value) not in (40, 64):
            raise RevisionCaptureFailure(f"invalid renderer-readback plan digest: {key}")
    configs = plan.get("config_sha256")
    if (
        not isinstance(configs, Mapping)
        or set(configs) != {"corridor", "single_occluder"}
        or any(not isinstance(value, str) or len(value) != 64 for value in configs.values())
    ):
        raise RevisionCaptureFailure("invalid renderer-readback config bindings")
    return plan


def validate_ledger(root: Path, *, allow_stopped_tail: bool = False) -> list[dict[str, Any]]:
    directory = root / "ledger"
    if not directory.is_dir():
        raise RevisionCaptureFailure("renderer-readback ledger absent")
    entries = list(directory.iterdir())
    paths = sorted(directory.glob("revision-*.json"))
    if set(entries) != set(paths) or paths != [_revision_path(root, i) for i in range(len(paths))]:
        raise RevisionCaptureFailure("renderer-readback ledger has foreign entry or gap")
    records: list[dict[str, Any]] = []
    predecessor: str | None = None
    for index, path in enumerate(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema") != LEDGER_SCHEMA
            or record.get("revision") != index
            or record.get("predecessor_sha256") != predecessor
        ):
            raise RevisionCaptureFailure("invalid renderer-readback ledger chain")
        predecessor = sha256_file(path)
        records.append(record)
    expected = [cell.__dict__ | {"name": cell.name} for cell in fixed_cells()]
    if (
        not records
        or records[0].get("event") != "initialised"
        or records[0].get("cells") != expected
    ):
        raise RevisionCaptureFailure("initial renderer-readback plan differs")
    _validate_plan(records[0].get("plan_binding"))
    next_ordinal = 0
    expecting_terminal = False
    for position, record in enumerate(records[1:], start=1):
        if not expecting_terminal:
            if (
                record.get("event") != "reserved"
                or record.get("cell_ordinal") != next_ordinal
                or record.get("cell_name") != fixed_cells()[next_ordinal].name
            ):
                raise RevisionCaptureFailure("renderer-readback reservation history invalid")
            expecting_terminal = True
        else:
            if (
                record.get("event") not in {"complete", "failed"}
                or record.get("cell_ordinal") != next_ordinal
            ):
                raise RevisionCaptureFailure("renderer-readback terminal history invalid")
            if record["event"] == "failed" and position != len(records) - 1:
                raise RevisionCaptureFailure("failed renderer-readback event is not terminal")
            next_ordinal += 1
            expecting_terminal = False
    if not allow_stopped_tail and records[-1].get("event") in {"reserved", "failed"}:
        raise RevisionCaptureFailure(
            "reserved/failed renderer-readback cell permanently stops probe"
        )
    return records


def initialise_ledger(root: Path, plan: Mapping[str, object]) -> Path:
    if root.exists():
        raise RevisionCaptureFailure("renderer-readback output root already exists")
    root.mkdir(parents=True)
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": 0,
        "predecessor_sha256": None,
        "event": "initialised",
        "plan_binding": dict(plan),
        "cells": [cell.__dict__ | {"name": cell.name} for cell in fixed_cells()],
    }
    publish_bytes(_revision_path(root, 0), canonical_json_bytes(record))
    return _revision_path(root, 0)


def append_revision(
    root: Path, event: str, cell: StudyCell, details: Mapping[str, object] | None = None
) -> Path:
    records = validate_ledger(root, allow_stopped_tail=True)
    tail = records[-1]
    if event == "reserved":
        if tail["event"] in {"reserved", "failed"}:
            raise RevisionCaptureFailure("stopped renderer-readback ledger cannot continue")
        completed = [record["cell_ordinal"] for record in records if record["event"] == "complete"]
        if completed != list(range(cell.ordinal)):
            raise RevisionCaptureFailure("wrong next renderer-readback cell")
    elif event in {"complete", "failed"}:
        if tail.get("event") != "reserved" or tail.get("cell_ordinal") != cell.ordinal:
            raise RevisionCaptureFailure("renderer-readback terminal event lacks reservation")
    else:
        raise RevisionCaptureFailure("invalid renderer-readback ledger event")
    index = len(records)
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": index,
        "predecessor_sha256": sha256_file(_revision_path(root, index - 1)),
        "event": event,
        "cell_ordinal": cell.ordinal,
        "cell_name": cell.name,
        "details": dict(details or {}),
    }
    publish_bytes(_revision_path(root, index), canonical_json_bytes(record))
    return _revision_path(root, index)


def verify_clean_source(plan: Mapping[str, object]) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()

    if git("rev-parse", "HEAD") != plan.get("source_head") or git(
        "rev-parse", "HEAD^{tree}"
    ) != plan.get("source_tree"):
        raise RevisionCaptureFailure("live source HEAD/tree differs from plan")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RevisionCaptureFailure("live source worktree is not clean")
    if sha256_file(Path("uv.lock")) != plan.get("dependency_lock_sha256"):
        raise RevisionCaptureFailure("live dependency lock differs from plan")
    configs = cast(Mapping[str, object], plan["config_sha256"])
    if (
        sha256_file(Path("configs/corridor_v0.yaml")) != configs["corridor"]
        or sha256_file(Path("configs/benchmark_v0.yaml")) != configs["single_occluder"]
    ):
        raise RevisionCaptureFailure("live config hashes differ from plan")
    if __import__("platform").node().upper() != STUDY_HOST:
        raise RevisionCaptureFailure("live host differs from plan")
    if detect_study_runtime() not in cast(list[str], plan["runtimes"]):
        raise RevisionCaptureFailure("live runtime differs from plan")


def validate_sdk(mujoco: Any) -> tuple[Path, int]:
    if str(mujoco.__version__) != SDK_VERSION:
        raise RevisionCaptureFailure("MuJoCo runtime version differs from 3.12.0")
    module = __import__("mujoco.rendering.classic.renderer", fromlist=["Renderer"])
    module_file = module.__file__
    if not isinstance(module_file, str):
        raise RevisionCaptureFailure("MuJoCo classic renderer has no source path")
    path = Path(module_file).resolve()
    if sha256_file(path) != SDK_RENDERER_SHA256:
        raise RevisionCaptureFailure("MuJoCo classic renderer source hash differs")
    bind_sdk_entrypoints(mujoco)
    return path, int(mujoco.mjtDepthMap.mjDEPTH_ZEROFAR)


def verify_handoff_records(root: Path) -> dict[str, object]:
    """Require a distinct two-way Windows/WSL exchange before preparation."""
    directory = root / "renderer-readback-handoff"
    expected = {"windows.json", "wsl.json", "verified-windows.json", "verified-wsl.json"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected:
        raise RevisionCaptureFailure("complete renderer-readback handoff is absent")
    runtime_now = detect_study_runtime()
    host_now = __import__("platform").node()
    if host_now.upper() != STUDY_HOST:
        raise RevisionCaptureFailure("renderer-readback host differs")
    records: dict[str, dict[str, object]] = {}
    verified: dict[str, dict[str, object]] = {}
    for runtime in ("windows", "wsl"):
        records[runtime] = json.loads((directory / f"{runtime}.json").read_text("utf-8"))
        verified[runtime] = json.loads((directory / f"verified-{runtime}.json").read_text("utf-8"))
        if (
            records[runtime].get("schema") != "renderer_readback_handoff/v1"
            or records[runtime].get("runtime") != runtime
        ):
            raise RevisionCaptureFailure("renderer-readback handoff source invalid")
        if (
            verified[runtime].get("schema") != "renderer_readback_handoff_verification/v1"
            or verified[runtime].get("runtime") != runtime
        ):
            raise RevisionCaptureFailure("renderer-readback handoff verification invalid")
        if (
            str(records[runtime].get("host", "")).upper() != STUDY_HOST
            or str(verified[runtime].get("host", "")).upper() != STUDY_HOST
        ):
            raise RevisionCaptureFailure("renderer-readback handoff host invalid")
        if verified[runtime].get("own_token") != records[runtime].get("token") or verified[
            runtime
        ].get("observed_root") != records[runtime].get("observed_root"):
            raise RevisionCaptureFailure("renderer-readback own handoff binding invalid")
    if records["windows"].get("token") == records["wsl"].get("token"):
        raise RevisionCaptureFailure("renderer-readback handoff tokens are not distinct")
    for runtime, peer in (("windows", "wsl"), ("wsl", "windows")):
        if verified[runtime].get("peer_token") != records[peer].get("token"):
            raise RevisionCaptureFailure("renderer-readback peer token invalid")
        translated = translate_study_root(str(records[runtime]["observed_root"]), runtime)
        if (
            translated != records[peer].get("observed_root")
            or verified[runtime].get("expected_peer_root") != translated
        ):
            raise RevisionCaptureFailure("renderer-readback cross-runtime root invalid")
    if str(records[runtime_now].get("observed_root")) != str(root.resolve()):
        raise RevisionCaptureFailure("invocation root differs from handoff")
    return {
        "windows_root": records["windows"]["observed_root"],
        "wsl_root": records["wsl"]["observed_root"],
    }


def run_attempt(
    output_root: Path,
    reference_archive: Path,
    cell: StudyCell,
    prepare: Callable[[], tuple[PreparedEpisode, StackFactory, GLReader, int]],
    *,
    geom_objtype: int,
) -> Path:
    lock = AttemptLock(output_root)
    lock.path = output_root / "renderer-readback-attempt.lock"
    lock.acquire()
    terminal: Path | None = None
    try:
        records = validate_ledger(output_root)
        append_revision(output_root, "reserved", cell)
        try:
            reference = verify_reference_archive(reference_archive)
            verify_handoff_records(output_root)
            plan = cast(Mapping[str, object], records[0]["plan_binding"])
            if (
                plan.get("archive_sha256") != ARCHIVE_SHA256
                or plan.get("manifest_sha256") != MANIFEST_SHA256
            ):
                raise RevisionCaptureFailure("ledger reference archive binding differs")
            verify_clean_source(plan)
            prepared, factory, gl, expected_depth_map = prepare()
            config_hashes = plan.get("config_sha256")
            if not isinstance(config_hashes, Mapping) or prepared.source.get(
                "config_sha256"
            ) != config_hashes.get(cell.family):
                raise RevisionCaptureFailure("prepared config differs from plan")
            if (
                prepared.source.get("head") != plan.get("source_head")
                or prepared.source.get("tree") != plan.get("source_tree")
                or prepared.source.get("dependency_lock_sha256")
                != plan.get("dependency_lock_sha256")
            ):
                raise RevisionCaptureFailure("prepared source differs from plan")
            cell_directory = output_root / "cells" / cell.name
            cell_directory.mkdir(parents=True, exist_ok=False)
            capture_instrumented_cell(
                cell,
                prepared,
                factory,
                cell_directory,
                reference_archive,
                gl=gl,
                expected_depth_map=expected_depth_map,
                geom_objtype=geom_objtype,
            )
            if len(reference) != 1831:
                raise RevisionCaptureFailure("reference manifest entry count differs")
        except BaseException as exc:
            terminal = append_revision(
                output_root,
                "failed",
                cell,
                {"error_type": type(exc).__name__, "error_message": str(exc)},
            )
            raise
        terminal = append_revision(
            output_root,
            "complete",
            cell,
            {"receipt_sha256": sha256_file(cell_directory / "renderer-readback-receipt.json")},
        )
        lock.release_after_success(terminal)
        return terminal
    except BaseException:
        raise
