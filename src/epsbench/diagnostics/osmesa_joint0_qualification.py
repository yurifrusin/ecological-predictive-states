"""Finite OSMesa/joint0 candidate qualification with immutable failure evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from epsbench.diagnostics.revision_capture import (
    AttemptLock,
    canonical_json_bytes,
    publish_bytes,
    sha256_file,
)

STATUS = "PARTIAL_EVIDENCE_ALIGNMENT_UNSPECIFIED"
SCHEMA = "osmesa_joint0_candidate/v1"
LEDGER_SCHEMA = "osmesa_joint0_ledger/v1"
ROOT_SEED = 1729
SDK_RENDERER_SHA256 = "c193df6a8b8cc1659819abd0ded5c17dab0e3e64edb7af6a1e8d249bb4a4a548"
SDK_VERSION = "3.12.0"
LOCK_SHA256 = "d8fbbd09590dd2c937db822668d168ed73947d79e3772de7dce1e311b89ebafc"
PROFILES = ("legacy_solid_base_v1", "legacy_solid_alternate_v1")
_HOOK_LOCK = threading.Lock()
_PRISTINE: dict[int, tuple[object, object, object]] = {}


class QualificationFailure(RuntimeError):
    """An integrity failure that permanently stops this qualification."""


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return canonical_json_bytes(value)


def digest_file(path: Path) -> str:
    return sha256_file(path)


@dataclass(frozen=True)
class Attempt:
    ordinal: int
    family: str
    profile: str
    repeat: int

    @property
    def name(self) -> str:
        return f"{self.ordinal:02d}-{self.family}-{self.profile}-repeat-{self.repeat}"

    @property
    def calls_per_episode(self) -> int:
        return 8 if self.family == "single_occluder" else 6

    @property
    def expected_native_calls(self) -> int:
        return self.calls_per_episode * 4


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
        "status_on_completion": STATUS,
        "backend": "osmesa",
        "quality": {"offsamples": 0},
        "runtime": {
            "python": "3.11.15",
            "mujoco": "3.12.0",
            "numpy": "2.4.6",
            "PyOpenGL": "3.1.10",
            "glfw": "2.10.2",
        },
        "seed": ROOT_SEED,
        "episodes_per_attempt": 4,
        "component_topology": False,
        "profiles": list(PROFILES),
        "attempts": [
            {
                **attempt.__dict__,
                "name": attempt.name,
                "expected_contexts": 4,
                "expected_native_render_calls": attempt.expected_native_calls,
                "expected_native_readbacks": attempt.expected_native_calls,
            }
            for attempt in fixed_attempts()
        ],
        "limits": {
            "batch_attempts": 8,
            "contexts": 32,
            "ordinary_pose_endpoints": 64,
            "ordinary_modality_renders": 192,
            "counterfactual_segmentation_renders": 32,
            "native_render_calls": 224,
            "native_readbacks": 224,
        },
        "repeat_exclusion": "dataset-relative run.json only",
        "completion_interpretation": STATUS,
    }


def validate_plan(value: object) -> None:
    if value != plan():
        raise QualificationFailure("immutable qualification plan differs")


def require_osmesa_linux() -> None:
    if platform.system() != "Linux":
        raise QualificationFailure("qualification requires Linux")
    if not (os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME")):
        raise QualificationFailure("qualification requires WSL")
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
        raise QualificationFailure("qualification requires explicit OSMesa environment")


def git_binding(
    root: Path,
    lock: Path,
    configs: Sequence[Path],
    registry: Path,
    seeds: Path | None = None,
) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise QualificationFailure("source worktree not clean")
    return {
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "lock_sha256": digest_file(lock),
        "config_sha256": {path.name: digest_file(path) for path in configs},
        "registry_sha256": digest_file(registry),
        "seed_registry_sha256": digest_file(
            seeds or root / "configs/evaluation_seed_candidates_v0.yaml"
        ),
    }


def validate_bindings(root: Path, value: Mapping[str, object]) -> None:
    current = git_binding(
        root,
        root / "uv.lock",
        (root / "configs/benchmark_v0.yaml", root / "configs/corridor_v0.yaml"),
        root / "configs/appearance_candidates_v0.yaml",
        root / "configs/evaluation_seed_candidates_v0.yaml",
    )
    if current != dict(value):
        raise QualificationFailure("source/config/registry binding changed")
    if current["lock_sha256"] != LOCK_SHA256:
        raise QualificationFailure("dependency lock differs from fixed qualification lock")


def verify_pristine_entrypoints(mujoco: Any, classic_module: Any, native_module: Any) -> None:
    if mujoco.Renderer is not classic_module.Renderer:
        raise QualificationFailure("public Renderer is not the installed classic SDK class")
    if (
        mujoco.mjr_render is not native_module.mjr_render
        or mujoco.mjr_readPixels is not native_module.mjr_readPixels
    ):
        raise QualificationFailure("public native entrypoints differ from installed extension")


def bind_pristine_sdk(mujoco: Any) -> dict[str, object]:
    module = __import__("mujoco.rendering.classic.renderer", fromlist=["Renderer"])
    native_module = __import__("mujoco._render", fromlist=["mjr_render", "mjr_readPixels"])
    path_value = module.__file__
    if not isinstance(path_value, str):
        raise QualificationFailure("classic renderer source path unavailable")
    path = Path(path_value).resolve()
    constructor = mujoco.Renderer
    render = mujoco.mjr_render
    read = mujoco.mjr_readPixels
    verify_pristine_entrypoints(mujoco, module, native_module)
    if str(mujoco.__version__) != SDK_VERSION or digest_file(path) != SDK_RENDERER_SHA256:
        raise QualificationFailure("MuJoCo version or classic renderer source differs")
    if (
        getattr(render, "__module__", None) != "mujoco._render"
        or getattr(read, "__module__", None) != "mujoco._render"
    ):
        raise QualificationFailure("native SDK entrypoints are not pristine")
    previous = _PRISTINE.get(id(mujoco))
    if previous is not None and previous != (constructor, render, read):
        raise QualificationFailure("SDK entrypoints changed after binding")
    _PRISTINE[id(mujoco)] = (constructor, render, read)
    package = Path(mujoco.__file__).resolve().parent
    binary = package / "libmujoco.so.3.12.0"
    native_path_value = native_module.__file__
    if not binary.is_file() or not isinstance(native_path_value, str):
        raise QualificationFailure("MuJoCo binary or native extension unavailable")
    native_path = Path(native_path_value).resolve()
    if not native_path.is_file():
        raise QualificationFailure("MuJoCo native extension unavailable")
    package_digest = hashlib.sha256()
    for item in sorted(package.glob("*.py")):
        package_digest.update(item.name.encode())
        package_digest.update(item.read_bytes())
    versions: dict[str, object] = {
        "python_version": platform.python_version(),
        "mujoco_version": str(mujoco.__version__),
        "numpy_version": importlib.metadata.version("numpy"),
        "pyopengl_version": importlib.metadata.version("PyOpenGL"),
        "glfw_version": importlib.metadata.version("glfw"),
        "sdk_renderer_path": str(path),
        "sdk_renderer_sha256": digest_file(path),
        "libmujoco_binary_sha256": digest_file(binary),
        "native_extension_path": str(native_path),
        "native_extension_sha256": digest_file(native_path),
        "top_level_python_sources_sha256": package_digest.hexdigest(),
        "host": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
    }
    expected = plan()["runtime"]
    observed = {
        "python": versions["python_version"],
        "mujoco": versions["mujoco_version"],
        "numpy": versions["numpy_version"],
        "PyOpenGL": versions["pyopengl_version"],
        "glfw": versions["glfw_version"],
    }
    if observed != expected:
        raise QualificationFailure("CPU/package runtime differs from fixed plan")
    return versions


class Observer(Protocol):
    def __call__(
        self, renderer: object, model: object, width: int, height: int
    ) -> Mapping[str, object]: ...


class _RendererProxy:
    def __init__(self, owner: RendererAdapter, renderer: Any, context_index: int):
        self._owner = owner
        self._renderer = renderer
        self._context_index = context_index
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
            if self._last_update is not None and self._last_update.get("geomgroup") is not None:
                group = cast(list[int], self._last_update["geomgroup"])
                if len(group) > 1 and group[1] == 0:
                    return "counterfactual_segmentation"
            return "segmentation"
        return "rgb"

    def render(self, *args: object, **kwargs: object) -> object:
        if threading.get_ident() != self._thread or self._rendering:
            raise QualificationFailure("renderer render thread/reentrancy differs")
        if self._last_update is None:
            raise QualificationFailure("render lacks preceding scene update")
        expected = (
            self._owner.expected_modalities[len(self._owner.native_events)]
            if len(self._owner.native_events) < len(self._owner.expected_modalities)
            else None
        )
        modality = self._modality()
        if expected is None or modality != expected:
            raise QualificationFailure("renderer modality sequence differs before native call")
        if self._context_index != len(self._owner.native_events) // self.attempt_calls:
            raise QualificationFailure("renderer context sequence differs before native call")
        mujoco = self._owner.mujoco
        pristine = _PRISTINE.get(id(mujoco))
        if (
            pristine is None
            or mujoco.mjr_render is not pristine[1]
            or mujoco.mjr_readPixels is not pristine[2]
        ):
            raise QualificationFailure("native entrypoints differ before scoped call")
        original_render, original_read = pristine[1], pristine[2]
        event: dict[str, object] = {
            "context_index": self._context_index,
            "sequence": len(self._owner.native_events),
            "modality": modality,
            "thread": self._thread,
            "renderer_identity": id(self._renderer),
            "scene_update": dict(self._last_update),
            "render_calls": 0,
            "readback_calls": 0,
        }
        error: BaseException | None = None
        result: object = None

        def render_hook(rect: object, scene: object, context: object) -> object:
            if event["render_calls"] != 0:
                raise QualificationFailure("duplicate native render rejected before call")
            if (
                threading.get_ident() != self._thread
                or rect is not self._renderer._rect
                or scene is not self._renderer._scene
                or context is not self._renderer._mjr_context
            ):
                raise QualificationFailure("native render identity/thread differs before call")
            event["render_calls"] = 1
            return cast(Callable[..., object], original_render)(rect, scene, context)

        def read_hook(rgb: object, depth: object, rect: object, context: object) -> object:
            if event["readback_calls"] != 0:
                raise QualificationFailure("duplicate native readback rejected before call")
            if (
                threading.get_ident() != self._thread
                or rect is not self._renderer._rect
                or context is not self._renderer._mjr_context
            ):
                raise QualificationFailure("native readback identity/thread differs before call")
            if event["render_calls"] != 1:
                raise QualificationFailure("native readback preceded native render")
            if modality == "depth":
                valid_buffers = rgb is None and isinstance(depth, np.ndarray)
            else:
                valid_buffers = isinstance(rgb, np.ndarray) and depth is None
            if not valid_buffers:
                raise QualificationFailure("native readback RGB/depth argument roles differ")
            event["readback_calls"] = 1
            return cast(Callable[..., object], original_read)(rgb, depth, rect, context)

        self._rendering = True
        try:
            mujoco.mjr_render = render_hook
            mujoco.mjr_readPixels = read_hook
            result = self._renderer.render(*args, **kwargs)
        except BaseException as exc:
            error = exc
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            mujoco.mjr_render = original_render
            mujoco.mjr_readPixels = original_read
            self._rendering = False
            restored = (
                mujoco.mjr_render is original_render and mujoco.mjr_readPixels is original_read
            )
            event["restored"] = restored
            failures = []
            if event["render_calls"] != 1 or event["readback_calls"] != 1:
                failures.append("native call count differs")
            if not restored:
                failures.append("native entrypoint restoration failed")
            event["validation_failures"] = failures
            self._owner.native_events.append(event)
        if error is not None:
            raise error
        if cast(list[object], event["validation_failures"]):
            raise QualificationFailure("native call/restoration validation failed")
        return result

    def close(self) -> None:
        if not self._closed:
            try:
                self._renderer.close()
            except BaseException as exc:
                self._owner.contexts[self._context_index]["close"] = {
                    "complete": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                raise
            else:
                self._closed = True
                self._owner.contexts[self._context_index]["close"] = {"complete": True}

    @property
    def attempt_calls(self) -> int:
        return self._owner.attempt.calls_per_episode


class RendererAdapter:
    """Scope the original constructor across exactly four canonical episode contexts."""

    def __init__(
        self,
        mujoco: Any,
        observe: Observer,
        attempt: Attempt,
        context_baseline: Mapping[str, object] | None = None,
    ):
        self.mujoco = mujoco
        self.observe = observe
        self.attempt = attempt
        self.thread = threading.get_ident()
        self.contexts: list[dict[str, object]] = []
        self.native_events: list[dict[str, object]] = []
        self.proxies: list[_RendererProxy] = []
        self.restored = False
        self._original: object | None = None
        self.constructor_failures: list[dict[str, object]] = []
        self.cleanup_complete = False
        self.context_baseline = dict(context_baseline) if context_baseline is not None else None
        self.expected_modalities = [
            modality
            for _episode in range(4)
            for _pose in ("before", "after")
            for modality in (
                ("rgb", "depth", "segmentation", "counterfactual_segmentation")
                if attempt.family == "single_occluder"
                else ("rgb", "depth", "segmentation")
            )
        ]

    def __enter__(self) -> RendererAdapter:
        pristine = _PRISTINE.get(id(self.mujoco))
        if pristine is None:
            raise QualificationFailure("pristine SDK was not bound")
        if not _HOOK_LOCK.acquire(blocking=False):
            raise QualificationFailure("qualification renderer hook already active")
        self._original = pristine[0]
        if self.mujoco.Renderer is not self._original:
            _HOOK_LOCK.release()
            raise QualificationFailure("pre-existing Renderer replacement")

        def constructor(model: Any, *args: Any, **kwargs: Any) -> _RendererProxy:
            if threading.get_ident() != self.thread:
                raise QualificationFailure("Renderer constructor wrong thread")
            if len(self.contexts) >= 4:
                raise QualificationFailure("fifth Renderer constructor rejected before call")
            model.vis.quality.offsamples = 0
            renderer: Any | None = None
            try:
                renderer = cast(Callable[..., Any], self._original)(model, *args, **kwargs)
                width = int(kwargs.get("width", getattr(renderer, "width", -1)))
                height = int(kwargs.get("height", getattr(renderer, "height", -1)))
                facts = dict(self.observe(renderer, model, width, height))
                binding = context_runtime_binding(facts)
                expected_binding = self.context_baseline
                if expected_binding is None and self.contexts:
                    expected_binding = context_runtime_binding(self.contexts[0])
                if expected_binding is not None and binding != expected_binding:
                    raise QualificationFailure("OSMesa context runtime differs before first render")
                facts.update({"context_index": len(self.contexts), "thread": self.thread})
                self.contexts.append(facts)
                proxy = _RendererProxy(self, renderer, len(self.contexts) - 1)
                self.proxies.append(proxy)
                return proxy
            except BaseException:
                if renderer is not None:
                    try:
                        renderer.close()
                    except BaseException as close_exc:
                        self.constructor_failures.append(
                            {
                                "context_index": len(self.contexts),
                                "close_complete": False,
                                "close_error_type": type(close_exc).__name__,
                                "close_error_message": str(close_exc),
                            }
                        )
                    else:
                        self.constructor_failures.append(
                            {"context_index": len(self.contexts), "close_complete": True}
                        )
                raise

        self.mujoco.Renderer = constructor
        return self

    def __exit__(self, typ: object, value: object, tb: object) -> Literal[False]:
        restore_error: BaseException | None = None
        try:
            self.mujoco.Renderer = self._original
        finally:
            for proxy in reversed(self.proxies):
                if not proxy._closed:
                    try:
                        proxy.close()
                    except BaseException as exc:
                        if restore_error is None:
                            restore_error = exc
            self.restored = self.mujoco.Renderer is self._original
            self.cleanup_complete = all(proxy._closed for proxy in self.proxies)
            if not self.restored and restore_error is None:
                restore_error = QualificationFailure("Renderer constructor restoration failed")
            if not self.cleanup_complete and restore_error is None:
                restore_error = QualificationFailure("renderer cleanup incomplete")
            _HOOK_LOCK.release()
        if value is None and restore_error is not None:
            raise restore_error
        return False

    def validate_complete(self) -> None:
        observed = [str(event["modality"]) for event in self.native_events]
        if len(self.contexts) != 4 or observed != self.expected_modalities:
            raise QualificationFailure("context count or family render sequence differs")
        if (
            not self.restored
            or not self.cleanup_complete
            or any(event.get("validation_failures") for event in self.native_events)
        ):
            raise QualificationFailure("renderer/native restoration validation differs")


def observe_zero_sample_osmesa(
    renderer: object, model: Any, width: int, height: int
) -> Mapping[str, object]:
    from OpenGL import GL  # type: ignore[import-untyped]

    from epsbench.diagnostics.gl_provenance import inspect_mujoco_offscreen_attachments
    from epsbench.diagnostics.mujoco_runner import _observed_backend
    from epsbench.diagnostics.revision_mujoco import _study_depth_attachments

    context = getattr(renderer, "_gl_context", None)
    if context is None or not hasattr(context, "make_current"):
        raise QualificationFailure("renderer has no current-context operation")
    context.make_current()
    before_error = int(GL.glGetError())
    provenance = dict(inspect_mujoco_offscreen_attachments(renderer))
    attachments = cast(dict[str, object], provenance["offscreen_attachments"])
    for name, depth in _study_depth_attachments(renderer).items():
        cast(dict[str, object], attachments[name])["depth"] = depth
    mjr = getattr(renderer, "_mjr_context", None)
    main = cast(Mapping[str, object], attachments.get("offFBO", {}))
    resolve = cast(Mapping[str, object], attachments.get("offFBO_r", {}))

    def scalar(value: object, name: str) -> int:
        array = np.asarray(value)
        if array.size != 1:
            raise QualificationFailure(f"{name} is not scalar")
        return int(array.reshape(-1)[0])

    def storage_dimensions(record: Mapping[str, object], name: str) -> list[int]:
        object_name = record.get("object_name")
        if not isinstance(object_name, int) or object_name <= 0:
            raise QualificationFailure(f"{name} renderbuffer identity unavailable")
        saved = scalar(GL.glGetIntegerv(GL.GL_RENDERBUFFER_BINDING), "GL_RENDERBUFFER_BINDING")
        try:
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, object_name)
            return [
                scalar(
                    GL.glGetRenderbufferParameteriv(GL.GL_RENDERBUFFER, GL.GL_RENDERBUFFER_WIDTH),
                    f"{name} width",
                ),
                scalar(
                    GL.glGetRenderbufferParameteriv(GL.GL_RENDERBUFFER, GL.GL_RENDERBUFFER_HEIGHT),
                    f"{name} height",
                ),
            ]
        finally:
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, saved)

    main_color = cast(Mapping[str, object], main.get("color0", {}))
    main_depth = cast(Mapping[str, object], main.get("depth", {}))
    sample_buffers = scalar(GL.glGetIntegerv(GL.GL_SAMPLE_BUFFERS), "GL_SAMPLE_BUFFERS")
    samples = scalar(GL.glGetIntegerv(GL.GL_SAMPLES), "GL_SAMPLES")
    color_dimensions = storage_dimensions(main_color, "color")
    depth_dimensions = storage_dimensions(main_depth, "depth")
    after_error = int(GL.glGetError())
    facts = {
        **provenance,
        "actual_backend": _observed_backend(renderer),
        "context_module": type(context).__module__,
        "requested_offsamples": 0,
        "actual_offsamples": int(getattr(mjr, "offSamples", -1)),
        "model_offsamples": int(model.vis.quality.offsamples),
        "width": width,
        "height": height,
        "mjr_off_width": int(getattr(mjr, "offWidth", -1)),
        "mjr_off_height": int(getattr(mjr, "offHeight", -1)),
        "color_storage_dimensions": color_dimensions,
        "depth_storage_dimensions": depth_dimensions,
        "sample_buffers": sample_buffers,
        "samples": samples,
        "gl_error_before": before_error,
        "gl_error_after": after_error,
    }
    valid = (
        facts["actual_backend"] == "osmesa"
        and facts["context_module"] == "mujoco.osmesa"
        and facts["actual_offsamples"] == facts["model_offsamples"] == 0
        and (width, height) == (160, 120)
        and (facts["mjr_off_width"], facts["mjr_off_height"]) == (160, 120)
        and color_dimensions == depth_dimensions == [160, 120]
        and facts["sample_buffers"] == facts["samples"] == 0
        and before_error == after_error == 0
        and main.get("present") is True
        and main.get("draw_framebuffer_samples") == 0
        and main_color.get("samples") == 0
        and main_depth.get("samples") == 0
        and resolve.get("present") is False
    )
    if not valid:
        raise QualificationFailure("zero-sample OSMesa context provenance invalid")
    return facts


def context_runtime_binding(facts: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "actual_backend",
        "context_module",
        "requested_offsamples",
        "actual_offsamples",
        "model_offsamples",
        "width",
        "height",
        "sample_buffers",
        "samples",
        "mjr_off_width",
        "mjr_off_height",
        "color_storage_dimensions",
        "depth_storage_dimensions",
        "attachment_format",
        "attachment_component_type",
        "gl_samples",
        "gl_vendor",
        "gl_renderer",
        "gl_version",
    )
    value = {key: facts.get(key) for key in keys}
    if any(item is None for item in value.values()):
        raise QualificationFailure("context runtime binding is incomplete")
    return value


def arrays_equal(left: Path, right: Path) -> None:
    a = np.load(left, allow_pickle=False)
    b = np.load(right, allow_pickle=False)
    if a.shape != b.shape or a.dtype != b.dtype or a.tobytes(order="C") != b.tobytes(order="C"):
        raise QualificationFailure(f"array dtype/shape/content bytes differ: {left.name}")
    if left.read_bytes() != right.read_bytes():
        raise QualificationFailure(f"array file bytes differ: {left.name}")


def deterministic_members(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "run.json"
    )


def compare_repeat_artifacts(left: Path, right: Path) -> list[str]:
    names = deterministic_members(left)
    if names != deterministic_members(right):
        raise QualificationFailure("repeat artifact membership differs")
    for name in names:
        a, b = left / name, right / name
        if a.suffix == ".npy":
            arrays_equal(a, b)
        elif a.read_bytes() != b.read_bytes():
            raise QualificationFailure(f"repeat artifact bytes differ: {name}")
    return names


def artifact_manifest(root: Path) -> list[dict[str, object]]:
    return [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": digest_file(root / name)}
        for name in deterministic_members(root)
    ]


def validate_artifact_binding(root: Path, recorded: object) -> None:
    if recorded != artifact_manifest(root):
        raise QualificationFailure("dataset artifact membership/hash differs from receipt")


def validate_recorded_attempt(
    output_root: Path,
    attempt: Attempt,
    *,
    context_baseline: Mapping[str, object] | None = None,
) -> dict[str, object]:
    dataset = output_root / "datasets" / attempt.name
    receipt_path = output_root / "receipts" / f"{attempt.name}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_plan(receipt.get("plan"))
    if receipt.get("status") != STATUS or receipt.get("attempt") != attempt.__dict__ | {
        "name": attempt.name
    }:
        raise QualificationFailure("attempt receipt identity/status differs")
    validate_artifact_binding(dataset, receipt.get("artifacts"))
    contexts = receipt.get("contexts")
    events = receipt.get("native_events")
    if not isinstance(contexts, list) or len(contexts) != 4 or not isinstance(events, list):
        raise QualificationFailure("receipt context/native event shape differs")
    baseline = (
        dict(context_baseline)
        if context_baseline is not None
        else context_runtime_binding(contexts[0])
    )
    if any(context_runtime_binding(context) != baseline for context in contexts):
        raise QualificationFailure("receipt context runtime binding differs")
    expected = [
        modality
        for _episode in range(4)
        for _pose in ("before", "after")
        for modality in (
            ("rgb", "depth", "segmentation", "counterfactual_segmentation")
            if attempt.family == "single_occluder"
            else ("rgb", "depth", "segmentation")
        )
    ]
    if [event.get("modality") for event in events] != expected:
        raise QualificationFailure("receipt native modality sequence differs")
    for index, event in enumerate(events):
        if (
            event.get("sequence") != index
            or event.get("context_index") != index // attempt.calls_per_episode
            or event.get("render_calls") != 1
            or event.get("readback_calls") != 1
            or event.get("restored") is not True
            or event.get("validation_failures") != []
        ):
            raise QualificationFailure("receipt native call fact differs")
        if event.get("modality") == "counterfactual_segmentation":
            group = event.get("scene_update", {}).get("geomgroup")
            if not isinstance(group, list) or len(group) < 2 or group[1] != 0:
                raise QualificationFailure("counterfactual scene-option evidence differs")
    if (
        receipt.get("constructor_restored") is not True
        or receipt.get("cleanup_complete") is not True
        or receipt.get("constructor_failures") != []
    ):
        raise QualificationFailure("receipt constructor cleanup/restoration differs")
    if receipt.get("permissions") != permission_probes(dataset):
        raise QualificationFailure("permission probes differ from receipt")
    return cast(dict[str, object], receipt)


def permission_probes(root: Path) -> dict[str, object]:
    import epsbench.data.loader as loader_module
    from epsbench.data import DatasetLoader, PermissionDeniedError
    from epsbench.schema import Modality, ModalityPermissionSet

    ecological = DatasetLoader(root, ModalityPermissionSet.ecological_only())
    positive: list[str] = []
    for episode in range(4):
        ecological.read_action(episode)
        ecological.read_segmentation(episode, 0)
        ecological.read_segmentation(episode, 1)
        ecological.read_ecological_transition(episode)
        ecological.read_oriented_boundaries(episode)
        ecological.read_ecological_visibility_events(episode)
        ecological.read_analytic_optical_transport(episode)
    positive.append("ecological_actions_regions_boundaries_events_transport")
    metric = DatasetLoader(root, ModalityPermissionSet(allowed=frozenset({Modality.DEPTH})))
    for episode in range(4):
        metric.read_depth(episode, 0)
        metric.read_depth(episode, 1)
    positive.append("authorized_metric_depth")
    calls: list[tuple[str, Callable[[], object]]] = [
        ("depth", lambda: ecological.read_depth(0, 0)),
        ("camera", lambda: ecological.read_camera_world_transform(0, 0)),
        ("raw_ids", lambda: ecological.read_raw_mujoco_geom_ids(0)),
        ("world_coordinates", lambda: ecological.read_raw_world_coordinates(0)),
        ("sampled_geometry", lambda: ecological.read_sampled_corridor_geometry(0)),
        ("semantic_names", lambda: ecological.read_semantic_surface_names(0)),
        ("appearance_control", lambda: ecological.read_appearance_control(0)),
        ("evaluation_seed_registry", ecological.read_evaluation_seed_registry_snapshot),
        ("generation_manifest", ecological.read_dataset_manifest),
        ("control_scene_family", ecological.read_scene_family),
    ]
    denied: list[str] = []
    sentinel_calls: list[str] = []

    def blocked(*args: object, **kwargs: object) -> object:
        sentinel_calls.append("protected_open_or_decode")
        raise AssertionError("permission probe reached protected I/O")

    original_transition = ecological._transition
    original_instrumentation = ecological._instrumentation
    original_open = loader_module.open_owned_regular_file  # type: ignore[attr-defined]
    ecological._transition = blocked  # type: ignore[assignment]
    ecological._instrumentation = blocked  # type: ignore[assignment]
    loader_module.open_owned_regular_file = blocked  # type: ignore[attr-defined,assignment]
    try:
        for name, operation in calls:
            try:
                operation()
            except PermissionDeniedError:
                denied.append(name)
            else:
                raise QualificationFailure(f"ecological loader exposed privileged {name}")
    finally:
        ecological._transition = original_transition  # type: ignore[method-assign]
        ecological._instrumentation = original_instrumentation  # type: ignore[method-assign]
        loader_module.open_owned_regular_file = original_open  # type: ignore[attr-defined]
    if sentinel_calls:
        raise QualificationFailure("permission denial occurred after protected I/O")
    return {
        "positive": positive,
        "denied_before_open": denied,
        "sentinel_calls": 0,
        "public_projection": "ecological_only",
    }


def public_projection(root: Path) -> dict[str, object]:
    from epsbench.data import DatasetLoader
    from epsbench.schema import ModalityPermissionSet

    loader = DatasetLoader(root, ModalityPermissionSet.ecological_only())

    def array_id(value: np.ndarray) -> dict[str, object]:
        array = np.asarray(value)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": digest_bytes(array.tobytes(order="C")),
        }

    episodes = []
    for index in range(4):
        events = loader.read_ecological_visibility_events(index)
        transport = loader.read_analytic_optical_transport(index)
        episodes.append(
            {
                "action": loader.read_action(index).model_dump(mode="json"),
                "transition": loader.read_ecological_transition(index).model_dump(mode="json"),
                "boundaries": loader.read_oriented_boundaries(index).model_dump(mode="json"),
                "segmentation": [
                    array_id(loader.read_segmentation(index, frame)) for frame in (0, 1)
                ],
                "event_arrays": [
                    array_id(getattr(events, name))
                    for name in (
                        "before_fate_codes",
                        "before_affected_surface_labels",
                        "before_owner_surface_labels",
                        "after_origin_codes",
                        "after_affected_surface_labels",
                        "after_owner_surface_labels",
                    )
                ],
                "transport_arrays": [
                    array_id(getattr(transport, name))
                    for name in (
                        "forward_vectors_fixed",
                        "forward_validity",
                        "forward_reasons",
                        "backward_vectors_fixed",
                        "backward_validity",
                        "backward_reasons",
                    )
                ],
            }
        )
    return {"episodes": episodes}


def compare_profile_outputs(base: Path, alternate: Path) -> dict[str, object]:
    if public_projection(base) != public_projection(alternate):
        raise QualificationFailure("cross-profile ecological/action target projection differs")
    from epsbench.data import DatasetLoader
    from epsbench.schema import ModalityPermissionSet

    permissions = ModalityPermissionSet.all_modalities()
    left, right = DatasetLoader(base, permissions), DatasetLoader(alternate, permissions)
    geometry_equal = True
    rgb_differences = 0
    depth_equal: list[bool] = []
    for episode in range(4):
        left_instrumentation: Any = left._instrumentation(episode)
        right_instrumentation: Any = right._instrumentation(episode)
        semantic_fields = (
            "raw_geom_ids",
            "raw_to_opaque_surface_ids",
            "raw_geom_world_positions",
            "raw_geom_compiled_sizes",
            "raw_geom_types",
            "raw_geom_world_rotations_row_major",
        )
        if any(
            getattr(left_instrumentation, field) != getattr(right_instrumentation, field)
            for field in semantic_fields
        ):
            geometry_equal = False
        if hasattr(left_instrumentation, "sampled_geometry") and (
            left_instrumentation.sampled_geometry != right_instrumentation.sampled_geometry
        ):
            geometry_equal = False
        if hasattr(left_instrumentation, "occlusion_oracle"):
            left_frames = sorted(
                left_instrumentation.occlusion_oracle.frames, key=lambda item: item.frame_index
            )
            right_frames = sorted(
                right_instrumentation.occlusion_oracle.frames, key=lambda item: item.frame_index
            )
            for left_frame, right_frame in zip(left_frames, right_frames, strict=True):
                left_raw = np.asarray(left._load_npy(left_frame.counterfactual_segmentation))
                right_raw = np.asarray(right._load_npy(right_frame.counterfactual_segmentation))
                if (
                    left_raw.dtype != right_raw.dtype
                    or left_raw.shape != right_raw.shape
                    or left_raw.tobytes(order="C") != right_raw.tobytes(order="C")
                ):
                    raise QualificationFailure(
                        "cross-profile counterfactual raw annotation input differs"
                    )
        for frame in (0, 1):
            if left.read_camera_world_transform(
                episode, frame
            ) != right.read_camera_world_transform(episode, frame):
                geometry_equal = False
            rgb_differences += int(
                not np.array_equal(left.read_rgb(episode, frame), right.read_rgb(episode, frame))
            )
            depth_equal.append(
                np.array_equal(left.read_depth(episode, frame), right.read_depth(episode, frame))
            )
    if not geometry_equal:
        raise QualificationFailure("cross-profile semantic geometry/camera differs")
    if rgb_differences == 0:
        raise QualificationFailure("appearance contrast produced no RGB difference")
    return {
        "public_projection_exact": True,
        "semantic_geometry_camera_action_exact": True,
        "opaque_remap_exact": True,
        "counterfactual_raw_annotation_inputs_exact": True,
        "rgb_different_endpoints": rgb_differences,
        "depth_exact_descriptive": depth_equal,
    }


def _compiled_geometry(instrumentation: Any) -> dict[str, object]:
    planes: list[dict[str, object]] = []
    boxes: list[dict[str, object]] = []
    for name, raw_id in sorted(instrumentation.raw_geom_ids.items()):
        item = {
            "name": name,
            "raw_geom_id": int(raw_id),
            "center": list(instrumentation.raw_geom_world_positions[name]),
            "rotation": np.asarray(
                instrumentation.raw_geom_world_rotations_row_major[name], dtype=np.float64
            )
            .reshape(3, 3)
            .tolist(),
            "half_extents": list(instrumentation.raw_geom_compiled_sizes[name]),
        }
        (planes if instrumentation.raw_geom_types[name] == "plane" else boxes).append(item)
    return {"finite_planes": planes, "oriented_boxes": boxes}


def _raw_segmentation(
    loader: Any, instrumentation: Any, episode: int, frame: int
) -> tuple[np.ndarray, str]:
    if hasattr(instrumentation, "raw_segmentation_frames"):
        record = next(
            x.raw_segmentation
            for x in instrumentation.raw_segmentation_frames
            if x.frame_index == frame
        )
        return np.asarray(
            loader._load_npy(record), dtype=np.int32
        ), "retained_privileged_raw_segmentation"
    opaque = np.asarray(loader.read_segmentation(episode, frame), dtype=np.int32)
    transition = loader.read_ecological_transition(episode)
    opaque_by_surface = {
        surface.surface_id: surface.segmentation_label for surface in transition.surfaces
    }
    if set(instrumentation.raw_to_opaque_surface_ids) != {
        str(raw_id) for raw_id in instrumentation.raw_geom_ids.values()
    } or set(instrumentation.raw_to_opaque_surface_ids.values()) != set(opaque_by_surface):
        raise QualificationFailure("raw-to-opaque mapping is incomplete for controlled surfaces")
    raw_by_label = {
        int(opaque_by_surface[surface_id]): int(raw_id)
        for raw_id, surface_id in instrumentation.raw_to_opaque_surface_ids.items()
    }
    if len(raw_by_label) != len(instrumentation.raw_to_opaque_surface_ids):
        raise QualificationFailure("raw-to-opaque mapping is not bijective")
    observed_labels = {int(value) for value in np.unique(opaque) if int(value) != 0}
    if not observed_labels.issubset(raw_by_label):
        raise QualificationFailure("ordinary opaque segmentation contains an unknown label")
    raw = np.full(opaque.shape, -1, dtype=np.int32)
    for label, raw_id in raw_by_label.items():
        raw[opaque == label] = raw_id
    return raw, "reconstructed_from_opaque_segmentation_and_privileged_mapping"


def geometry_diagnostics(root: Path) -> dict[str, object]:
    """Report descriptive centre-ray ID/depth residuals without an alignment threshold."""
    from epsbench.data import DatasetLoader
    from epsbench.data.validate import validate_dataset
    from epsbench.diagnostics.revision_analysis import (
        analytic_boundary_band,
        analytic_geometry_maps,
    )
    from epsbench.schema import ModalityPermissionSet

    validate_dataset(root)
    loader = DatasetLoader(root, ModalityPermissionSet.all_modalities())
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
    height, width = int(config["render"]["height"]), int(config["render"]["width"])
    fovy = float(config["camera"]["field_of_view_degrees"])
    reports: list[dict[str, object]] = []
    for episode in range(4):
        instrumentation: Any = loader._instrumentation(episode)
        geometry = _compiled_geometry(instrumentation)
        for frame in (0, 1):
            camera = loader.read_camera_world_transform(episode, frame)
            analytic = analytic_geometry_maps(
                geometry,
                {
                    "camera_world_position": camera.camera_world_position,
                    "camera_world_rotation_row_major": camera.camera_world_rotation_row_major,
                    "camera_field_of_view_degrees": fovy,
                },
                height,
                width,
            )
            observed_ids, source = _raw_segmentation(loader, instrumentation, episode, frame)
            depth = np.asarray(loader.read_depth(episode, frame), dtype=np.float32)
            expected_ids = np.asarray(analytic["raw_geom_ids"], dtype=np.int32)
            expected_depth = np.asarray(analytic["camera_axis_depth"], dtype=np.float64)
            boundary = analytic_boundary_band(expected_ids) & (expected_ids >= 0)
            masks = {
                "controlled_boundary": boundary,
                "no_hit": expected_ids < 0,
                "controlled_interior": (~boundary) & (expected_ids >= 0),
            }
            strata: dict[str, object] = {}
            for name, mask in masks.items():
                finite = mask & np.isfinite(depth) & np.isfinite(expected_depth)
                residual = np.abs(depth[finite].astype(np.float64) - expected_depth[finite])
                strata[name] = {
                    "pixels": int(np.count_nonzero(mask)),
                    "centre_id_equal": int(np.count_nonzero(mask & (observed_ids == expected_ids))),
                    "finite_depth_pairs": int(residual.size),
                    "depth_abs_residual_min": None
                    if not residual.size
                    else float(np.min(residual)),
                    "depth_abs_residual_max": None
                    if not residual.size
                    else float(np.max(residual)),
                    "depth_abs_residual_mean": None
                    if not residual.size
                    else float(np.mean(residual)),
                }
            reports.append(
                {"episode": episode, "frame": frame, "raw_id_source": source, "strata": strata}
            )
    return {
        "interpretation": (
            "descriptive centre-ray diagnostics; no fitted tolerance "
            "or all-pixel alignment decision"
        ),
        "frames": reports,
    }


def _revision_path(root: Path, index: int) -> Path:
    return root / "ledger" / f"revision-{index:04d}.json"


def initialise_ledger(root: Path, binding: Mapping[str, object]) -> Path:
    if root.exists():
        raise QualificationFailure("qualification output root already exists")
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
    directory = root / "ledger"
    paths = sorted(directory.glob("revision-*.json")) if directory.is_dir() else []
    if (
        not paths
        or set(directory.iterdir()) != set(paths)
        or paths != [_revision_path(root, index) for index in range(len(paths))]
    ):
        raise QualificationFailure("ledger absent, gapped, or contains foreign entries")
    records: list[dict[str, Any]] = []
    predecessor: str | None = None
    next_ordinal = 0
    expecting_terminal = False
    for index, path in enumerate(paths):
        record = json.loads(path.read_text())
        if (
            record.get("schema") != LEDGER_SCHEMA
            or record.get("revision") != index
            or record.get("predecessor_sha256") != predecessor
        ):
            raise QualificationFailure("ledger hash chain invalid")
        predecessor = digest_file(path)
        records.append(record)
        if any(prior.get("event") == "failed" for prior in records[:-1]):
            raise QualificationFailure("failed qualification event is not terminal")
        if index == 0:
            if record.get("event") != "initialised":
                raise QualificationFailure("ledger initial event invalid")
            validate_plan(record.get("plan"))
            continue
        if not expecting_terminal:
            if (
                record.get("event") != "reserved"
                or record.get("attempt_ordinal") != next_ordinal
                or record.get("attempt_name") != fixed_attempts()[next_ordinal].name
            ):
                raise QualificationFailure("ledger reservation order invalid")
            expecting_terminal = True
        else:
            if (
                record.get("event") not in {"complete", "failed"}
                or record.get("attempt_ordinal") != next_ordinal
            ):
                raise QualificationFailure("ledger terminal event invalid")
            expecting_terminal = False
            next_ordinal += 1
    if not allow_stopped and records[-1]["event"] in {"reserved", "failed"}:
        raise QualificationFailure("reserved/failed qualification permanently stops")
    return records


def append_revision(
    root: Path, event: str, attempt: Attempt, details: Mapping[str, object] | None = None
) -> Path:
    records = validate_ledger(root, allow_stopped=True)
    tail = records[-1]
    if event == "reserved":
        if tail["event"] in {"reserved", "failed"}:
            raise QualificationFailure("stopped qualification cannot reserve")
        complete = [
            record["attempt_ordinal"] for record in records if record["event"] == "complete"
        ]
        if complete != list(range(attempt.ordinal)):
            raise QualificationFailure("wrong next qualification attempt")
    elif event in {"complete", "failed"}:
        if tail.get("event") != "reserved" or tail.get("attempt_ordinal") != attempt.ordinal:
            raise QualificationFailure("terminal qualification event lacks reservation")
    else:
        raise QualificationFailure("unknown qualification ledger event")
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


def run_attempt(
    output_root: Path,
    attempt: Attempt,
    source_root: Path,
    binding: Mapping[str, object],
    generate: Callable[[Path, RendererAdapter], object],
    mujoco: Any,
    observer: Observer = observe_zero_sample_osmesa,
    assess: Callable[[Path, object], Mapping[str, object]] | None = None,
) -> Path:
    lock = AttemptLock(output_root)
    lock.path = output_root / "osmesa-joint0-attempt.lock"
    lock.acquire()
    terminal: Path | None = None
    try:
        records = validate_ledger(output_root)
        append_revision(output_root, "reserved", attempt)
        dataset = output_root / "datasets" / attempt.name
        dataset.mkdir(parents=True, exist_ok=False)
        receipt_path = output_root / "receipts" / f"{attempt.name}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt: dict[str, object] = {
            "schema": SCHEMA,
            "attempt": attempt.__dict__ | {"name": attempt.name},
            "status": "reserved",
            "plan": records[0]["plan"],
            "source": dict(binding),
            "invocation": {"argv": list(sys.argv), "cwd": os.getcwd(), "host": platform.node()},
        }
        adapter: RendererAdapter | None = None
        try:
            require_osmesa_linux()
            validate_bindings(source_root, binding)
            receipt["runtime"] = bind_pristine_sdk(mujoco)
            completed_records = [record for record in records if record.get("event") == "complete"]
            context_baseline = None
            if completed_records:
                first = fixed_attempts()[0]
                first_receipt = json.loads(
                    (output_root / "receipts" / f"{first.name}.json").read_text(encoding="utf-8")
                )
                context_baseline = context_runtime_binding(first_receipt["contexts"][0])
            with RendererAdapter(mujoco, observer, attempt, context_baseline) as active:
                adapter = active
                manifest = generate(dataset, active)
            assert adapter is not None
            adapter.validate_complete()
            receipt["dataset_manifest"] = (
                manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else manifest
            )
            receipt["contexts"] = adapter.contexts
            receipt["native_events"] = adapter.native_events
            receipt["constructor_restored"] = adapter.restored
            receipt["cleanup_complete"] = adapter.cleanup_complete
            receipt["constructor_failures"] = adapter.constructor_failures
            receipt["artifacts"] = artifact_manifest(dataset)
            receipt["permissions"] = permission_probes(dataset)
            receipt["assessment"] = dict(assess(dataset, manifest)) if assess is not None else {}
            receipt["status"] = STATUS
            publish_bytes(receipt_path, canonical(receipt))
        except BaseException as exc:
            if adapter is not None:
                receipt["contexts"] = adapter.contexts
                receipt["native_events"] = adapter.native_events
                receipt["constructor_restored"] = adapter.restored
                receipt["cleanup_complete"] = adapter.cleanup_complete
                receipt["constructor_failures"] = adapter.constructor_failures
            receipt["status"] = "failed"
            receipt["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc)),
            }
            publish_bytes(receipt_path, canonical(receipt))
            terminal = append_revision(
                output_root,
                "failed",
                attempt,
                {"error_type": type(exc).__name__, "error_message": str(exc)},
            )
            raise
        terminal = append_revision(
            output_root,
            "complete",
            attempt,
            {"receipt_sha256": digest_file(receipt_path), "status": STATUS},
        )
        lock.release_after_success(terminal)
        return terminal
    except BaseException:
        raise
