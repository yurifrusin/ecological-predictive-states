"""Scoped native guard for eight-transition capture; importing creates no graphics context.

Review profile: DUAL_REVIEW. Evidence class: PUBLIC_REPOSITORY_ONLY.
Phase-gate effect: NONE. This adapter conveys no capture authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from epsbench.data.paths import open_owned_regular_file
from epsbench.schema import CanonicalPairedFrameRecord, ModalityPermissionSet
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash

MODALITIES = ("rgb", "canonical_pair", "counterfactual_segmentation") * 2
_MONITOR_LOCK = threading.Lock()
_GRAPHICS_PREPARED = False


class NativeCaptureFailure(RuntimeError):
    """Native integrity failure; the owning executor must permanently stop."""


def bind_pristine_sdk(module: Any) -> object:
    from epsbench.diagnostics.osmesa_joint0_qualification import bind_pristine_sdk as bind

    return bind(module)


def context_runtime_binding(facts: Mapping[str, object]) -> dict[str, object]:
    from epsbench.diagnostics.osmesa_joint0_qualification import context_runtime_binding as bind

    return bind(facts)


def observe_shared_context(
    renderer: Any, model: Any, width: int, height: int
) -> Mapping[str, object]:
    from epsbench.diagnostics.shared_raster_capture import observe_shared_context as observe

    return observe(renderer, model, width, height)


def observe_canonical_paired_state(renderer: Any) -> Mapping[str, object]:
    from epsbench.sim.canonical_paired import observe_canonical_paired_state as observe

    return observe(renderer)


def stable_state(state: Mapping[str, object]) -> Mapping[str, object]:
    from epsbench.sim.canonical_paired import stable_state as stable

    return stable(state)


def validate_saved_state(state: Mapping[str, object], width: int, height: int) -> None:
    from epsbench.sim.canonical_paired import validate_saved_state as validate

    validate(state, width, height)


def _validate_context(facts: Mapping[str, Any]) -> None:
    binding = context_runtime_binding(facts)
    if (
        binding["actual_backend"] != "osmesa"
        or binding["width"] != 160
        or binding["height"] != 120
        or any(
            binding[k] != 0
            for k in (
                "requested_offsamples",
                "actual_offsamples",
                "model_offsamples",
                "sample_buffers",
                "samples",
                "gl_samples",
            )
        )
        or "osmesa" not in str(binding["context_module"]).lower()
        or not any(
            s in str(binding["gl_renderer"]).lower()
            for s in ("llvmpipe", "softpipe", "software", "swrast")
        )
    ):
        raise NativeCaptureFailure("context is not software zero-sample OSMesa 160x120")


def _read(root: Path, relative: str) -> bytes:
    with open_owned_regular_file(root, relative) as owned:
        return owned.payload


class _RendererProxy:
    def __init__(self, owner: NativeMonitor, renderer: Any, index: int) -> None:
        self._owner, self._renderer, self._index = owner, renderer, index
        self._closed = False
        self._update: dict[str, object] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._renderer, name)

    def update_scene(self, data: Any, *args: Any, **kwargs: Any) -> object:
        self._owner._check_thread()
        if self._owner.failed or self._closed:
            raise NativeCaptureFailure("native monitor permanently stopped or renderer closed")
        option = kwargs.get("scene_option")
        self._update = {
            "data_identity": id(data),
            "camera": kwargs.get("camera", args[0] if args else None),
            "geomgroup": None
            if option is None
            else np.asarray(option.geomgroup).astype(int).tolist(),
        }
        try:
            return self._renderer.update_scene(data, *args, **kwargs)
        except BaseException as error:
            self._owner.failed = True
            self._owner.failed_calls.append(
                {
                    "operation": "update_scene",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            raise

    def close(self) -> None:
        self._owner._check_thread()
        if not self._closed:
            self._renderer.close()
            self._closed = True
            self._owner.contexts[self._index]["close_complete"] = True


class NativeMonitor:
    """Qualification-only native observer, with pristine functions restored in finally."""

    def __init__(self, mujoco: Any, baseline: Mapping[str, object] | None = None) -> None:
        self.mujoco = mujoco
        self.failed_calls: list[dict[str, object]] = []
        self.baseline = dict(baseline) if baseline is not None else None
        self.constructor_attempts: list[dict[str, object]] = []
        self.failed = False
        self.thread = threading.get_ident()
        self.contexts: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.proxies: list[_RendererProxy] = []
        self.originals: tuple[Any, Any, Any] | None = None
        self.restored = False
        self.cleanup_complete = False

    def _check_thread(self) -> None:
        if threading.get_ident() != self.thread:
            self.failed = True
            self.failed_calls.append(
                {
                    "operation": "thread",
                    "error_type": "NativeCaptureFailure",
                    "error_message": "native qualification thread differs",
                }
            )
            raise NativeCaptureFailure("native qualification thread differs")

    def __enter__(self) -> NativeMonitor:
        self._check_thread()
        bind_pristine_sdk(self.mujoco)
        if not _MONITOR_LOCK.acquire(blocking=False):
            raise NativeCaptureFailure("native monitor is already active")
        self.originals = self.mujoco.Renderer, self.mujoco.mjr_render, self.mujoco.mjr_readPixels

        def terminal(callback: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                if self.failed:
                    raise NativeCaptureFailure("native monitor permanently stopped after failure")
                try:
                    self._check_thread()
                    return callback(*args, **kwargs)
                except BaseException as error:
                    self.failed = True
                    self.failed_calls.append(
                        {
                            "operation": callback.__name__,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    raise

            return wrapped

        self.mujoco.Renderer = terminal(self._construct)
        self.mujoco.mjr_render = terminal(self._draw)
        self.mujoco.mjr_readPixels = terminal(self._read_pixels)
        return self

    def _construct(self, model: Any, *args: Any, **kwargs: Any) -> _RendererProxy:
        self._check_thread()
        if len(self.constructor_attempts) >= 1 or int(model.vis.quality.offsamples) != 0:
            raise NativeCaptureFailure("constructor budget or sample request differs")
        height = kwargs.get("height", args[0] if args else 240)
        width = kwargs.get("width", args[1] if len(args) > 1 else 320)
        if (width, height) != (160, 120):
            raise NativeCaptureFailure("constructor budget or sample request differs")
        assert self.originals is not None
        constructor_attempt: dict[str, object] = {
            "ordinal": len(self.constructor_attempts),
            "status": "reserved",
        }
        self.constructor_attempts.append(constructor_attempt)
        renderer: Any = None
        try:
            renderer = self.originals[0](model, *args, **kwargs)
            facts = dict(observe_shared_context(renderer, model, renderer.width, renderer.height))
            _validate_context(facts)
            if self.contexts and context_runtime_binding(facts) != context_runtime_binding(
                self.contexts[0]
            ):
                raise NativeCaptureFailure("native context runtime changed within batch")
            if self.baseline is not None and context_runtime_binding(facts) != self.baseline:
                raise NativeCaptureFailure("native context differs from completed prefix")
            index = len(self.proxies)
            facts.update(
                {
                    "context_index": index,
                    "renderer_identity": id(renderer),
                    "context_identity": id(renderer._mjr_context),
                    "close_complete": False,
                }
            )
            self.contexts.append(facts)
            proxy = _RendererProxy(self, renderer, index)
            self.proxies.append(proxy)
            constructor_attempt.update({"status": "complete", "context_index": index})
            return proxy
        except BaseException as error:
            constructor_attempt.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            if renderer is not None:
                try:
                    renderer.close()
                    constructor_attempt["close_complete"] = True
                except BaseException as close_error:
                    constructor_attempt["close_error"] = str(close_error)
                    constructor_attempt["close_complete"] = False
            raise

    def _lookup(self, rect: Any, context: Any) -> _RendererProxy:
        self._check_thread()
        for proxy in self.proxies:
            if (
                context is proxy._renderer._mjr_context
                and rect is proxy._renderer._rect
                and not proxy._closed
            ):
                return proxy
        raise NativeCaptureFailure("native call uses an unowned renderer/rect/context")

    def _draw(self, rect: Any, scene: Any, context: Any) -> object:
        proxy = self._lookup(rect, context)
        index = len(self.events)
        if index >= len(MODALITIES) or (self.events and self.events[-1]["readback_calls"] != 1):
            raise NativeCaptureFailure("native render budget/order differs before draw")
        if scene is not proxy._renderer._scene or proxy._update is None:
            raise NativeCaptureFailure("native draw lacks the owned scene/update")
        if proxy._index != index // 6:
            raise NativeCaptureFailure("native context/episode sequence differs")
        modality = MODALITIES[index]
        group = proxy._update["geomgroup"]
        if (modality == "counterfactual_segmentation") != (
            isinstance(group, list) and len(group) > 1 and group[1] == 0
        ):
            raise NativeCaptureFailure("counterfactual scene membership differs")
        segment = bool(scene.flags[int(self.mujoco.mjtRndFlag.mjRND_SEGMENT)])
        idcolor = bool(scene.flags[int(self.mujoco.mjtRndFlag.mjRND_IDCOLOR)])
        if segment != (modality != "rgb") or idcolor != (modality != "rgb"):
            raise NativeCaptureFailure("native draw modality flags differ")
        event: dict[str, Any] = {
            "sequence": index,
            "context_index": proxy._index,
            "frame_index": (index % 6) // (6 // 2),
            "modality": modality,
            "render_calls": 1,
            "readback_calls": 0,
            "scene_update": copy.deepcopy(proxy._update),
            "draw_input": copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer))),
        }
        self._validate_state(event["draw_input"])
        self.events.append(event)
        assert self.originals is not None
        result = self.originals[1](rect, scene, context)
        event["draw_output"] = copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer)))
        self._validate_state(event["draw_output"])
        return result

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        facts = self.contexts[0]
        current = state.get("actual_current_context")
        main = facts["offscreen_attachments"]["offFBO"]["framebuffer"]
        if (
            state.get("context_runtime") != context_runtime_binding(facts)
            or type(current) is not int
            or current <= 0
            or current != state.get("expected_current_context")
            or current != facts.get("osmesa_context_identity")
            or state.get("offscreen_attachments") != facts["offscreen_attachments"]
            or state.get("read_framebuffer_binding") != main
            or state.get("draw_framebuffer_binding") != main
        ):
            raise NativeCaptureFailure("native provenance drift before call")

    def _read_pixels(self, color: Any, depth: Any, rect: Any, context: Any) -> object:
        proxy = self._lookup(rect, context)
        if not self.events or self.events[-1]["readback_calls"] != 0:
            raise NativeCaptureFailure("native readback lacks a unique draw")
        event = self.events[-1]
        if proxy._index != event["context_index"] or event["scene_update"] != proxy._update:
            raise NativeCaptureFailure("native readback scene/update differs")
        paired = event["modality"] == "canonical_pair"
        if (
            color is None
            or np.asarray(color).shape != (120, 160, 3)
            or np.asarray(color).dtype != np.uint8
            or (
                paired
                and (
                    depth is None
                    or np.asarray(depth).shape != (120, 160)
                    or np.asarray(depth).dtype != np.float32
                )
            )
            or (not paired and depth is not None)
        ):
            raise NativeCaptureFailure("native readback destinations differ")
        before = copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer)))
        self._validate_state(before)
        if before != event.get("draw_output"):
            raise NativeCaptureFailure("native producer changed before readback")
        if paired:
            validate_saved_state(stable_state(before), 160, 120)
        event["read_input"] = before
        event["readback_calls"] = 1
        assert self.originals is not None
        result = self.originals[2](color, depth, rect, context)
        event["read_output"] = copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer)))
        if event["read_output"] != before:
            raise NativeCaptureFailure("native producer changed across readback")
        retained_color = np.ascontiguousarray(np.flipud(np.asarray(color).copy()))
        event["color_image_logical_sha256"] = logical_array_hash(retained_color)
        if paired:
            event["native_depth_logical_sha256"] = logical_array_hash(
                np.ascontiguousarray(np.flipud(np.asarray(depth).copy()))
            )
        elif event["modality"] == "counterfactual_segmentation":
            event["counterfactual_raw_geom_logical_sha256"] = counterfactual_hash(
                retained_color, cast(list[dict[str, int]], before["scene_map"])
            )
        return result

    def __exit__(self, typ: object, value: object, tb: object) -> Literal[False]:
        if value is not None:
            self.failed = True
        error: BaseException | None = None
        assert self.originals is not None
        try:
            self.mujoco.Renderer, self.mujoco.mjr_render, self.mujoco.mjr_readPixels = (
                self.originals
            )
            self.restored = (
                self.mujoco.Renderer,
                self.mujoco.mjr_render,
                self.mujoco.mjr_readPixels,
            ) == self.originals
            for proxy in reversed(self.proxies):
                try:
                    proxy.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
            self.cleanup_complete = all(proxy._closed for proxy in self.proxies)
        finally:
            _MONITOR_LOCK.release()
        if error is not None:
            self.failed = True
            self.failed_calls.append(
                {
                    "operation": "cleanup",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        if value is None and error is not None:
            raise error
        return False

    def validate_complete(self) -> None:
        if (
            len(self.constructor_attempts) != 1
            or self.failed
            or len(self.contexts) != 1
            or len(self.events) != len(MODALITIES)
            or not self.restored
            or not self.cleanup_complete
            or any(
                event["readback_calls"] != 1 or "read_output" not in event for event in self.events
            )
        ):
            raise NativeCaptureFailure("native qualification schedule or cleanup is incomplete")

    def receipt(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "constructor_attempts": self.constructor_attempts,
                "contexts": self.contexts,
                "native_events": self.events,
                "failed_calls": self.failed_calls,
                "failed": self.failed,
                "constructor_restored": self.restored,
                "cleanup_complete": self.cleanup_complete,
            }
        )


def validate_saved_native(dataset: Path, receipt: Mapping[str, Any]) -> dict[str, object]:
    from epsbench.data.loader import DatasetLoader

    loader = DatasetLoader(dataset, ModalityPermissionSet.all_modalities())
    manifest = loader.read_dataset_manifest()
    if manifest.schema_version != "0.1.0-dev.11" or len(manifest.episodes) != 1:
        raise NativeCaptureFailure("qualification child is not a complete paired dataset")
    contexts, events = receipt.get("contexts"), receipt.get("native_events")
    if (
        not isinstance(contexts, list)
        or len(contexts) != 1
        or not isinstance(events, list)
        or len(events) != len(MODALITIES)
        or receipt.get("constructor_restored") is not True
        or receipt.get("cleanup_complete") is not True
    ):
        raise NativeCaptureFailure("saved native evidence counts/cleanup differ")
    expected_constructor_attempts = [
        {"ordinal": i, "status": "complete", "context_index": i} for i in range(1)
    ]
    if receipt.get("constructor_attempts") != expected_constructor_attempts:
        raise NativeCaptureFailure("saved constructor attempt accounting differs")
    if receipt.get("failed") is not False or receipt.get("failed_calls") != []:
        raise NativeCaptureFailure("saved native monitor failed")
    _validate_context(contexts[0])
    baseline = context_runtime_binding(contexts[0])
    if any(
        context_runtime_binding(item) != baseline or item.get("close_complete") is not True
        for item in contexts
    ):
        raise NativeCaptureFailure("saved native context binding/cleanup differs")
    operational = json.loads(_read(dataset, "run.json")).get("canonical_paired_events")
    if not isinstance(operational, list) or len(operational) != 2:
        raise NativeCaptureFailure("canonical operational evidence membership differs")
    for index, modality in enumerate(MODALITIES):
        event = events[index]
        episode_index = index // 6
        frame_index = (index % 6) // (6 // 2)
        if (
            event.get("sequence") != index
            or event.get("modality") != modality
            or event.get("context_index") != episode_index
            or event.get("frame_index") != frame_index
            or event.get("render_calls") != 1
            or event.get("readback_calls") != 1
        ):
            raise NativeCaptureFailure("saved native call schedule differs")
        draw, before, after = (
            event.get(key) for key in ("draw_output", "read_input", "read_output")
        )
        if not isinstance(draw, dict) or draw != before or draw != after:
            raise NativeCaptureFailure("saved draw/read producer state differs")
        main = contexts[episode_index]["offscreen_attachments"]["offFBO"]
        if (
            draw.get("read_framebuffer_binding") != main["framebuffer"]
            or draw.get("draw_framebuffer_binding") != main["framebuffer"]
            or draw.get("offscreen_attachments") != contexts[episode_index]["offscreen_attachments"]
        ):
            raise NativeCaptureFailure("saved native framebuffer/attachment identity differs")
        if context_runtime_binding(contexts[episode_index]) != draw.get("context_runtime"):
            raise NativeCaptureFailure("saved context runtime differs from draw")
        draw_input = event.get("draw_input")
        if not isinstance(draw_input, dict):
            raise NativeCaptureFailure("saved draw input is absent")
        installed_by_draw = {
            "projection_matrix_float32",
            "modelview_matrix_float32",
            "clip_origin",
            "clip_depth_mode",
        }
        if {key: value for key, value in draw_input.items() if key not in installed_by_draw} != {
            key: value for key, value in draw.items() if key not in installed_by_draw
        }:
            raise NativeCaptureFailure("saved scene/camera inputs drifted across draw")
        current = draw.get("actual_current_context")
        if (
            type(current) is not int
            or current <= 0
            or current != draw.get("expected_current_context")
            or current != contexts[episode_index].get("osmesa_context_identity")
        ):
            raise NativeCaptureFailure("saved current context is invalid")
        if draw.get("segment_enabled") is not (modality != "rgb") or draw.get(
            "idcolor_enabled"
        ) is not (modality != "rgb"):
            raise NativeCaptureFailure("saved native modality flags differ")
        transition = loader._transition(episode_index)
        frame = transition.before if frame_index == 0 else transition.after
        if not isinstance(frame, CanonicalPairedFrameRecord):
            raise NativeCaptureFailure("canonical frame provenance is absent")
        pair = loader.read_canonical_paired_output(episode_index, frame_index)
        if modality == "rgb":
            if event.get("color_image_logical_sha256") != frame.rgb.logical_sha256:
                raise NativeCaptureFailure("ordinary RGB native output binding differs")
        elif modality == "counterfactual_segmentation":
            cf = pair.provenance.counterfactual_provenance
            if (
                cf is None
                or event.get("counterfactual_raw_geom_logical_sha256") != cf.artifact_logical_sha256
            ):
                raise NativeCaptureFailure("counterfactual native output binding differs")
        else:
            validate_saved_state(stable_state(draw), 160, 120)
            item = operational[episode_index * 2 + frame_index]
            if (
                item.get("episode_id") != transition.episode_id
                or item.get("frame_index") != frame_index
                or item.get("endpoint_logical_sha256") != pair.provenance.endpoint_logical_sha256
                or pair.producer_state != stable_state(draw)
                or event.get("color_image_logical_sha256")
                != pair.provenance.native_id_rgb.logical_sha256
                or event.get("native_depth_logical_sha256")
                != pair.provenance.native_depth_pre_metric.logical_sha256
            ):
                raise NativeCaptureFailure("paired native/canonical endpoint binding differs")
            observations = item.get("observations")
            if not isinstance(observations, dict) or set(observations) != {
                "draw_input",
                "draw_output",
                "read_input",
                "read_output",
            }:
                raise NativeCaptureFailure("owned capture observations are incomplete")
            if any(observations[key] != event[key] for key in observations):
                raise NativeCaptureFailure("owned capture and native monitor observations differ")
    return baseline


def counterfactual_hash(color: Any, mapping: list[dict[str, int]]) -> str:
    from epsbench.sim.canonical_paired import SceneMapEntry, decode_id_colors

    return logical_array_hash(decode_id_colors(color, tuple(SceneMapEntry(**i) for i in mapping)))


def prepare_graphics_environment() -> None:
    """Set both backends before graphics import; reject previously imported graphics."""
    global _GRAPHICS_PREPARED
    if not _GRAPHICS_PREPARED and any(
        name == "mujoco"
        or name.startswith("mujoco.")
        or name == "OpenGL"
        or name.startswith("OpenGL.")
        for name in sys.modules
    ):
        raise NativeCaptureFailure("graphics imported before controlled OSMesa environment")
    if any(
        os.environ.get(key) not in (("osmesa",) if _GRAPHICS_PREPARED else (None, "osmesa"))
        for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
    ):
        raise NativeCaptureFailure("graphics backend environment differs")
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    _GRAPHICS_PREPARED = True


def capture_cell(
    cell: Mapping[str, Any],
    dataset_path: Path,
    *,
    source: Path | None = None,
    baseline: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Capture one already reserved cell; never initialize or resume a study root.

    The executor must bind source/runtime and provide containment and owner authority.
    A failure carries its partial native receipt on ``native_receipt`` for retention.
    """
    source = source or Path(__file__).resolve().parents[3]
    membership = json.loads(
        (source / "docs/protocols/topology-eight-transition-v1-membership.json").read_bytes()
    )
    if dict(cell) not in membership["records"]:
        raise NativeCaptureFailure("cell differs from fixed membership")
    if dataset_path.exists() or dataset_path.is_symlink():
        raise NativeCaptureFailure("dataset path already exists")
    from epsbench.config import SingleOccluderConfig

    config = SingleOccluderConfig.model_validate(cell["resolved_configuration"])
    if hashlib.sha256(canonical_json_bytes(config)).hexdigest() != cell["configuration_sha256"]:
        raise NativeCaptureFailure("cell configuration hash differs")
    prepare_graphics_environment()
    import mujoco

    from epsbench.appearance import load_appearance_registry, load_evaluation_seed_registry
    from epsbench.data.generate import generate_dataset
    from epsbench.sim.canonical_paired import require_supported_runtime

    require_supported_runtime("osmesa")
    monitor = NativeMonitor(mujoco, baseline)
    try:
        with monitor:
            generate_dataset(
                config,
                1,
                dataset_path,
                appearance_registry=load_appearance_registry(
                    source / "configs/appearance_candidates_v0.yaml"
                ),
                seed_registry=load_evaluation_seed_registry(
                    source / "configs/evaluation_seed_candidates_v0.yaml"
                ),
                component_topology=False,
                capture_mode="canonical_paired",
            )
        monitor.validate_complete()
        receipt = monitor.receipt()
        receipt["runtime_binding"] = validate_saved_native(dataset_path, receipt)
        return receipt
    except BaseException as error:
        # Failure information belongs to the owning executor's retained receipts.
        error.native_receipt = monitor.receipt()  # type: ignore[attr-defined]
        raise
