"""Finite qualification of opt-in canonical paired generation.

This owns a new ledger namespace. Historical capture-study ledgers are never
initialized, resumed or modified here. Native monitoring is scoped to this
qualification utility; the production paired API installs no global hooks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from epsbench.data.paths import open_owned_regular_file
from epsbench.diagnostics.osmesa_joint0_qualification import (
    bind_pristine_sdk,
    compare_profile_outputs,
    compare_repeat_artifacts,
    context_runtime_binding,
    geometry_diagnostics,
    permission_probes,
    validate_child_source_provenance,
    validate_source_linkage,
)
from epsbench.diagnostics.revision_capture import AttemptLock, publish_bytes
from epsbench.diagnostics.shared_raster_capture import git_binding, observe_shared_context
from epsbench.schema import (
    CanonicalPairedFrameRecord,
    DatasetManifest,
    Modality,
    ModalityPermissionSet,
)
from epsbench.sim.canonical_paired import (
    SceneMapEntry,
    decode_id_colors,
    observe_canonical_paired_state,
    require_supported_runtime,
    stable_state,
    validate_saved_state,
)
from epsbench.utils.canonical import canonical_json_bytes, logical_array_hash

SCHEMA = "canonical_paired_qualification/v1"
LEDGER_SCHEMA = "canonical_paired_qualification_ledger/v1"
STATUS = "FINITE_CANONICAL_PAIRED_CORRESPONDENCE_QUALIFIED_METRIC_AND_GEOMETRY_SEPARATE"
_PROFILES = ("legacy_solid_base_v1", "legacy_solid_alternate_v1")
_MONITOR_LOCK = threading.Lock()


class CanonicalQualificationFailure(RuntimeError):
    """A finite qualification invariant failed; its namespace must be preserved."""


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
    def modalities(self) -> tuple[str, ...]:
        per_pose = (
            ("rgb", "canonical_pair", "counterfactual_segmentation")
            if self.family == "single_occluder"
            else ("rgb", "canonical_pair")
        )
        return per_pose * 8

    @property
    def calls_per_episode(self) -> int:
        return 6 if self.family == "single_occluder" else 4


def fixed_attempts() -> tuple[Attempt, ...]:
    values = [
        (family, profile, repeat)
        for family in ("single_occluder", "corridor")
        for profile in _PROFILES
        for repeat in (0, 1)
    ]
    return tuple(Attempt(i, *value) for i, value in enumerate(values))


def plan() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "review_profile": "DUAL_REVIEW",
        "evidence_class": "PUBLIC_REPOSITORY_ONLY",
        "phase_gate_effect": "NONE",
        "seed": 1729,
        "episodes_per_attempt": 4,
        "resolution": [160, 120],
        "capture_mode": "canonical_paired",
        "component_topology": False,
        "backend": "osmesa",
        "offsamples": 0,
        "runtime": {
            "python": "3.11.15",
            "mujoco": "3.12.0",
            "numpy": "2.4.6",
            "PyOpenGL": "3.1.10",
            "glfw": "2.10.2",
        },
        "attempts": [
            {
                **asdict(a),
                "name": a.name,
                "contexts": 4,
                "native_render_calls": len(a.modalities),
                "native_readbacks": len(a.modalities),
            }
            for a in fixed_attempts()
        ],
        "limits": {
            "attempts": 8,
            "contexts": 32,
            "ordinary_endpoints": 64,
            "rgb_draws": 64,
            "paired_draws": 64,
            "counterfactual_draws": 32,
            "native_render_calls": 160,
            "native_readbacks": 160,
        },
        "repeat_exclusion": "dataset-relative run.json only",
        "depth_term": "native SDK readback before metric conversion",
        "readback_atomicity": "sequential color then depth; not hardware-atomic",
        "failure_rule": "failed or orphaned reservation permanently stops this namespace",
        "completion_interpretation": STATUS,
    }


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read(root: Path, relative: str) -> bytes:
    with open_owned_regular_file(root, relative) as owned:
        return owned.payload


def tree_snapshot(root: Path) -> list[dict[str, object]]:
    """Bind all saved files, including volatile run metadata, without following aliases."""
    if not root.is_dir() or root.is_symlink():
        raise CanonicalQualificationFailure("dataset tree is absent or linked")
    result: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CanonicalQualificationFailure("linked artifact in qualification tree")
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        payload = _read(root, name)
        result.append({"path": name, "bytes": len(payload), "sha256": _digest(payload)})
    if not result:
        raise CanonicalQualificationFailure("empty qualification dataset")
    return result


def _revision_path(root: Path, index: int) -> Path:
    return root / "canonical-paired-ledger" / f"revision-{index:04d}.json"


def initialise_ledger(
    root: Path, binding: Mapping[str, object], runtime: Mapping[str, object]
) -> Path:
    if root.exists() or root.is_symlink():
        raise CanonicalQualificationFailure("canonical qualification root already exists")
    root.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": 0,
        "predecessor_sha256": None,
        "event": "initialised",
        "plan": plan(),
        "binding": dict(binding),
        "runtime": dict(runtime),
    }
    path = _revision_path(root, 0)
    publish_bytes(path, canonical_json_bytes(record))
    return path


def validate_ledger(root: Path, *, allow_stopped: bool = False) -> list[dict[str, Any]]:
    directory = root / "canonical-paired-ledger"
    paths = sorted(directory.glob("revision-*.json")) if directory.is_dir() else []
    if (
        directory.is_symlink()
        or not paths
        or set(directory.iterdir()) != set(paths)
        or paths != [_revision_path(root, i) for i in range(len(paths))]
    ):
        raise CanonicalQualificationFailure("canonical ledger is absent, foreign or gapped")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    ordinal = 0
    pending = False
    stopped = False
    for i, path in enumerate(paths):
        payload = _read(root, path.relative_to(root).as_posix())
        record = json.loads(payload)
        expected_fields = (
            {"schema", "revision", "predecessor_sha256", "event", "plan", "binding", "runtime"}
            if i == 0
            else {
                "schema",
                "revision",
                "predecessor_sha256",
                "event",
                "attempt_ordinal",
                "attempt_name",
                "details",
            }
        )
        if (
            not isinstance(record, dict)
            or set(record) != expected_fields
            or record["schema"] != LEDGER_SCHEMA
            or type(record["revision"]) is not int
            or record["revision"] != i
            or record["predecessor_sha256"] != previous
            or stopped
        ):
            raise CanonicalQualificationFailure("canonical ledger chain or fields differ")
        if i == 0:
            if (
                record["event"] != "initialised"
                or record["plan"] != plan()
                or not isinstance(record["binding"], dict)
                or not record["binding"]
                or not isinstance(record["runtime"], dict)
                or not record["runtime"]
            ):
                raise CanonicalQualificationFailure("canonical initial binding or plan differs")
        else:
            if (
                ordinal >= len(fixed_attempts())
                or type(record["attempt_ordinal"]) is not int
                or record["attempt_ordinal"] != ordinal
                or record["attempt_name"] != fixed_attempts()[ordinal].name
                or not isinstance(record["details"], dict)
            ):
                raise CanonicalQualificationFailure("canonical attempt identity/order differs")
            if not pending:
                if record["event"] != "reserved":
                    raise CanonicalQualificationFailure("canonical terminal lacks reservation")
                pending = True
            else:
                if record["event"] not in {"complete", "failed"}:
                    raise CanonicalQualificationFailure("canonical reservation lacks terminal")
                stopped = record["event"] == "failed"
                pending = False
                ordinal += 1
        previous = _digest(payload)
        records.append(record)
    if not allow_stopped and (pending or stopped):
        raise CanonicalQualificationFailure(
            "failed/orphaned reservation permanently stops namespace"
        )
    return records


def append_revision(
    root: Path, event: str, attempt: Attempt, details: Mapping[str, object] | None = None
) -> Path:
    if attempt not in fixed_attempts():
        raise CanonicalQualificationFailure("attempt differs from fixed qualification plan")
    records = validate_ledger(root, allow_stopped=True)
    tail = records[-1]
    completed = [r["attempt_ordinal"] for r in records if r["event"] == "complete"]
    if event == "reserved":
        if tail["event"] not in {"initialised", "complete"} or completed != list(
            range(attempt.ordinal)
        ):
            raise CanonicalQualificationFailure("cannot reserve this canonical attempt")
    elif event in {"complete", "failed"}:
        if tail["event"] != "reserved" or tail["attempt_ordinal"] != attempt.ordinal:
            raise CanonicalQualificationFailure("terminal lacks matching canonical reservation")
    else:
        raise CanonicalQualificationFailure("unknown canonical ledger event")
    index = len(records)
    record = {
        "schema": LEDGER_SCHEMA,
        "revision": index,
        "predecessor_sha256": _digest(
            _read(root, _revision_path(root, index - 1).relative_to(root).as_posix())
        ),
        "event": event,
        "attempt_ordinal": attempt.ordinal,
        "attempt_name": attempt.name,
        "details": dict(details or {}),
    }
    path = _revision_path(root, index)
    publish_bytes(path, canonical_json_bytes(record))
    return path


class _RendererProxy:
    def __init__(self, owner: NativeMonitor, renderer: Any, index: int) -> None:
        self._owner, self._renderer, self._index = owner, renderer, index
        self._closed = False
        self._update: dict[str, object] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._renderer, name)

    def update_scene(self, data: Any, *args: Any, **kwargs: Any) -> object:
        self._owner._check_thread()
        option = kwargs.get("scene_option")
        self._update = {
            "data_identity": id(data),
            "camera": kwargs.get("camera", args[0] if args else None),
            "geomgroup": None
            if option is None
            else np.asarray(option.geomgroup).astype(int).tolist(),
        }
        return self._renderer.update_scene(data, *args, **kwargs)

    def close(self) -> None:
        self._owner._check_thread()
        if not self._closed:
            self._renderer.close()
            self._closed = True
            self._owner.contexts[self._index]["close_complete"] = True


class NativeMonitor:
    """Qualification-only native observer, with pristine functions restored in finally."""

    def __init__(
        self, mujoco: Any, attempt: Attempt, baseline: Mapping[str, object] | None = None
    ) -> None:
        self.mujoco, self.attempt = mujoco, attempt
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
            raise CanonicalQualificationFailure("native qualification thread differs")

    def __enter__(self) -> NativeMonitor:
        self._check_thread()
        bind_pristine_sdk(self.mujoco)
        if not _MONITOR_LOCK.acquire(blocking=False):
            raise CanonicalQualificationFailure("native monitor is already active")
        self.originals = self.mujoco.Renderer, self.mujoco.mjr_render, self.mujoco.mjr_readPixels

        def terminal(callback: Callable[..., Any]) -> Callable[..., Any]:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                self._check_thread()
                if self.failed:
                    raise CanonicalQualificationFailure(
                        "native monitor permanently stopped after failure"
                    )
                try:
                    return callback(*args, **kwargs)
                except BaseException:
                    self.failed = True
                    raise

            return wrapped

        self.mujoco.Renderer = terminal(self._construct)
        self.mujoco.mjr_render = terminal(self._draw)
        self.mujoco.mjr_readPixels = terminal(self._read_pixels)
        return self

    def _construct(self, model: Any, *args: Any, **kwargs: Any) -> _RendererProxy:
        self._check_thread()
        if len(self.proxies) >= 4 or int(model.vis.quality.offsamples) != 0:
            raise CanonicalQualificationFailure("constructor budget or sample request differs")
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
            if self.contexts and context_runtime_binding(facts) != context_runtime_binding(
                self.contexts[0]
            ):
                raise CanonicalQualificationFailure("native context runtime changed within batch")
            if self.baseline is not None and context_runtime_binding(facts) != self.baseline:
                raise CanonicalQualificationFailure("native context differs from completed prefix")
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
        raise CanonicalQualificationFailure("native call uses an unowned renderer/rect/context")

    def _draw(self, rect: Any, scene: Any, context: Any) -> object:
        proxy = self._lookup(rect, context)
        index = len(self.events)
        if index >= len(self.attempt.modalities) or (
            self.events and self.events[-1]["readback_calls"] != 1
        ):
            raise CanonicalQualificationFailure("native render budget/order differs before draw")
        if scene is not proxy._renderer._scene or proxy._update is None:
            raise CanonicalQualificationFailure("native draw lacks the owned scene/update")
        if proxy._index != index // self.attempt.calls_per_episode:
            raise CanonicalQualificationFailure("native context/episode sequence differs")
        modality = self.attempt.modalities[index]
        group = proxy._update["geomgroup"]
        if (modality == "counterfactual_segmentation") != (
            isinstance(group, list) and len(group) > 1 and group[1] == 0
        ):
            raise CanonicalQualificationFailure("counterfactual scene membership differs")
        segment = bool(scene.flags[int(self.mujoco.mjtRndFlag.mjRND_SEGMENT)])
        idcolor = bool(scene.flags[int(self.mujoco.mjtRndFlag.mjRND_IDCOLOR)])
        if segment != (modality != "rgb") or idcolor != (modality != "rgb"):
            raise CanonicalQualificationFailure("native draw modality flags differ")
        event: dict[str, Any] = {
            "sequence": index,
            "context_index": proxy._index,
            "frame_index": (index % self.attempt.calls_per_episode)
            // (self.attempt.calls_per_episode // 2),
            "modality": modality,
            "render_calls": 1,
            "readback_calls": 0,
            "scene_update": copy.deepcopy(proxy._update),
            "draw_input": copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer))),
        }
        self.events.append(event)
        assert self.originals is not None
        result = self.originals[1](rect, scene, context)
        event["draw_output"] = copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer)))
        return result

    def _read_pixels(self, color: Any, depth: Any, rect: Any, context: Any) -> object:
        proxy = self._lookup(rect, context)
        if not self.events or self.events[-1]["readback_calls"] != 0:
            raise CanonicalQualificationFailure("native readback lacks a unique draw")
        event = self.events[-1]
        if proxy._index != event["context_index"] or event["scene_update"] != proxy._update:
            raise CanonicalQualificationFailure("native readback scene/update differs")
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
            raise CanonicalQualificationFailure("native readback destinations differ")
        before = copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer)))
        if before != event.get("draw_output"):
            raise CanonicalQualificationFailure("native producer changed before readback")
        event["read_input"] = before
        event["readback_calls"] = 1
        assert self.originals is not None
        result = self.originals[2](color, depth, rect, context)
        event["read_output"] = copy.deepcopy(dict(observe_canonical_paired_state(proxy._renderer)))
        if event["read_output"] != before:
            raise CanonicalQualificationFailure("native producer changed across readback")
        retained_color = np.ascontiguousarray(np.flipud(np.asarray(color).copy()))
        event["color_image_logical_sha256"] = logical_array_hash(retained_color)
        if paired:
            event["native_depth_logical_sha256"] = logical_array_hash(
                np.ascontiguousarray(np.flipud(np.asarray(depth).copy()))
            )
        elif event["modality"] == "counterfactual_segmentation":
            mapping = tuple(
                SceneMapEntry(**item) for item in cast(list[dict[str, int]], before["scene_map"])
            )
            event["counterfactual_raw_geom_logical_sha256"] = logical_array_hash(
                decode_id_colors(retained_color, mapping)
            )
        return result

    def __exit__(self, typ: object, value: object, tb: object) -> Literal[False]:
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
        if value is None and error is not None:
            raise error
        return False

    def validate_complete(self) -> None:
        if (
            len(self.constructor_attempts) != 4
            or self.failed
            or len(self.contexts) != 4
            or len(self.events) != len(self.attempt.modalities)
            or not self.restored
            or not self.cleanup_complete
            or any(
                event["readback_calls"] != 1 or "read_output" not in event for event in self.events
            )
        ):
            raise CanonicalQualificationFailure(
                "native qualification schedule or cleanup is incomplete"
            )


def paired_permission_probes(dataset: Path) -> dict[str, object]:
    from epsbench.data.loader import DatasetLoader, PermissionDeniedError

    required = {Modality.DEPTH, Modality.MUJOCO_GEOM_IDS, Modality.PRIVILEGED_GENERATION_RECORDS}
    results: dict[str, object] = {}
    for missing in (None, *sorted(required, key=lambda value: value.value)):
        permissions = (
            ModalityPermissionSet(allowed=frozenset(required - {missing}))
            if missing
            else ModalityPermissionSet.ecological_only()
        )
        loader = DatasetLoader(dataset, permissions)

        def forbidden(*_args: object) -> Any:
            raise CanonicalQualificationFailure(
                "paired permission probe reached transition/file access"
            )

        loader._transition = forbidden  # type: ignore[method-assign, assignment]
        try:
            loader.read_canonical_paired_output(999, 999)
        except PermissionDeniedError:
            results["ecological_only" if missing is None else "without_" + missing.value] = (
                "denied_before_transition"
            )
        else:
            raise CanonicalQualificationFailure("paired permission denial failed")
    return results


def _verify_native_evidence(
    dataset: Path, attempt: Attempt, receipt: Mapping[str, Any]
) -> dict[str, object]:
    from epsbench.data.loader import DatasetLoader

    loader = DatasetLoader(dataset, ModalityPermissionSet.all_modalities())
    manifest = loader.read_dataset_manifest()
    if manifest.schema_version != "0.1.0-dev.11" or len(manifest.episodes) != 4:
        raise CanonicalQualificationFailure("qualification child is not a complete paired dataset")
    contexts, events = receipt.get("contexts"), receipt.get("native_events")
    if (
        not isinstance(contexts, list)
        or len(contexts) != 4
        or not isinstance(events, list)
        or len(events) != len(attempt.modalities)
        or receipt.get("constructor_restored") is not True
        or receipt.get("cleanup_complete") is not True
    ):
        raise CanonicalQualificationFailure("saved native evidence counts/cleanup differ")
    expected_constructor_attempts = [
        {"ordinal": i, "status": "complete", "context_index": i} for i in range(4)
    ]
    if receipt.get("constructor_attempts") != expected_constructor_attempts:
        raise CanonicalQualificationFailure("saved constructor attempt accounting differs")
    baseline = context_runtime_binding(contexts[0])
    if any(
        context_runtime_binding(item) != baseline or item.get("close_complete") is not True
        for item in contexts
    ):
        raise CanonicalQualificationFailure("saved native context binding/cleanup differs")
    operational = json.loads(_read(dataset, "run.json")).get("canonical_paired_events")
    if not isinstance(operational, list) or len(operational) != 8:
        raise CanonicalQualificationFailure("canonical operational evidence membership differs")
    for index, modality in enumerate(attempt.modalities):
        event = events[index]
        episode_index = index // attempt.calls_per_episode
        frame_index = (index % attempt.calls_per_episode) // (attempt.calls_per_episode // 2)
        if (
            event.get("sequence") != index
            or event.get("modality") != modality
            or event.get("context_index") != episode_index
            or event.get("frame_index") != frame_index
            or event.get("render_calls") != 1
            or event.get("readback_calls") != 1
        ):
            raise CanonicalQualificationFailure("saved native call schedule differs")
        draw, before, after = (
            event.get(key) for key in ("draw_output", "read_input", "read_output")
        )
        if not isinstance(draw, dict) or draw != before or draw != after:
            raise CanonicalQualificationFailure("saved draw/read producer state differs")
        main = contexts[episode_index]["offscreen_attachments"]["offFBO"]
        if (
            draw.get("read_framebuffer_binding") != main["framebuffer"]
            or draw.get("draw_framebuffer_binding") != main["framebuffer"]
            or draw.get("offscreen_attachments") != contexts[episode_index]["offscreen_attachments"]
        ):
            raise CanonicalQualificationFailure(
                "saved native framebuffer/attachment identity differs"
            )
        if context_runtime_binding(contexts[episode_index]) != draw.get("context_runtime"):
            raise CanonicalQualificationFailure("saved context runtime differs from draw")
        draw_input = event.get("draw_input")
        if not isinstance(draw_input, dict):
            raise CanonicalQualificationFailure("saved draw input is absent")
        installed_by_draw = {
            "projection_matrix_float32",
            "modelview_matrix_float32",
            "clip_origin",
            "clip_depth_mode",
        }
        if {key: value for key, value in draw_input.items() if key not in installed_by_draw} != {
            key: value for key, value in draw.items() if key not in installed_by_draw
        }:
            raise CanonicalQualificationFailure("saved scene/camera inputs drifted across draw")
        current = draw.get("actual_current_context")
        if (
            type(current) is not int
            or current <= 0
            or current != draw.get("expected_current_context")
            or current != contexts[episode_index].get("osmesa_context_identity")
        ):
            raise CanonicalQualificationFailure("saved current context is invalid")
        if draw.get("segment_enabled") is not (modality != "rgb") or draw.get(
            "idcolor_enabled"
        ) is not (modality != "rgb"):
            raise CanonicalQualificationFailure("saved native modality flags differ")
        transition = loader._transition(episode_index)
        frame = transition.before if frame_index == 0 else transition.after
        if not isinstance(frame, CanonicalPairedFrameRecord):
            raise CanonicalQualificationFailure("canonical frame provenance is absent")
        pair = loader.read_canonical_paired_output(episode_index, frame_index)
        if modality == "rgb":
            if event.get("color_image_logical_sha256") != frame.rgb.logical_sha256:
                raise CanonicalQualificationFailure("ordinary RGB native output binding differs")
        elif modality == "counterfactual_segmentation":
            cf = pair.provenance.counterfactual_provenance
            if (
                cf is None
                or event.get("counterfactual_raw_geom_logical_sha256") != cf.artifact_logical_sha256
            ):
                raise CanonicalQualificationFailure("counterfactual native output binding differs")
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
                raise CanonicalQualificationFailure(
                    "paired native/canonical endpoint binding differs"
                )
            observations = item.get("observations")
            if not isinstance(observations, dict) or set(observations) != {
                "draw_input",
                "draw_output",
                "read_input",
                "read_output",
            }:
                raise CanonicalQualificationFailure("owned capture observations are incomplete")
            if any(observations[key] != event[key] for key in observations):
                raise CanonicalQualificationFailure(
                    "owned capture and native monitor observations differ"
                )
    return baseline


def compare_paired_appearance(base: Path, alternate: Path) -> int:
    from epsbench.data.loader import DatasetLoader

    loaders = [
        DatasetLoader(root, ModalityPermissionSet.all_modalities()) for root in (base, alternate)
    ]
    compared = 0
    for episode in range(4):
        for frame in (0, 1):
            a, b = [loader.read_canonical_paired_output(episode, frame) for loader in loaders]
            if (
                a.native_id_rgb.tobytes() != b.native_id_rgb.tobytes()
                or a.native_depth_pre_metric.tobytes() != b.native_depth_pre_metric.tobytes()
                or a.provenance.scene_map != b.provenance.scene_map
                or a.provenance.raw_to_opaque_surface_ids != b.provenance.raw_to_opaque_surface_ids
                or a.provenance.opaque_surface_labels != b.provenance.opaque_surface_labels
                or a.producer_state != b.producer_state
            ):
                raise CanonicalQualificationFailure(
                    "appearance changed paired native/producer terms"
                )
            compared += 2
    return compared


def assess_dataset(output: Path, attempt: Attempt, dataset: Path) -> dict[str, object]:
    from epsbench.data.inspect import create_inspection_image
    from epsbench.data.validate import validate_dataset

    validate_dataset(dataset)
    image = output / "inspections" / f"{attempt.name}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    create_inspection_image(dataset, 0, image)
    result: dict[str, object] = {
        "validate_dataset": "passed",
        "inspection": {
            "path": image.relative_to(output).as_posix(),
            "sha256": _digest(image.read_bytes()),
        },
        "permissions": paired_permission_probes(dataset),
        "legacy_permissions": permission_probes(dataset),
        "geometry_descriptive_only": geometry_diagnostics(dataset),
    }
    if attempt.repeat == 1:
        previous = fixed_attempts()[attempt.ordinal - 1]
        result["repeat_exact_members"] = compare_repeat_artifacts(
            output / "datasets" / previous.name, dataset
        )
    if attempt.profile == _PROFILES[1]:
        base = next(
            item
            for item in fixed_attempts()
            if item.family == attempt.family
            and item.profile == _PROFILES[0]
            and item.repeat == attempt.repeat
        )
        base_path = output / "datasets" / base.name
        result["appearance"] = compare_profile_outputs(base_path, dataset)
        result["appearance_paired_array_pairs"] = compare_paired_appearance(base_path, dataset)
    return result


def validate_completed_prefix(
    output: Path, records: list[dict[str, Any]]
) -> dict[str, object] | None:
    allowed_root_members = {
        "canonical-paired-ledger",
        "canonical-paired-attempt.lock",
        "datasets",
        "canonical-paired-receipts",
        "inspections",
    }
    if any(path.is_symlink() or path.name not in allowed_root_members for path in output.iterdir()):
        raise CanonicalQualificationFailure("foreign or linked qualification root member")
    completed = [record for record in records if record["event"] == "complete"]
    expected_names = {a.name for a in fixed_attempts()[: len(completed)]}
    for folder, suffix in (
        ("datasets", ""),
        ("canonical-paired-receipts", ".json"),
        ("inspections", ".png"),
    ):
        directory = output / folder
        found = (
            {
                path.name.removesuffix(suffix) if suffix else path.name
                for path in directory.iterdir()
            }
            if directory.is_dir()
            else set()
        )
        if found != expected_names:
            raise CanonicalQualificationFailure(
                "saved qualification membership differs from completed prefix"
            )
    baseline: dict[str, object] | None = None
    for attempt, terminal in zip(fixed_attempts(), completed, strict=False):
        relative = f"canonical-paired-receipts/{attempt.name}.json"
        payload = _read(output, relative)
        if terminal["details"] != {"receipt_sha256": _digest(payload), "status": STATUS}:
            raise CanonicalQualificationFailure("completed ledger receipt binding differs")
        receipt = json.loads(payload)
        dataset = output / "datasets" / attempt.name
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("status") != STATUS
            or receipt.get("plan") != plan()
            or receipt.get("attempt") != {**asdict(attempt), "name": attempt.name}
            or receipt.get("source") != records[0]["binding"]
            or receipt.get("runtime") != records[0]["runtime"]
            or receipt.get("dataset_artifacts") != tree_snapshot(dataset)
        ):
            raise CanonicalQualificationFailure(
                "saved receipt plan/source/runtime/artifacts differ"
            )
        manifest = DatasetManifest.model_validate_json(_read(dataset, "manifest.json"))
        if receipt.get("dataset_manifest") != manifest.model_dump(mode="json"):
            raise CanonicalQualificationFailure("saved receipt and dataset manifest differ")
        validate_child_source_provenance(manifest, records[0]["binding"])
        current = _verify_native_evidence(dataset, attempt, receipt)
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise CanonicalQualificationFailure("cross-batch context runtime differs")
        inspection = receipt.get("assessment", {}).get("inspection", {})
        expected_image = f"inspections/{attempt.name}.png"
        if inspection != {"path": expected_image, "sha256": _digest(_read(output, expected_image))}:
            raise CanonicalQualificationFailure("saved inspection binding differs")
    return baseline


def generate_attempt(source: Path, attempt: Attempt, dataset: Path) -> DatasetManifest:
    from epsbench.appearance import load_appearance_registry, load_evaluation_seed_registry
    from epsbench.config import load_config
    from epsbench.data.generate import generate_dataset

    name = "benchmark_v0.yaml" if attempt.family == "single_occluder" else "corridor_v0.yaml"
    config = load_config(source / "configs" / name)
    config = config.model_copy(
        update={"appearance": config.appearance.model_copy(update={"profile_id": attempt.profile})}
    )
    if config.seed != 1729 or (config.render.width, config.render.height) != (160, 120):
        raise CanonicalQualificationFailure("configuration differs from finite plan")
    return generate_dataset(
        config,
        4,
        dataset,
        appearance_registry=load_appearance_registry(
            source / "configs/appearance_candidates_v0.yaml"
        ),
        seed_registry=load_evaluation_seed_registry(
            source / "configs/evaluation_seed_candidates_v0.yaml"
        ),
        component_topology=False,
        capture_mode="canonical_paired",
    )


def run_attempt(output: Path, source: Path, attempt: Attempt) -> Path:
    source = validate_source_linkage(source)
    records = validate_ledger(output)
    lock = AttemptLock(output)
    lock.path = output / "canonical-paired-attempt.lock"
    lock.acquire()
    try:
        baseline = validate_completed_prefix(output, records)
        append_revision(output, "reserved", attempt)
    except BaseException:
        if lock.fd is not None:
            os.close(lock.fd)
            lock.fd = None
        raise
    dataset = output / "datasets" / attempt.name
    receipt_path = output / "canonical-paired-receipts" / f"{attempt.name}.json"
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "attempt": {**asdict(attempt), "name": attempt.name},
        "plan": plan(),
        "source": records[0]["binding"],
        "runtime": records[0]["runtime"],
        "invocation": {"argv": list(sys.argv), "cwd": os.getcwd(), "host": platform.node()},
    }
    monitor: NativeMonitor | None = None
    try:
        require_supported_runtime(os.environ.get("MUJOCO_GL", ""))
        if git_binding(source) != records[0]["binding"]:
            raise CanonicalQualificationFailure("source changed since initialization")
        import mujoco

        runtime = bind_pristine_sdk(mujoco)
        if runtime != records[0]["runtime"]:
            raise CanonicalQualificationFailure("runtime changed since initialization")
        dataset.mkdir(parents=True, exist_ok=False)
        with NativeMonitor(mujoco, attempt, baseline) as monitor:
            manifest = generate_attempt(source, attempt, dataset)
        monitor.validate_complete()
        validate_child_source_provenance(manifest, records[0]["binding"])
        receipt.update(
            {
                "dataset_manifest": manifest.model_dump(mode="json"),
                "constructor_attempts": monitor.constructor_attempts,
                "contexts": monitor.contexts,
                "native_events": monitor.events,
                "constructor_restored": monitor.restored,
                "cleanup_complete": monitor.cleanup_complete,
            }
        )
        _verify_native_evidence(dataset, attempt, receipt)
        receipt["assessment"] = assess_dataset(output, attempt, dataset)
        if git_binding(source) != records[0]["binding"] or bind_pristine_sdk(mujoco) != runtime:
            raise CanonicalQualificationFailure("source/runtime changed during attempt")
        receipt["dataset_artifacts"] = tree_snapshot(dataset)
        receipt["status"] = STATUS
        publish_bytes(receipt_path, canonical_json_bytes(receipt))
    except BaseException as error:
        if monitor is not None:
            receipt.update(
                {
                    "constructor_attempts": monitor.constructor_attempts,
                    "contexts": monitor.contexts,
                    "native_events": monitor.events,
                    "constructor_restored": monitor.restored,
                    "cleanup_complete": monitor.cleanup_complete,
                }
            )
        receipt["status"] = "failed"
        receipt["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        # A publication failure may leave a completed receipt. Preserve it rather than overwriting.
        if not receipt_path.exists():
            publish_bytes(receipt_path, canonical_json_bytes(receipt))
        append_revision(
            output,
            "failed",
            attempt,
            {"receipt_sha256": _digest(_read(output, receipt_path.relative_to(output).as_posix()))},
        )
        if lock.fd is not None:
            os.close(lock.fd)
            lock.fd = None
        raise
    terminal = append_revision(
        output,
        "complete",
        attempt,
        {
            "receipt_sha256": _digest(_read(output, receipt_path.relative_to(output).as_posix())),
            "status": STATUS,
        },
    )
    lock.release_after_success(terminal)
    return terminal


def validate_saved(output: Path, source: Path) -> dict[str, object]:
    from epsbench.data.validate import validate_dataset

    records = validate_ledger(output)
    if (output / "canonical-paired-attempt.lock").exists():
        raise CanonicalQualificationFailure("retained attempt lock permanently stops namespace")
    if git_binding(source) != records[0]["binding"]:
        raise CanonicalQualificationFailure("source binding differs from saved qualification")
    require_supported_runtime(os.environ.get("MUJOCO_GL", ""))
    import mujoco

    if bind_pristine_sdk(mujoco) != records[0]["runtime"]:
        raise CanonicalQualificationFailure("runtime binding differs from saved qualification")
    baseline = validate_completed_prefix(output, records)
    completed = [record for record in records if record["event"] == "complete"]
    for attempt in fixed_attempts()[: len(completed)]:
        dataset = output / "datasets" / attempt.name
        validate_dataset(dataset)
        if attempt.repeat == 1:
            compare_repeat_artifacts(
                output / "datasets" / fixed_attempts()[attempt.ordinal - 1].name, dataset
            )
        if attempt.profile == _PROFILES[1]:
            base = next(
                a
                for a in fixed_attempts()
                if a.family == attempt.family
                and a.profile == _PROFILES[0]
                and a.repeat == attempt.repeat
            )
            compare_profile_outputs(output / "datasets" / base.name, dataset)
            compare_paired_appearance(output / "datasets" / base.name, dataset)
    complete = len(completed) == 8
    return {
        "status": STATUS if complete else "in_progress",
        "completed_attempts": len(completed),
        "execution_complete": complete,
        "context_binding": baseline,
        "phase_gate_effect": "NONE",
    }
