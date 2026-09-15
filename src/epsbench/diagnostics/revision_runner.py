"""Failure-preserving orchestration for revision capture cells.

The capture core depends on a small renderer-stack protocol so CPU synthetic tests can
exercise every stage without constructing graphics contexts.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from epsbench.diagnostics.revision_capture import (
    SCHEMA,
    ArtifactWriter,
    AttemptLock,
    RevisionCaptureFailure,
    StudyCell,
    append_revision,
    canonical_json_bytes,
    load_anchor_array,
    publish_bytes,
    validate_ledger,
    verify_input_archive,
)


class RendererStack(Protocol):
    role: str
    width: int
    height: int

    def make_current(self) -> None: ...
    def set_pose(self, pose_name: str) -> Mapping[str, object]: ...
    def update_scene(self) -> None: ...
    def render_rgb(self) -> npt.NDArray[np.uint8]: ...
    def enable_depth(self) -> None: ...
    def render_depth(self) -> npt.NDArray[np.float32]: ...
    def disable_depth(self) -> None: ...
    def enable_segmentation(self) -> None: ...
    def scene_map(self) -> Mapping[int, tuple[int, int]]: ...
    def render_segmentation(self, out: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]: ...
    def disable_segmentation(self) -> None: ...
    def provenance(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


class StackFactory(Protocol):
    def __call__(self, samples: int, role: str) -> RendererStack: ...


@dataclass(frozen=True)
class PreparedEpisode:
    episode_seed: int
    source: Mapping[str, object]
    config: Mapping[str, object]
    model: Mapping[str, object]
    geometry_facts: Mapping[str, object]
    pose_facts: Mapping[str, Mapping[str, object]]

    def validate(self) -> None:
        if self.episode_seed < 0:
            raise RevisionCaptureFailure("episode seed must be an exact non-negative integer")
        if set(self.pose_facts) != {"before", "after"}:
            raise RevisionCaptureFailure("prepared episode must bind exactly before/after poses")
        required_model = {
            "normalized_mjb_sha256",
            "actual_models",
            "semantic_facts_sha256",
        }
        if not required_model <= set(self.model):
            raise RevisionCaptureFailure("prepared model binding is incomplete")
        canonical_json_bytes(
            {
                "source": dict(self.source),
                "config": dict(self.config),
                "model": dict(self.model),
                "geometry_facts": dict(self.geometry_facts),
                "pose_facts": {key: dict(value) for key, value in self.pose_facts.items()},
            }
        )


@dataclass
class StageFailure(Exception):
    stage: str
    cause: BaseException
    receipt: dict[str, Any]


_REQUIRED_PROVENANCE = {
    "role",
    "requested_offsamples",
    "actual_offsamples",
    "actual_backend",
    "gl_vendor",
    "gl_renderer",
    "gl_version",
    "offscreen_attachments",
    "model_stat_extent",
    "model_vis_map_znear",
    "model_vis_map_zfar",
    "model_mjb_sha256",
    "python_version",
    "mujoco_version",
    "numpy_version",
    "pyopengl_version",
    "glfw_version",
    "package_sha256",
    "binary_sha256",
    "renderer_py_sha256",
}


def _event(receipt: dict[str, Any], stage: str, state: str, **details: object) -> None:
    receipt.setdefault("stage_events", []).append(
        {
            "sequence": len(receipt.get("stage_events", [])),
            "stage": stage,
            "state": state,
            **details,
        }
    )


def _call(receipt: dict[str, Any], stage: str, operation: Callable[[], Any]) -> Any:
    _event(receipt, stage, "started")
    try:
        value = operation()
    except BaseException as exc:
        _event(receipt, stage, "failed", error_type=type(exc).__name__, error_message=str(exc))
        raise StageFailure(stage, exc, receipt) from exc
    _event(receipt, stage, "complete")
    return value


def _current(
    receipt: dict[str, Any], stack: RendererStack, stage: str, operation: Callable[[], Any]
) -> Any:
    _call(receipt, f"{stage}.make_current", stack.make_current)
    return _call(receipt, stage, operation)


def _validate_provenance(
    value: Mapping[str, object], cell: StudyCell, role: str, requested: int
) -> dict[str, object]:
    missing = sorted(key for key in _REQUIRED_PROVENANCE if value.get(key) in (None, ""))
    if missing:
        raise RevisionCaptureFailure("missing renderer provenance: " + ", ".join(missing))
    if (value["role"], value["requested_offsamples"], value["actual_backend"]) != (
        role,
        requested,
        cell.backend,
    ):
        raise RevisionCaptureFailure("renderer identity/backend/request differs from cell")
    actual = int(value["actual_offsamples"])  # type: ignore[call-overload]
    if actual != requested:
        raise RevisionCaptureFailure("actual offsamples differs from requested")
    attachments = value["offscreen_attachments"]
    if not isinstance(attachments, Mapping):
        raise RevisionCaptureFailure("offscreen attachment provenance is not a mapping")
    main = attachments.get("offFBO")
    resolve = attachments.get("offFBO_r")
    if (
        not isinstance(main, Mapping)
        or not main.get("present")
        or int(main.get("draw_framebuffer_samples", -1)) != requested
    ):
        raise RevisionCaptureFailure("live main FBO samples differ from request")
    color = main.get("color0")
    if not isinstance(color, Mapping) or int(color.get("samples", -1)) != requested:
        raise RevisionCaptureFailure("live main attachment samples differ from request")
    if requested > 0:
        if (
            not isinstance(resolve, Mapping)
            or not resolve.get("present")
            or int(resolve.get("draw_framebuffer_samples", -1)) != 0
        ):
            raise RevisionCaptureFailure("multisample resolve FBO is missing or sampled")
    elif isinstance(resolve, Mapping) and resolve.get("present"):
        raise RevisionCaptureFailure("unexpected resolve FBO for zero-sample renderer")
    return dict(value)


def _save_array(
    writer: ArtifactWriter,
    relative: str,
    value: npt.NDArray[Any],
    *,
    complete: bool = True,
    validated: bool = False,
) -> dict[str, object]:
    return writer.array(relative, np.asarray(value), complete=complete, validated=validated)


def _segmentation(
    receipt: dict[str, Any],
    pose: dict[str, Any],
    writer: ArtifactWriter,
    stack: RendererStack,
    prefix: str,
    geom_objtype: int,
) -> None:
    _current(receipt, stack, f"{prefix}.segmentation_enable", stack.enable_segmentation)
    _current(receipt, stack, f"{prefix}.segmentation_update", stack.update_scene)
    mapping = _current(receipt, stack, f"{prefix}.segmentation_scene_map", stack.scene_map)
    triples = [[int(segid), int(pair[0]), int(pair[1])] for segid, pair in sorted(mapping.items())]
    seg = pose["modalities"].setdefault("segmentation", {})
    seg["segid_map"] = writer.json(f"{prefix}-segid-map.json", triples, validated=False)
    out = np.zeros((stack.height, stack.width, 3), dtype=np.uint8)
    # Publish the post-call caller buffer even when decoding/readback raises.
    try:
        pairs = _current(
            receipt,
            stack,
            f"{prefix}.segmentation_readback",
            lambda: stack.render_segmentation(out),
        )
    except StageFailure:
        seg["encoded_rgb"] = _save_array(
            writer,
            f"{prefix}-encoded-rgb.npy",
            out,
            complete=False,
            validated=False,
        )
        raise
    seg["encoded_rgb"] = _save_array(writer, f"{prefix}-encoded-rgb.npy", out, validated=False)
    pairs_array = np.asarray(pairs)
    seg["decoded_pairs"] = _save_array(
        writer, f"{prefix}-decoded-pairs.npy", pairs_array, validated=False
    )
    if pairs_array.dtype != np.int32 or pairs_array.shape != (stack.height, stack.width, 2):
        raise StageFailure(
            f"{prefix}.segmentation_schema",
            RevisionCaptureFailure("segmentation return must be exact HxWx2 int32"),
            receipt,
        )
    packed = (
        out[..., 0].astype(np.int64)
        + 256 * out[..., 1].astype(np.int64)
        + 65536 * out[..., 2].astype(np.int64)
    )
    independently_decoded = np.full(pairs_array.shape, -1, dtype=np.int32)
    for segid, pair in mapping.items():
        independently_decoded[packed == int(segid) + 1] = (int(pair[0]), int(pair[1]))
    independently_decoded = np.flipud(independently_decoded)
    if not np.array_equal(independently_decoded, pairs_array):
        raise StageFailure(
            f"{prefix}.independent_decode",
            RevisionCaptureFailure(
                "encoded caller buffer does not independently decode to returned pairs"
            ),
            receipt,
        )
    raw = np.where(pairs_array[..., 1] == geom_objtype, pairs_array[..., 0], -1).astype(np.int32)
    seg["raw_geom_ids"] = _save_array(writer, f"{prefix}-raw-geom-ids.npy", raw, validated=True)
    for key in ("segid_map", "encoded_rgb", "decoded_pairs"):
        seg[key]["validated"] = True
    seg["readback_requires_vertical_flip"] = True
    _current(receipt, stack, f"{prefix}.segmentation_disable", stack.disable_segmentation)


def _rgb_depth(
    receipt: dict[str, Any],
    pose: dict[str, Any],
    writer: ArtifactWriter,
    stack: RendererStack,
    prefix: str,
) -> None:
    _current(receipt, stack, f"{prefix}.rgb_update", stack.update_scene)
    rgb = np.asarray(_current(receipt, stack, f"{prefix}.rgb_readback", stack.render_rgb))
    pose["modalities"]["rgb"] = _save_array(writer, f"{prefix}-rgb.npy", rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (stack.height, stack.width, 3):
        raise StageFailure(
            f"{prefix}.rgb_schema",
            RevisionCaptureFailure("RGB readback must be exact HxWx3 uint8"),
            receipt,
        )
    pose["modalities"]["rgb"]["validated"] = True
    _current(receipt, stack, f"{prefix}.depth_enable", stack.enable_depth)
    _current(receipt, stack, f"{prefix}.depth_update", stack.update_scene)
    depth = np.asarray(_current(receipt, stack, f"{prefix}.depth_readback", stack.render_depth))
    pose["modalities"]["depth"] = _save_array(writer, f"{prefix}-depth.npy", depth)
    if (
        depth.dtype != np.float32
        or depth.shape != (stack.height, stack.width)
        or not np.isfinite(depth).all()
    ):
        raise StageFailure(
            f"{prefix}.depth_schema",
            RevisionCaptureFailure("depth readback must be finite HxW float32"),
            receipt,
        )
    pose["modalities"]["depth"]["validated"] = True
    _current(receipt, stack, f"{prefix}.depth_disable", stack.disable_depth)


def _capture_pose(
    receipt: dict[str, Any],
    writer: ArtifactWriter,
    pose_name: str,
    prepared: PreparedEpisode,
    primary: RendererStack,
    segmentation: RendererStack,
    geom_objtype: int,
) -> dict[str, Any]:
    pose: dict[str, Any] = {
        "pose_name": pose_name,
        "pose_facts": dict(prepared.pose_facts[pose_name]),
        "geometry_facts": dict(prepared.geometry_facts),
        "modalities": {},
    }
    receipt["poses"].append(pose)
    observed_primary = _current(
        receipt,
        primary,
        f"{pose_name}.primary_set_pose",
        lambda: primary.set_pose(pose_name),
    )
    if dict(observed_primary) != dict(prepared.pose_facts[pose_name]):
        raise StageFailure(
            f"{pose_name}.primary_pose_validation",
            RevisionCaptureFailure("primary pose differs from CPU binding"),
            receipt,
        )
    if segmentation is not primary:
        observed_seg = _current(
            receipt,
            segmentation,
            f"{pose_name}.segmentation_set_pose",
            lambda: segmentation.set_pose(pose_name),
        )
        if dict(observed_seg) != dict(prepared.pose_facts[pose_name]):
            raise StageFailure(
                f"{pose_name}.segmentation_pose_validation",
                RevisionCaptureFailure("hybrid pose differs from CPU binding"),
                receipt,
            )
    _rgb_depth(receipt, pose, writer, primary, pose_name)
    _segmentation(receipt, pose, writer, segmentation, pose_name, geom_objtype)
    return pose


def _cleanup(
    receipt: dict[str, Any], stacks: Sequence[RendererStack], original: BaseException | None
) -> BaseException | None:
    cleanup_error: BaseException | None = None
    for stack in reversed(tuple(dict.fromkeys(stacks))):
        try:
            _current(receipt, stack, f"{stack.role}.cleanup", stack.close)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
    return original if original is not None else cleanup_error


def capture_cell(
    cell: StudyCell,
    prepared: PreparedEpisode,
    factory: StackFactory,
    cell_directory: Path,
    *,
    geom_objtype: int,
    final_validate: Callable[[dict[str, Any], Path], None] | None = None,
) -> dict[str, Any]:
    prepared.validate()
    writer = ArtifactWriter(cell_directory)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "identity": {
            "ordinal": cell.ordinal,
            "family": cell.family,
            "episode_index": cell.episode_index,
            "episode_seed": prepared.episode_seed,
            "backend": cell.backend,
            "policy": cell.policy,
        },
        "source": dict(prepared.source),
        "config": dict(prepared.config),
        "model": dict(prepared.model),
        "renderers": [],
        "poses": [],
        "stage_events": [],
        "invocation": {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "host": platform.node(),
        },
        "state": "reserved",
    }
    primary: RendererStack | None = None
    segmentation: RendererStack | None = None
    original: BaseException | None = None
    try:
        primary_samples = 0 if cell.policy == "joint0" else 4
        primary = _call(
            receipt,
            "primary_context_construct",
            lambda: factory(primary_samples, "primary"),
        )
        primary_provenance = _current(receipt, primary, "primary.provenance", primary.provenance)
        checked_primary = _validate_provenance(primary_provenance, cell, "primary", primary_samples)
        actual_models = prepared.model["actual_models"]
        if not isinstance(actual_models, Mapping) or checked_primary[
            "model_mjb_sha256"
        ] != actual_models.get("primary"):
            raise RevisionCaptureFailure("primary live model hash differs from CPU binding")
        receipt["renderers"].append(checked_primary)
        segmentation = primary
        if cell.policy == "hybrid":
            segmentation = _call(
                receipt,
                "segmentation_context_construct",
                lambda: factory(0, "segmentation"),
            )
            segmentation_provenance = _current(
                receipt,
                segmentation,
                "segmentation.provenance",
                segmentation.provenance,
            )
            checked_segmentation = _validate_provenance(
                segmentation_provenance, cell, "segmentation", 0
            )
            if not isinstance(actual_models, Mapping) or checked_segmentation[
                "model_mjb_sha256"
            ] != actual_models.get("segmentation"):
                raise RevisionCaptureFailure(
                    "segmentation live model hash differs from CPU binding"
                )
            receipt["renderers"].append(checked_segmentation)
        for pose_name in ("before", "after"):
            _capture_pose(
                receipt,
                writer,
                pose_name,
                prepared,
                primary,
                segmentation,
                geom_objtype,
            )
    except BaseException as exc:
        original = exc
    stacks = [stack for stack in (primary, segmentation) if stack is not None]
    terminal = _cleanup(receipt, stacks, original)
    if terminal is None and final_validate is not None:
        try:
            final_validate(receipt, cell_directory)
        except BaseException as exc:
            terminal = exc
            _event(
                receipt,
                "post_capture_integrity",
                "failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
    if terminal is not None:
        receipt["state"] = "failed"
        receipt["failure"] = {
            "error_type": type(terminal).__name__,
            "error_message": str(terminal),
            "stage": terminal.stage if isinstance(terminal, StageFailure) else "cleanup",
            "traceback": "".join(traceback.format_exception(terminal)),
        }
        publish_bytes(cell_directory / "receipt.json", canonical_json_bytes(receipt))
        raise StageFailure(str(receipt["failure"]["stage"]), terminal, receipt)
    receipt["state"] = "complete"
    publish_bytes(cell_directory / "receipt.json", canonical_json_bytes(receipt))
    return receipt


def run_attempt(
    output_root: Path,
    input_archive: Path,
    cell: StudyCell,
    prepare: Callable[[], tuple[PreparedEpisode, StackFactory]],
    *,
    geom_objtype: int,
) -> Path:
    lock = AttemptLock(output_root)
    lock.acquire()
    terminal: Path | None = None
    try:
        records = validate_ledger(output_root)
        append_revision(output_root, "reserved", cell)
        try:
            archive = verify_input_archive(input_archive)
            plan = records[0].get("plan_binding", {})
            if plan.get("archive_sha256") != archive["archive_sha256"]:
                raise RevisionCaptureFailure("ledger archive binding differs")
            prepared, factory = prepare()
            config_hashes = plan.get("config_sha256")
            if not isinstance(config_hashes, Mapping):
                raise RevisionCaptureFailure("initial plan config binding is absent")
            if (
                prepared.source.get("head") != plan.get("source_head")
                or prepared.source.get("tree") != plan.get("source_tree")
                or prepared.source.get("dependency_lock_sha256")
                != plan.get("dependency_lock_sha256")
                or prepared.source.get("config_sha256") != config_hashes.get(cell.family)
                or prepared.source.get("appearance_registry_sha256")
                != plan.get("appearance_registry_sha256")
                or prepared.source.get("seed_registry_sha256") != plan.get("seed_registry_sha256")
            ):
                raise RevisionCaptureFailure(
                    "attempt source/config/registry/lock differs from initial plan"
                )
            cell_directory = output_root / "cells" / cell.name
            cell_directory.mkdir(parents=True, exist_ok=False)

            def final_validate(receipt: dict[str, Any], directory: Path) -> None:
                if (
                    cell.family == "corridor"
                    and cell.episode_index == 0
                    and cell.policy == "joint4"
                ):
                    for pose in receipt["poses"]:
                        name = str(pose["pose_name"])
                        actual_ref = pose["modalities"]["segmentation"]["raw_geom_ids"]
                        actual = np.load(directory / actual_ref["path"], allow_pickle=False)
                        expected = load_anchor_array(input_archive, cell.backend, name)
                        if not np.array_equal(actual, expected):
                            raise RevisionCaptureFailure(
                                f"historical {cell.backend} {name} raw-ID anchor mismatch"
                            )
                verify_input_archive(input_archive)

            receipt = capture_cell(
                cell,
                prepared,
                factory,
                cell_directory,
                geom_objtype=geom_objtype,
                final_validate=final_validate,
            )
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
            {
                "receipt_sha256": receipt
                and __import__("hashlib")
                .sha256((cell_directory / "receipt.json").read_bytes())
                .hexdigest()
            },
        )
        lock.release_after_success(terminal)
        return terminal
    except BaseException:
        # Deliberately retain the lock on all failure/interruption paths.
        raise
