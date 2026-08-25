"""Whole-dataset schema, alignment, identity, and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from epsbench.annotations import derive_boundary_structure, derive_visibility
from epsbench.config import BenchmarkConfig
from epsbench.data.identity import compute_dataset_logical_hash, compute_ecological_label_hash
from epsbench.schema import (
    ArtifactRecord,
    CameraInstrumentation,
    DatasetManifest,
    PrivilegedInstrumentation,
    TransitionRecord,
)
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
    sha256_file,
)
from epsbench.utils.seeding import derive_seed


class DatasetValidationError(ValueError):
    """Raised when a dataset fails the public data contract."""


def _path(root: Path, record: ArtifactRecord) -> Path:
    candidate = (root / record.path).resolve()
    if not candidate.is_relative_to(root):
        raise DatasetValidationError(f"artifact escapes dataset root: {record.path}")
    if not candidate.is_file():
        raise DatasetValidationError(f"missing artifact: {record.path}")
    if candidate.stat().st_size != record.byte_count:
        raise DatasetValidationError(f"artifact byte count mismatch: {record.path}")
    if sha256_file(candidate) != record.file_sha256:
        raise DatasetValidationError(f"artifact file hash mismatch: {record.path}")
    return candidate


def _verify_json(root: Path, record: ArtifactRecord) -> dict[str, Any]:
    path = _path(root, record)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetValidationError(f"invalid JSON artifact: {record.path}") from error
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"JSON artifact must contain an object: {record.path}")
    logical_bytes = canonical_json_bytes(payload)
    if sha256_bytes(logical_bytes) != record.logical_sha256:
        raise DatasetValidationError(f"artifact logical hash mismatch: {record.path}")
    if record.dtype != "json" or record.shape != (len(logical_bytes),):
        raise DatasetValidationError(f"JSON artifact metadata mismatch: {record.path}")
    return payload


def _verify_array(
    root: Path,
    record: ArtifactRecord,
    array: np.ndarray[Any, Any],
) -> None:
    _path(root, record)
    if tuple(array.shape) != record.shape or array.dtype.str != record.dtype:
        raise DatasetValidationError(f"array metadata mismatch: {record.path}")
    if logical_array_hash(array) != record.logical_sha256:
        raise DatasetValidationError(f"array logical hash mismatch: {record.path}")


def _load_rgb(root: Path, record: ArtifactRecord) -> np.ndarray[Any, Any]:
    path = _path(root, record)
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except OSError as error:
        raise DatasetValidationError(f"unreadable RGB artifact: {record.path}") from error
    _verify_array(root, record, rgb)
    return rgb


def _load_npy(root: Path, record: ArtifactRecord) -> np.ndarray[Any, Any]:
    path = _path(root, record)
    try:
        array = cast(np.ndarray[Any, Any], np.load(path, allow_pickle=False))
    except (OSError, ValueError) as error:
        raise DatasetValidationError(f"unreadable NumPy artifact: {record.path}") from error
    _verify_array(root, record, array)
    return array


def validate_dataset(root: Path) -> DatasetManifest:
    """Validate every declared artifact and cross-record invariant."""

    resolved_root = root.resolve()
    manifest_path = resolved_root / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetValidationError("missing manifest.json")
    try:
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise DatasetValidationError("dataset manifest failed schema validation") from error
    if compute_dataset_logical_hash(manifest) != manifest.dataset_logical_sha256:
        raise DatasetValidationError("dataset logical hash mismatch")

    config_payload = _verify_json(resolved_root, manifest.resolved_config)
    config = BenchmarkConfig.model_validate_json(canonical_json_bytes(config_payload))
    if sha256_bytes(canonical_json_bytes(config)) != manifest.config_logical_sha256:
        raise DatasetValidationError("resolved configuration hash mismatch")
    if config.seed != manifest.root_seed:
        raise DatasetValidationError("manifest seed does not match resolved configuration")
    if config.appearance.variant != manifest.appearance_variant:
        raise DatasetValidationError("appearance variant is inconsistent")

    for episode in manifest.episodes:
        if episode.episode_id != f"episode-{episode.episode_index:06d}":
            raise DatasetValidationError("episode identifier does not match its index")
        if episode.episode_seed != derive_seed(config.seed, f"episode:{episode.episode_index}"):
            raise DatasetValidationError("episode derived seed mismatch")
        transition_payload = _verify_json(resolved_root, episode.transition)
        transition = TransitionRecord.model_validate_json(canonical_json_bytes(transition_payload))
        if transition.episode_id != episode.episode_id:
            raise DatasetValidationError("episode and transition identifiers differ")
        if compute_ecological_label_hash(transition) != transition.ecological_label_sha256:
            raise DatasetValidationError("ecological-label hash mismatch")
        if transition.ecological_label_sha256 != episode.ecological_label_sha256:
            raise DatasetValidationError("episode ecological-label hash mismatch")

        instrumentation_payload = _verify_json(resolved_root, episode.privileged_instrumentation)
        instrumentation = PrivilegedInstrumentation.model_validate_json(
            canonical_json_bytes(instrumentation_payload)
        )
        if instrumentation.episode_id != episode.episode_id:
            raise DatasetValidationError("instrumentation episode identifier mismatch")
        if instrumentation.appearance_variant != manifest.appearance_variant:
            raise DatasetValidationError("instrumentation appearance variant mismatch")
        if len(set(instrumentation.raw_geom_ids.values())) != len(instrumentation.raw_geom_ids):
            raise DatasetValidationError("raw MuJoCo geom identifiers must be unique")
        if set(instrumentation.raw_geom_ids) != set(instrumentation.raw_geom_world_positions):
            raise DatasetValidationError("raw geom coordinate records are incomplete")
        if set(instrumentation.raw_to_opaque_surface_ids.values()) != {
            surface.surface_id for surface in transition.surfaces
        }:
            raise DatasetValidationError("privileged surface remapping is incomplete")
        if set(instrumentation.raw_to_opaque_surface_ids) != {
            str(raw_id) for raw_id in instrumentation.raw_geom_ids.values()
        }:
            raise DatasetValidationError("privileged raw-ID remapping keys are inconsistent")

        frame_arrays: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = []
        for frame in (transition.before, transition.after):
            rgb = _load_rgb(resolved_root, frame.rgb)
            depth = _load_npy(resolved_root, frame.depth)
            segmentation = _load_npy(resolved_root, frame.segmentation)
            if rgb.shape[:2] != depth.shape or depth.shape != segmentation.shape:
                raise DatasetValidationError("RGB, depth, and segmentation are misaligned")
            camera_payload = _verify_json(resolved_root, frame.camera_world_transform)
            camera = CameraInstrumentation.model_validate_json(canonical_json_bytes(camera_payload))
            if camera.frame_index != frame.frame_index:
                raise DatasetValidationError("camera instrumentation frame mismatch")
            frame_arrays.append((rgb, depth, segmentation))

        if tuple(item[0].shape for item in frame_arrays) != (
            transition.before.rgb.shape,
            transition.after.rgb.shape,
        ):
            raise DatasetValidationError("frame shapes differ from transition metadata")
        if episode.rgb_logical_sha256 != (
            transition.before.rgb.logical_sha256,
            transition.after.rgb.logical_sha256,
        ):
            raise DatasetValidationError("episode RGB hashes differ from frame records")

        declared_labels = {surface.segmentation_label for surface in transition.surfaces}
        observed_labels: set[int] = set()
        for _, _, segmentation in frame_arrays:
            observed_labels.update(int(value) for value in np.unique(segmentation) if value != 0)
        if not observed_labels.issubset(declared_labels):
            raise DatasetValidationError("segmentation contains an undeclared surface label")
        if observed_labels != declared_labels:
            raise DatasetValidationError("a declared surface is absent from both frames")

        derived_visibility, derived_correspondence, derived_events = derive_visibility(
            frame_arrays[0][2], frame_arrays[1][2], transition.surfaces
        )
        if derived_visibility != transition.visibility_states:
            raise DatasetValidationError("visibility-state annotations are inconsistent")
        if derived_correspondence != transition.region_correspondence:
            raise DatasetValidationError("region-correspondence annotations are inconsistent")
        if derived_events != transition.visibility_events:
            raise DatasetValidationError("visibility-event annotations are inconsistent")
        derived_boundaries = (
            derive_boundary_structure(frame_arrays[0][2], transition.surfaces, 0),
            derive_boundary_structure(frame_arrays[1][2], transition.surfaces, 1),
        )
        if derived_boundaries != transition.boundary_structures:
            raise DatasetValidationError("boundary annotations are inconsistent")
    return manifest
