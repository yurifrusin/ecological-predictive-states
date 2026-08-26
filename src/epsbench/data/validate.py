"""Whole-dataset schema, alignment, identity, and artifact validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from epsbench.annotations import derive_boundary_structure, derive_visibility
from epsbench.config import (
    BenchmarkConfig,
    CorridorConfig,
    SingleOccluderConfig,
    parse_config,
)
from epsbench.data.identity import (
    compute_content_provenance_binding,
    compute_dataset_logical_hash,
    compute_ecological_label_hash,
    compute_renderer_execution_provenance_hash,
    compute_source_provenance_hash,
)
from epsbench.schema import (
    Action,
    ArtifactRecord,
    CameraInstrumentation,
    CorridorInstrumentation,
    DatasetManifest,
    OcclusionRelation,
    PrivilegedInstrumentation,
    SceneFamily,
    SingleOccluderInstrumentation,
    TransitionRecord,
    parse_privileged_instrumentation_json,
)
from epsbench.sim import CORRIDOR_SURFACE_NAMES, corridor_generation_seeds
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
    sha256_file,
)
from epsbench.utils.seeding import derive_seed


class DatasetValidationError(ValueError):
    """Raised when a dataset fails the public data contract."""


class _ArtifactRegistry:
    """Reject duplicate logical paths and filesystem aliases across declared roles."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths: set[str] = set()
        self.resolved_paths: set[Path] = set()
        self.file_identities: set[tuple[int, int]] = set()

    def claim(self, record: ArtifactRecord) -> None:
        candidate = (self.root / record.path).resolve()
        if record.path in self.paths:
            raise DatasetValidationError(f"duplicate artifact path: {record.path}")
        if candidate in self.resolved_paths:
            raise DatasetValidationError(f"artifact path aliases another role: {record.path}")
        if candidate.is_file():
            stat = candidate.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in self.file_identities:
                raise DatasetValidationError(f"artifact path aliases another role: {record.path}")
            self.file_identities.add(identity)
        self.paths.add(record.path)
        self.resolved_paths.add(candidate)


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


def _verify_json(
    root: Path,
    record: ArtifactRecord,
    registry: _ArtifactRegistry,
) -> dict[str, Any]:
    registry.claim(record)
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
    if tuple(array.shape) != record.shape or str(array.dtype) != record.dtype:
        raise DatasetValidationError(f"array metadata mismatch: {record.path}")
    if logical_array_hash(array) != record.logical_sha256:
        raise DatasetValidationError(f"array logical hash mismatch: {record.path}")


def _load_rgb(
    root: Path,
    record: ArtifactRecord,
    registry: _ArtifactRegistry,
) -> np.ndarray[Any, Any]:
    registry.claim(record)
    path = _path(root, record)
    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                raise DatasetValidationError(f"RGB artifact must use RGB mode: {record.path}")
            rgb = np.asarray(image, dtype=np.uint8).copy()
    except OSError as error:
        raise DatasetValidationError(f"unreadable RGB artifact: {record.path}") from error
    _verify_array(root, record, rgb)
    return rgb


def _load_npy(
    root: Path,
    record: ArtifactRecord,
    registry: _ArtifactRegistry,
) -> np.ndarray[Any, Any]:
    registry.claim(record)
    path = _path(root, record)
    try:
        array = cast(np.ndarray[Any, Any], np.load(path, allow_pickle=False))
    except (OSError, ValueError) as error:
        raise DatasetValidationError(f"unreadable NumPy artifact: {record.path}") from error
    _verify_array(root, record, array)
    return array


def _require_camera_action_alignment(
    config: BenchmarkConfig,
    transition: TransitionRecord,
    cameras: tuple[CameraInstrumentation, CameraInstrumentation],
    instrumentation: PrivilegedInstrumentation,
) -> None:
    expected_action = Action.model_validate(config.action.model_dump(mode="python"))
    if transition.action != expected_action:
        raise DatasetValidationError("persisted action differs from resolved configuration")
    before, after = cameras
    if isinstance(config, SingleOccluderConfig):
        if not isinstance(instrumentation, SingleOccluderInstrumentation):
            raise DatasetValidationError("single-occluder config requires matching instrumentation")
        expected_before = np.asarray(
            (config.camera.before_lateral, config.camera.forward, config.camera.height),
            dtype=np.float64,
        )
        expected_after = np.asarray(
            (config.camera.after_lateral, config.camera.forward, config.camera.height),
            dtype=np.float64,
        )
    elif isinstance(config, CorridorConfig):
        if not isinstance(instrumentation, CorridorInstrumentation):
            raise DatasetValidationError("corridor config requires matching instrumentation")
        geometry = instrumentation.sampled_geometry
        expected_before = np.asarray(
            (
                geometry.camera_lateral_position,
                geometry.camera_before_forward_position,
                geometry.camera_height,
            ),
            dtype=np.float64,
        )
        expected_after = np.asarray(
            (
                geometry.camera_lateral_position,
                geometry.camera_after_forward_position,
                geometry.camera_height,
            ),
            dtype=np.float64,
        )
        if before != instrumentation.camera_before or after != instrumentation.camera_after:
            raise DatasetValidationError(
                "corridor camera artifacts differ from scene instrumentation"
            )
    else:
        raise DatasetValidationError("unsupported scene configuration")
    before_position = np.asarray(before.camera_world_position, dtype=np.float64)
    after_position = np.asarray(after.camera_world_position, dtype=np.float64)
    if not np.allclose(before_position, expected_before, atol=1e-12, rtol=0.0):
        raise DatasetValidationError("before camera pose differs from resolved configuration")
    if not np.allclose(after_position, expected_after, atol=1e-12, rtol=0.0):
        raise DatasetValidationError("after camera pose differs from resolved configuration")
    expected_delta = np.asarray(
        (
            transition.action.delta_lateral,
            transition.action.delta_forward,
            0.0,
        ),
        dtype=np.float64,
    )
    if not np.allclose(after_position - before_position, expected_delta, atol=1e-12, rtol=0.0):
        raise DatasetValidationError("persisted camera displacement differs from executed action")
    if not np.allclose(
        before.camera_world_rotation_row_major,
        after.camera_world_rotation_row_major,
        atol=1e-12,
        rtol=0.0,
    ):
        raise DatasetValidationError("camera rotation changed despite zero executed yaw")


def _corridor_scene_content_hash(
    episode_seed: int,
    instrumentation: CorridorInstrumentation,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "episode_seed": episode_seed,
                "sampled_geometry": instrumentation.sampled_geometry.model_dump(mode="json"),
                "scene_family": SceneFamily.CORRIDOR,
            }
        )
    )


def _single_occluder_scene_content_hash(episode_seed: int) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "episode_seed": episode_seed,
                "scene_family": SceneFamily.SINGLE_OCCLUDER,
            }
        )
    )


def _require_corridor_instrumentation(
    config: CorridorConfig,
    episode_seed: int,
    instrumentation: CorridorInstrumentation,
) -> None:
    geometry_seed, remapping_seed, appearance_seed = corridor_generation_seeds(episode_seed)
    if instrumentation.generation_seeds.model_dump(mode="python") != {
        "episode_seed": episode_seed,
        "geometry_sampling_seed": geometry_seed,
        "surface_remapping_seed": remapping_seed,
        "appearance_seed": appearance_seed,
    }:
        raise DatasetValidationError("corridor generation seed namespaces are inconsistent")
    geometry = instrumentation.sampled_geometry
    geometry_rng = np.random.default_rng(geometry_seed)
    expected_width = float(
        geometry_rng.uniform(config.geometry.width.minimum, config.geometry.width.maximum)
    )
    expected_length = float(
        geometry_rng.uniform(config.geometry.length.minimum, config.geometry.length.maximum)
    )
    expected_sampled_values = (
        expected_width,
        expected_length,
        config.geometry.wall_height,
        config.camera.lateral_position,
        config.camera.starting_forward_position,
        config.camera.starting_forward_position + config.action.delta_forward,
        config.camera.height,
        config.camera.field_of_view_degrees,
    )
    observed_sampled_values = (
        geometry.width,
        geometry.length,
        geometry.wall_height,
        geometry.camera_lateral_position,
        geometry.camera_before_forward_position,
        geometry.camera_after_forward_position,
        geometry.camera_height,
        geometry.field_of_view_degrees,
    )
    if observed_sampled_values != expected_sampled_values:
        raise DatasetValidationError(
            "sampled corridor geometry differs from deterministic configuration"
        )
    expected_positions = {
        "corridor_floor": (0.0, geometry.length / 2.0, -0.05),
        "corridor_left_surface": (
            -geometry.width / 2.0,
            geometry.length / 2.0,
            geometry.wall_height / 2.0,
        ),
        "corridor_right_surface": (
            geometry.width / 2.0,
            geometry.length / 2.0,
            geometry.wall_height / 2.0,
        ),
        "corridor_end_surface": (0.0, geometry.length, geometry.wall_height / 2.0),
    }
    if set(instrumentation.apparatus_surface_names) != set(CORRIDOR_SURFACE_NAMES):
        raise DatasetValidationError("corridor apparatus surface membership is inconsistent")
    for name, expected_position in expected_positions.items():
        if not np.allclose(
            instrumentation.raw_geom_world_positions[name],
            expected_position,
            atol=1e-12,
            rtol=0.0,
        ):
            raise DatasetValidationError(
                "corridor raw geometry does not match sampled privileged geometry"
            )


def _require_corridor_raw_segmentation(
    root: Path,
    transition: TransitionRecord,
    instrumentation: CorridorInstrumentation,
    segmentations: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    registry: _ArtifactRegistry,
) -> None:
    labels = {surface.surface_id: surface.segmentation_label for surface in transition.surfaces}
    raw_ids = set(instrumentation.raw_geom_ids.values())
    observed_raw_ids: set[int] = set()
    evidence_by_frame = sorted(
        instrumentation.raw_segmentation_frames,
        key=lambda item: item.frame_index,
    )
    for frame_index, public_segmentation in enumerate(segmentations):
        raw_segmentation = _load_npy(
            root,
            evidence_by_frame[frame_index].raw_segmentation,
            registry,
        )
        if raw_segmentation.dtype != np.dtype("int32") or (
            raw_segmentation.shape != public_segmentation.shape
        ):
            raise DatasetValidationError("corridor raw segmentation dtype or shape is invalid")
        frame_raw_ids = {int(value) for value in np.unique(raw_segmentation)}
        if not frame_raw_ids.issubset(raw_ids | {-1}):
            raise DatasetValidationError("corridor raw segmentation contains an unknown raw ID")
        observed_raw_ids.update(frame_raw_ids - {-1})
        reconstructed = np.zeros(raw_segmentation.shape, dtype=np.int32)
        for raw_id_text, opaque_id in instrumentation.raw_to_opaque_surface_ids.items():
            reconstructed[raw_segmentation == int(raw_id_text)] = labels[opaque_id]
        if not np.array_equal(reconstructed, public_segmentation):
            raise DatasetValidationError(
                "corridor public segmentation does not match the raw-to-opaque mapping"
            )
    if observed_raw_ids != raw_ids:
        raise DatasetValidationError("corridor raw segmentation does not contain every surface")


def _require_occlusion_oracle(
    root: Path,
    transition: TransitionRecord,
    instrumentation: SingleOccluderInstrumentation,
    segmentations: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    registry: _ArtifactRegistry,
) -> None:
    oracle = instrumentation.occlusion_oracle
    raw_ids = set(instrumentation.raw_geom_ids.values())
    if oracle.candidate_occluder_raw_geom_id not in raw_ids:
        raise DatasetValidationError("occlusion oracle names an unknown occluder raw ID")
    if oracle.candidate_occluded_raw_geom_id not in raw_ids:
        raise DatasetValidationError("occlusion oracle names an unknown occluded raw ID")
    if oracle.candidate_occluder_raw_geom_id != instrumentation.raw_geom_ids.get(
        "occluding_surface"
    ):
        raise DatasetValidationError("occlusion oracle candidate is not the apparatus occluder")
    if oracle.candidate_occluded_raw_geom_id != instrumentation.raw_geom_ids.get(
        "background_surface"
    ):
        raise DatasetValidationError("occlusion oracle candidate is not the apparatus background")

    mapping = instrumentation.raw_to_opaque_surface_ids
    occluder_id = mapping.get(str(oracle.candidate_occluder_raw_geom_id))
    occluded_id = mapping.get(str(oracle.candidate_occluded_raw_geom_id))
    if occluder_id is None or occluded_id is None:
        raise DatasetValidationError("occlusion candidates lack opaque surface mappings")
    labels = {surface.surface_id: surface.segmentation_label for surface in transition.surfaces}
    if occluder_id not in labels or occluded_id not in labels:
        raise DatasetValidationError("occlusion candidates map outside declared surfaces")
    opaque_to_raw = {opaque_id: int(raw_id) for raw_id, opaque_id in mapping.items()}
    if set(opaque_to_raw) != set(labels) or len(opaque_to_raw) != len(mapping):
        raise DatasetValidationError("apparatus raw-to-opaque mapping is not exact and bijective")

    evidence_by_frame = sorted(oracle.frames, key=lambda item: item.frame_index)
    supported_frames: list[int] = []
    for frame_index, ordinary_segmentation in enumerate(segmentations):
        evidence = evidence_by_frame[frame_index]
        counterfactual = _load_npy(
            root,
            evidence.counterfactual_segmentation,
            registry,
        )
        if (
            counterfactual.dtype != np.dtype("int32")
            or counterfactual.shape != ordinary_segmentation.shape
        ):
            raise DatasetValidationError("counterfactual segmentation dtype or shape is invalid")
        observed_raw_ids = {int(value) for value in np.unique(counterfactual)}
        if not observed_raw_ids.issubset(raw_ids | {-1}):
            raise DatasetValidationError("counterfactual segmentation contains an unknown raw ID")
        ordinary_raw = np.full(ordinary_segmentation.shape, -1, dtype=np.int32)
        for opaque_id, raw_id in opaque_to_raw.items():
            ordinary_raw[ordinary_segmentation == labels[opaque_id]] = raw_id
        if np.any(counterfactual == oracle.candidate_occluder_raw_geom_id):
            raise DatasetValidationError(
                "counterfactual segmentation still contains the excluded occluder"
            )
        ordinary_occluder_mask = ordinary_raw == oracle.candidate_occluder_raw_geom_id
        changed_mask = counterfactual != ordinary_raw
        if not np.array_equal(changed_mask, ordinary_occluder_mask):
            raise DatasetValidationError(
                "counterfactual changes are not confined exactly to the occluder footprint"
            )
        reveal_mask = ordinary_occluder_mask & (
            counterfactual == oracle.candidate_occluded_raw_geom_id
        )
        count = int(np.count_nonzero(reveal_mask))
        if count != evidence.revealed_pixel_count:
            raise DatasetValidationError("counterfactual revealed-pixel count is inconsistent")
        if logical_array_hash(reveal_mask) != evidence.reveal_mask_logical_sha256:
            raise DatasetValidationError("counterfactual reveal-mask hash is inconsistent")
        if count > 0:
            supported_frames.append(frame_index)

    expected_relations = (
        OcclusionRelation(
            occluder_surface_id=occluder_id,
            occluded_surface_id=occluded_id,
            frame_indices=tuple(supported_frames),  # type: ignore[arg-type]
        ),
    )
    if not supported_frames or transition.occlusion_relations != expected_relations:
        raise DatasetValidationError("occlusion relations do not match counterfactual evidence")


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
    source_provenance_sha256 = compute_source_provenance_hash(manifest.source_provenance)
    if source_provenance_sha256 != manifest.source_provenance_sha256:
        raise DatasetValidationError("source provenance hash mismatch")
    renderer_execution_provenance_sha256 = compute_renderer_execution_provenance_hash(
        manifest.renderer_provenance
    )
    if renderer_execution_provenance_sha256 != manifest.renderer_execution_provenance_sha256:
        raise DatasetValidationError("renderer/execution provenance hash mismatch")
    if (
        compute_content_provenance_binding(
            manifest.dataset_logical_sha256,
            manifest.source_provenance_sha256,
            manifest.renderer_execution_provenance_sha256,
        )
        != manifest.content_provenance_binding_sha256
    ):
        raise DatasetValidationError("content/provenance binding hash mismatch")

    registry = _ArtifactRegistry(resolved_root)

    config_payload = _verify_json(resolved_root, manifest.resolved_config, registry)
    try:
        config = parse_config(config_payload)
    except Exception as error:
        raise DatasetValidationError("resolved configuration failed schema validation") from error
    if sha256_bytes(canonical_json_bytes(config)) != manifest.config_logical_sha256:
        raise DatasetValidationError("resolved configuration hash mismatch")
    if config.seed != manifest.root_seed:
        raise DatasetValidationError("manifest seed does not match resolved configuration")
    if config.appearance.variant != manifest.appearance_variant:
        raise DatasetValidationError("appearance variant is inconsistent")
    if config.scene_family != manifest.scene_family:
        raise DatasetValidationError("manifest scene family differs from resolved configuration")

    dataset_surface_ids: set[str] = set()
    for episode in manifest.episodes:
        if episode.episode_id != f"episode-{episode.episode_index:06d}":
            raise DatasetValidationError("episode identifier does not match its index")
        if episode.episode_seed != derive_seed(config.seed, f"episode:{episode.episode_index}"):
            raise DatasetValidationError("episode derived seed mismatch")
        transition_payload = _verify_json(resolved_root, episode.transition, registry)
        try:
            transition = TransitionRecord.model_validate_json(
                canonical_json_bytes(transition_payload)
            )
        except Exception as error:
            raise DatasetValidationError("transition failed schema validation") from error
        if transition.episode_id != episode.episode_id:
            raise DatasetValidationError("episode and transition identifiers differ")
        if compute_ecological_label_hash(transition) != transition.ecological_label_sha256:
            raise DatasetValidationError("ecological-label hash mismatch")
        if transition.ecological_label_sha256 != episode.ecological_label_sha256:
            raise DatasetValidationError("episode ecological-label hash mismatch")

        instrumentation_payload = _verify_json(
            resolved_root,
            episode.privileged_instrumentation,
            registry,
        )
        try:
            instrumentation = parse_privileged_instrumentation_json(
                canonical_json_bytes(instrumentation_payload)
            )
        except Exception as error:
            raise DatasetValidationError("instrumentation failed schema validation") from error
        if instrumentation.episode_id != episode.episode_id:
            raise DatasetValidationError("instrumentation episode identifier mismatch")
        if instrumentation.appearance_variant != manifest.appearance_variant:
            raise DatasetValidationError("instrumentation appearance variant mismatch")
        if instrumentation.scene_family != manifest.scene_family:
            raise DatasetValidationError("instrumentation scene family mismatch")
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
        ordinary_transition_bytes = canonical_json_bytes(transition_payload)
        forbidden_control_or_semantic_tokens = (
            b"scene_family",
            b"sampled_geometry",
            *(name.encode("utf-8") for name in instrumentation.raw_geom_ids),
        )
        if any(
            token in ordinary_transition_bytes for token in forbidden_control_or_semantic_tokens
        ):
            raise DatasetValidationError(
                "ordinary ecological transition leaks scene control or semantic apparatus names"
            )
        episode_surface_ids = {surface.surface_id for surface in transition.surfaces}
        duplicate_surface_ids = dataset_surface_ids & episode_surface_ids
        if duplicate_surface_ids:
            raise DatasetValidationError("surface identifiers must be disjoint across episodes")
        dataset_surface_ids.update(episode_surface_ids)

        if isinstance(instrumentation, SingleOccluderInstrumentation):
            expected_scene_content_sha256 = _single_occluder_scene_content_hash(
                episode.episode_seed
            )
        elif isinstance(instrumentation, CorridorInstrumentation):
            if not isinstance(config, CorridorConfig):
                raise DatasetValidationError("corridor instrumentation/configuration mismatch")
            _require_corridor_instrumentation(config, episode.episode_seed, instrumentation)
            expected_scene_content_sha256 = _corridor_scene_content_hash(
                episode.episode_seed,
                instrumentation,
            )
        else:
            raise DatasetValidationError("unsupported scene instrumentation")
        if episode.scene_content_sha256 != expected_scene_content_sha256:
            raise DatasetValidationError("episode scene-content hash mismatch")

        frame_arrays: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = []
        cameras: list[CameraInstrumentation] = []
        for frame in (transition.before, transition.after):
            rgb = _load_rgb(resolved_root, frame.rgb, registry)
            depth = _load_npy(resolved_root, frame.depth, registry)
            segmentation = _load_npy(resolved_root, frame.segmentation, registry)
            if rgb.dtype != np.dtype("uint8") or rgb.shape != (
                frame.height,
                frame.width,
                3,
            ):
                raise DatasetValidationError("RGB dtype or shape is invalid")
            if depth.dtype != np.dtype("float32") or depth.shape != (
                frame.height,
                frame.width,
            ):
                raise DatasetValidationError("depth dtype or shape is invalid")
            if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
                raise DatasetValidationError("depth values must be finite and non-negative")
            if segmentation.dtype != np.dtype("int32") or segmentation.shape != (
                frame.height,
                frame.width,
            ):
                raise DatasetValidationError("segmentation dtype or shape is invalid")
            if rgb.shape[:2] != depth.shape or depth.shape != segmentation.shape:
                raise DatasetValidationError("RGB, depth, and segmentation are misaligned")
            camera_payload = _verify_json(
                resolved_root,
                frame.camera_world_transform,
                registry,
            )
            try:
                camera = CameraInstrumentation.model_validate_json(
                    canonical_json_bytes(camera_payload)
                )
            except Exception as error:
                raise DatasetValidationError("camera failed schema validation") from error
            if camera.frame_index != frame.frame_index:
                raise DatasetValidationError("camera instrumentation frame mismatch")
            cameras.append(camera)
            frame_arrays.append((rgb, depth, segmentation))

        _require_camera_action_alignment(
            config,
            transition,
            (cameras[0], cameras[1]),
            instrumentation,
        )

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
        if isinstance(instrumentation, CorridorInstrumentation) and any(
            np.any(segmentation == 0) for _, _, segmentation in frame_arrays
        ):
            raise DatasetValidationError(
                "corridor optical field contains uncontrolled renderer background"
            )

        if isinstance(instrumentation, SingleOccluderInstrumentation):
            if not isinstance(config, SingleOccluderConfig):
                raise DatasetValidationError(
                    "single-occluder instrumentation/configuration mismatch"
                )
            _require_occlusion_oracle(
                resolved_root,
                transition,
                instrumentation,
                (frame_arrays[0][2], frame_arrays[1][2]),
                registry,
            )
        elif transition.occlusion_relations:
            raise DatasetValidationError(
                "corridor occlusion relations require controlled oracle evidence"
            )
        else:
            _require_corridor_raw_segmentation(
                resolved_root,
                transition,
                instrumentation,
                (frame_arrays[0][2], frame_arrays[1][2]),
                registry,
            )

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
