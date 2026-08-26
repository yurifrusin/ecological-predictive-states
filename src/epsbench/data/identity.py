"""Explicit logical identity domains for annotations and datasets."""

from typing import Any

from epsbench.config import CorridorConfig, SingleOccluderConfig
from epsbench.schema import (
    AvailableDenseOpticalTransport,
    CorridorSampledGeometry,
    DatasetManifest,
    RendererProvenance,
    SourceProvenance,
    TransitionRecord,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


def ecological_label_domain(transition: TransitionRecord) -> dict[str, Any]:
    """Return only appearance-invariant ecological labels and executed action."""

    return {
        "schema_version": transition.schema_version,
        "action": transition.action.model_dump(mode="json"),
        "surfaces": [item.model_dump(mode="json") for item in transition.surfaces],
        "segmentation_logical_sha256": [
            transition.before.segmentation.logical_sha256,
            transition.after.segmentation.logical_sha256,
        ],
        "visibility_states": [
            item.model_dump(mode="json") for item in transition.visibility_states
        ],
        "region_correspondence": [
            item.model_dump(mode="json") for item in transition.region_correspondence
        ],
        "region_mask_changes": [
            item.model_dump(mode="json") for item in transition.region_mask_changes
        ],
        "ecological_visibility_events": transition.ecological_visibility_events.model_dump(
            mode="json"
        ),
        "occlusion": transition.occlusion.model_dump(mode="json"),
        "boundary_structures": [
            item.model_dump(mode="json") for item in transition.boundary_structures
        ],
        "analytic_optical_transport": (
            {
                **analytic_transport_domain(transition.analytic_optical_transport),
                "analytic_transport_sha256": (
                    transition.analytic_optical_transport.analytic_transport_sha256
                ),
            }
            if isinstance(
                transition.analytic_optical_transport,
                AvailableDenseOpticalTransport,
            )
            else transition.analytic_optical_transport.model_dump(mode="json")
        ),
    }


def compute_ecological_label_hash(transition: TransitionRecord) -> str:
    return sha256_bytes(canonical_json_bytes(ecological_label_domain(transition)))


def analytic_transport_domain(transport: AvailableDenseOpticalTransport) -> dict[str, Any]:
    """Return the backend-independent optical-transport logical identity domain."""

    def direction_domain(direction: Any) -> dict[str, Any]:
        return {
            "source_frame_index": direction.source_frame_index,
            "target_frame_index": direction.target_frame_index,
            "vectors_fixed_logical_sha256": direction.vectors_fixed.logical_sha256,
            "validity_logical_sha256": direction.validity.logical_sha256,
            "reasons_logical_sha256": direction.reasons.logical_sha256,
        }

    return {
        "method": transport.method,
        "coordinate_convention": transport.coordinate_convention.model_dump(mode="json"),
        "quantisation": transport.quantisation.model_dump(mode="json"),
        "intersection_visibility": transport.intersection_visibility.model_dump(mode="json"),
        "boundary_ambiguity": transport.boundary_ambiguity.model_dump(mode="json"),
        "reason_code_domain": transport.reason_code_domain,
        "forward": direction_domain(transport.forward),
        "backward": direction_domain(transport.backward),
    }


def compute_analytic_transport_hash(transport: AvailableDenseOpticalTransport) -> str:
    return sha256_bytes(canonical_json_bytes(analytic_transport_domain(transport)))


def single_occluder_scene_content_domain(config: SingleOccluderConfig) -> dict[str, Any]:
    """Describe fixed apparatus and trajectory content without appearance or seed history."""

    return {
        "scene_family": config.scene_family,
        "apparatus_version": "single_occluder_v1",
        "surfaces": {
            "support_surface": {
                "type": "plane",
                "position": [0.0, 0.0, 0.0],
                "size": [4.0, 7.0, 0.1],
            },
            "background_surface": {
                "type": "box",
                "position": [0.0, 2.5, 1.05],
                "half_size": [2.2, 0.05, 1.05],
            },
            "occluding_surface": {
                "type": "box",
                "position": [0.0, 0.8, 0.9],
                "half_size": [0.55, 0.05, 0.9],
            },
        },
        "camera": {
            "before_position": [
                config.camera.before_lateral,
                config.camera.forward,
                config.camera.height,
            ],
            "after_position": [
                config.camera.after_lateral,
                config.camera.forward,
                config.camera.height,
            ],
            "orientation_rule": "xyaxes_1_0_0_0_0.16_1",
            "field_of_view_degrees": config.camera.field_of_view_degrees,
        },
        "action": config.action.model_dump(mode="json"),
    }


def corridor_scene_content_domain(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
) -> dict[str, Any]:
    """Describe sampled corridor layout and trajectory without appearance or seed history."""

    return {
        "scene_family": config.scene_family,
        "apparatus_version": "corridor_v1",
        "surfaces": {
            "surface_names": [
                "corridor_floor",
                "corridor_left_surface",
                "corridor_right_surface",
                "corridor_end_surface",
            ],
            "width": geometry.width,
            "length": geometry.length,
            "wall_height": geometry.wall_height,
            "wall_thickness": 0.05,
            "floor_thickness": 0.05,
        },
        "camera": {
            "before_position": [
                geometry.camera_lateral_position,
                geometry.camera_before_forward_position,
                geometry.camera_height,
            ],
            "after_position": [
                geometry.camera_lateral_position,
                geometry.camera_after_forward_position,
                geometry.camera_height,
            ],
            "orientation_rule": "xyaxes_1_0_0_0_0_1",
            "field_of_view_degrees": geometry.field_of_view_degrees,
        },
        "action": config.action.model_dump(mode="json"),
    }


def compute_single_occluder_scene_content_hash(config: SingleOccluderConfig) -> str:
    return sha256_bytes(canonical_json_bytes(single_occluder_scene_content_domain(config)))


def compute_corridor_scene_content_hash(
    config: CorridorConfig,
    geometry: CorridorSampledGeometry,
) -> str:
    return sha256_bytes(canonical_json_bytes(corridor_scene_content_domain(config, geometry)))


def dataset_logical_domain(manifest: DatasetManifest) -> dict[str, Any]:
    """Return the non-volatile logical dataset identity domain."""

    return {
        "schema_version": manifest.schema_version,
        "generator_version": manifest.generator_version,
        "scene_family": manifest.scene_family,
        "root_seed": manifest.root_seed,
        "config_logical_sha256": manifest.config_logical_sha256,
        "appearance_variant": manifest.appearance_variant,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "episode_index": episode.episode_index,
                "episode_seed": episode.episode_seed,
                "scene_content_sha256": episode.scene_content_sha256,
                "ecological_label_sha256": episode.ecological_label_sha256,
                "analytic_transport_sha256": episode.analytic_transport_sha256,
                "rgb_logical_sha256": list(episode.rgb_logical_sha256),
            }
            for episode in manifest.episodes
        ],
    }


def compute_dataset_logical_hash(manifest: DatasetManifest) -> str:
    return sha256_bytes(canonical_json_bytes(dataset_logical_domain(manifest)))


def compute_source_provenance_hash(provenance: SourceProvenance) -> str:
    """Hash source provenance independently of scientific content identity."""

    return sha256_bytes(canonical_json_bytes(provenance))


def compute_renderer_execution_provenance_hash(provenance: RendererProvenance) -> str:
    """Hash stable renderer and execution-environment facts outside content identity."""

    return sha256_bytes(canonical_json_bytes(provenance))


def compute_content_provenance_binding(
    dataset_logical_sha256: str,
    source_provenance_sha256: str,
    renderer_execution_provenance_sha256: str,
) -> str:
    """Bind content, source, and renderer/execution identities without recursion."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "dataset_logical_sha256": dataset_logical_sha256,
                "renderer_execution_provenance_sha256": (renderer_execution_provenance_sha256),
                "source_provenance_sha256": source_provenance_sha256,
            }
        )
    )
