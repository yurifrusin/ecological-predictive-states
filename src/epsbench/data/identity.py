"""Explicit logical identity domains for annotations and datasets."""

from typing import Any

from epsbench.schema import DatasetManifest, SourceProvenance, TransitionRecord
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
        "visibility_events": [
            item.model_dump(mode="json") for item in transition.visibility_events
        ],
        "occlusion_relations": [
            item.model_dump(mode="json") for item in transition.occlusion_relations
        ],
        "boundary_structures": [
            item.model_dump(mode="json") for item in transition.boundary_structures
        ],
        "dense_optical_flow": transition.dense_optical_flow.model_dump(mode="json"),
    }


def compute_ecological_label_hash(transition: TransitionRecord) -> str:
    return sha256_bytes(canonical_json_bytes(ecological_label_domain(transition)))


def dataset_logical_domain(manifest: DatasetManifest) -> dict[str, Any]:
    """Return the non-volatile logical dataset identity domain."""

    return {
        "schema_version": manifest.schema_version,
        "generator_version": manifest.generator_version,
        "root_seed": manifest.root_seed,
        "config_logical_sha256": manifest.config_logical_sha256,
        "appearance_variant": manifest.appearance_variant,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "episode_index": episode.episode_index,
                "episode_seed": episode.episode_seed,
                "ecological_label_sha256": episode.ecological_label_sha256,
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


def compute_content_provenance_binding(
    dataset_logical_sha256: str,
    source_provenance_sha256: str,
) -> str:
    """Bind already-computed content and provenance identities without recursion."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "dataset_logical_sha256": dataset_logical_sha256,
                "source_provenance_sha256": source_provenance_sha256,
            }
        )
    )
