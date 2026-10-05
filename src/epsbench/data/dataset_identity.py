"""Canonical dataset identity domains without native annotation imports."""

from typing import Any

from epsbench.schema import (
    CanonicalPairedEpisodeManifest,
    DatasetManifest,
    RendererProvenance,
    SourceProvenance,
)
from epsbench.utils.canonical import canonical_json_bytes, sha256_bytes


def dataset_logical_domain(manifest: DatasetManifest) -> dict[str, Any]:
    """Return the non-volatile logical dataset identity domain."""

    return {
        "schema_version": manifest.schema_version,
        "generator_version": manifest.generator_version,
        "scene_family": manifest.scene_family,
        "root_seed": manifest.root_seed,
        "config_logical_sha256": manifest.config_logical_sha256,
        "appearance_registry_sha256": manifest.appearance_registry_sha256,
        "appearance_profile_id": manifest.appearance_profile_id,
        "appearance_profile_sha256": manifest.appearance_profile_sha256,
        "evaluation_seed_registry_sha256": manifest.evaluation_seed_registry_sha256,
        "appearance_assignment_schedule_source": (manifest.appearance_assignment_schedule_source),
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "episode_index": episode.episode_index,
                "episode_seed": episode.episode_seed,
                "scene_content_sha256": episode.scene_content_sha256,
                "ecological_label_sha256": episode.ecological_label_sha256,
                "analytic_transport_sha256": episode.analytic_transport_sha256,
                "oriented_boundary_sha256": episode.oriented_boundary_sha256,
                "visibility_event_sha256": episode.visibility_event_sha256,
                "appearance_instance_sha256": episode.appearance_instance_sha256,
                "rgb_logical_sha256": list(episode.rgb_logical_sha256),
                **(
                    {
                        "paired_output_provenance_sha256": list(
                            episode.paired_output_provenance_sha256
                        )
                    }
                    if isinstance(episode, CanonicalPairedEpisodeManifest)
                    else {}
                ),
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
