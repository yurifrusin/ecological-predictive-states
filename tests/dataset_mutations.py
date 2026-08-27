"""Hash-consistent dataset mutations used to exercise independent validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from epsbench.data.identity import (
    compute_analytic_transport_hash,
    compute_content_provenance_binding,
    compute_dataset_logical_hash,
    compute_ecological_label_hash,
    compute_oriented_boundary_hash,
    compute_visibility_event_hash,
)
from epsbench.schema import (
    ArtifactRecord,
    AvailableDenseOpticalTransport,
    DatasetManifest,
    PrivilegedInstrumentation,
    TransitionRecord,
    parse_privileged_instrumentation_json,
)
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    sha256_bytes,
    sha256_file,
    write_canonical_json,
)


def load_manifest(root: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json((root / "manifest.json").read_text(encoding="utf-8"))


def load_episode_payloads(
    root: Path,
    episode_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(root)
    episode = manifest.episodes[episode_index]
    transition = json.loads((root / episode.transition.path).read_text(encoding="utf-8"))
    instrumentation = json.loads(
        (root / episode.privileged_instrumentation.path).read_text(encoding="utf-8")
    )
    assert isinstance(transition, dict)
    assert isinstance(instrumentation, dict)
    return transition, instrumentation


def rewrite_json_artifact(
    root: Path,
    record: ArtifactRecord,
    payload: dict[str, Any],
) -> ArtifactRecord:
    path = root / record.path
    write_canonical_json(path, payload)
    logical_bytes = canonical_json_bytes(payload)
    return ArtifactRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "shape": (len(logical_bytes),),
            "logical_sha256": sha256_bytes(logical_bytes),
            "file_sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
    )


def rewrite_array_artifact(
    root: Path,
    record: ArtifactRecord,
    array: np.ndarray[Any, Any],
) -> ArtifactRecord:
    path = root / record.path
    np.save(path, array, allow_pickle=False)
    return ArtifactRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "dtype": str(array.dtype),
            "shape": tuple(array.shape),
            "logical_sha256": logical_array_hash(array),
            "file_sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
    )


def commit_episode_payloads(
    root: Path,
    episode_index: int,
    transition_payload: dict[str, Any],
    instrumentation_payload: dict[str, Any],
) -> None:
    """Write valid episode schemas and rebuild every affected identity link."""

    manifest = load_manifest(root)
    episode = manifest.episodes[episode_index]
    events_payload = transition_payload.get("ecological_visibility_events")
    if isinstance(events_payload, dict):
        summaries = events_payload.get("occluding_event_summaries")
        if isinstance(summaries, list):
            summaries.sort(
                key=lambda item: (
                    item["kind"],
                    item["affected_surface_id"],
                    item["owner_surface_id"],
                )
            )
        whole = events_payload.get("whole_surface_events")
        if isinstance(whole, list):
            whole.sort(key=lambda item: item["surface_id"])
    transition = TransitionRecord.model_validate_json(canonical_json_bytes(transition_payload))
    if isinstance(transition.analytic_optical_transport, AvailableDenseOpticalTransport):
        analytic_transport = transition.analytic_optical_transport.model_copy(
            update={"analytic_transport_sha256": "0" * 64}
        )
        analytic_transport = analytic_transport.model_copy(
            update={
                "analytic_transport_sha256": compute_analytic_transport_hash(analytic_transport)
            }
        )
        transition = transition.model_copy(
            update={"analytic_optical_transport": analytic_transport}
        )
    assert isinstance(transition.analytic_optical_transport, AvailableDenseOpticalTransport)
    boundary = transition.oriented_boundary_ownership.model_copy(
        update={"oriented_boundary_sha256": "0" * 64}
    )
    boundary = boundary.model_copy(
        update={"oriented_boundary_sha256": compute_oriented_boundary_hash(boundary)}
    )
    events = transition.ecological_visibility_events.model_copy(
        update={
            "analytic_transport_sha256": (
                transition.analytic_optical_transport.analytic_transport_sha256
            ),
            "oriented_boundary_sha256": boundary.oriented_boundary_sha256,
            "visibility_event_sha256": "0" * 64,
        }
    )
    events = events.model_copy(
        update={"visibility_event_sha256": compute_visibility_event_hash(events)}
    )
    transition = transition.model_copy(
        update={
            "oriented_boundary_ownership": boundary,
            "ecological_visibility_events": events,
        }
    )
    transition = TransitionRecord.model_validate(
        {
            **transition.model_dump(mode="python"),
            "ecological_label_sha256": compute_ecological_label_hash(transition),
        }
    )
    instrumentation: PrivilegedInstrumentation = parse_privileged_instrumentation_json(
        canonical_json_bytes(instrumentation_payload)
    )
    transition_record = rewrite_json_artifact(
        root,
        episode.transition,
        transition.model_dump(mode="json"),
    )
    instrumentation_record = rewrite_json_artifact(
        root,
        episode.privileged_instrumentation,
        instrumentation.model_dump(mode="json"),
    )
    episodes = list(manifest.episodes)
    episodes[episode_index] = episode.model_copy(
        update={
            "transition": transition_record,
            "privileged_instrumentation": instrumentation_record,
            "ecological_label_sha256": transition.ecological_label_sha256,
            "analytic_transport_sha256": (
                transition.analytic_optical_transport.analytic_transport_sha256
                if isinstance(
                    transition.analytic_optical_transport,
                    AvailableDenseOpticalTransport,
                )
                else episode.analytic_transport_sha256
            ),
            "oriented_boundary_sha256": (
                transition.oriented_boundary_ownership.oriented_boundary_sha256
            ),
            "visibility_event_sha256": (
                transition.ecological_visibility_events.visibility_event_sha256
            ),
            "rgb_logical_sha256": (
                transition.before.rgb.logical_sha256,
                transition.after.rgb.logical_sha256,
            ),
        }
    )
    _write_manifest(root, manifest, episodes)


def commit_raw_transition_payload(
    root: Path,
    episode_index: int,
    transition_payload: dict[str, Any],
) -> None:
    """Rebuild container hashes while intentionally retaining an invalid transition schema."""

    manifest = load_manifest(root)
    episode = manifest.episodes[episode_index]
    transition_record = rewrite_json_artifact(root, episode.transition, transition_payload)
    episodes = list(manifest.episodes)
    episodes[episode_index] = episode.model_copy(update={"transition": transition_record})
    _write_manifest(root, manifest, episodes)


def commit_declared_transport_identity_corruption(
    root: Path,
    episode_index: int,
    corrupted_identity: str,
) -> None:
    """Rebuild every enclosing hash while preserving a false analytic identity."""

    manifest = load_manifest(root)
    episode = manifest.episodes[episode_index]
    transition_payload = json.loads((root / episode.transition.path).read_text(encoding="utf-8"))
    transition_payload["analytic_optical_transport"]["analytic_transport_sha256"] = (
        corrupted_identity
    )
    transition_payload["ecological_visibility_events"]["analytic_transport_sha256"] = (
        corrupted_identity
    )
    transition = TransitionRecord.model_validate_json(canonical_json_bytes(transition_payload))
    events = transition.ecological_visibility_events.model_copy(
        update={
            "analytic_transport_sha256": corrupted_identity,
            "visibility_event_sha256": "0" * 64,
        }
    )
    events = events.model_copy(
        update={"visibility_event_sha256": compute_visibility_event_hash(events)}
    )
    transition = transition.model_copy(update={"ecological_visibility_events": events})
    transition = transition.model_copy(
        update={"ecological_label_sha256": compute_ecological_label_hash(transition)}
    )
    transition_record = rewrite_json_artifact(
        root,
        episode.transition,
        transition.model_dump(mode="json"),
    )
    episodes = list(manifest.episodes)
    episodes[episode_index] = episode.model_copy(
        update={
            "transition": transition_record,
            "ecological_label_sha256": transition.ecological_label_sha256,
            "analytic_transport_sha256": corrupted_identity,
            "visibility_event_sha256": events.visibility_event_sha256,
        }
    )
    _write_manifest(root, manifest, episodes)


def commit_declared_boundary_identity_corruption(
    root: Path,
    episode_index: int,
    corrupted_identity: str,
) -> None:
    """Rebuild every enclosing hash while preserving a false boundary identity."""

    manifest = load_manifest(root)
    episode = manifest.episodes[episode_index]
    transition = TransitionRecord.model_validate_json(
        (root / episode.transition.path).read_text(encoding="utf-8")
    )
    boundary = transition.oriented_boundary_ownership.model_copy(
        update={"oriented_boundary_sha256": corrupted_identity}
    )
    events = transition.ecological_visibility_events.model_copy(
        update={
            "oriented_boundary_sha256": corrupted_identity,
            "visibility_event_sha256": "0" * 64,
        }
    )
    events = events.model_copy(
        update={"visibility_event_sha256": compute_visibility_event_hash(events)}
    )
    transition = transition.model_copy(
        update={
            "oriented_boundary_ownership": boundary,
            "ecological_visibility_events": events,
        }
    )
    transition = transition.model_copy(
        update={"ecological_label_sha256": compute_ecological_label_hash(transition)}
    )
    transition_record = rewrite_json_artifact(
        root,
        episode.transition,
        transition.model_dump(mode="json"),
    )
    episodes = list(manifest.episodes)
    episodes[episode_index] = episode.model_copy(
        update={
            "transition": transition_record,
            "ecological_label_sha256": transition.ecological_label_sha256,
            "oriented_boundary_sha256": corrupted_identity,
            "visibility_event_sha256": events.visibility_event_sha256,
        }
    )
    _write_manifest(root, manifest, episodes)


def commit_declared_event_identity_corruption(
    root: Path,
    episode_index: int,
    corrupted_identity: str,
) -> None:
    """Rebuild every enclosing hash while preserving a false event identity."""

    manifest = load_manifest(root)
    episode = manifest.episodes[episode_index]
    transition = TransitionRecord.model_validate_json(
        (root / episode.transition.path).read_text(encoding="utf-8")
    )
    events = transition.ecological_visibility_events.model_copy(
        update={"visibility_event_sha256": corrupted_identity}
    )
    transition = transition.model_copy(update={"ecological_visibility_events": events})
    transition = transition.model_copy(
        update={"ecological_label_sha256": compute_ecological_label_hash(transition)}
    )
    transition_record = rewrite_json_artifact(
        root,
        episode.transition,
        transition.model_dump(mode="json"),
    )
    episodes = list(manifest.episodes)
    episodes[episode_index] = episode.model_copy(
        update={
            "transition": transition_record,
            "ecological_label_sha256": transition.ecological_label_sha256,
            "visibility_event_sha256": corrupted_identity,
        }
    )
    _write_manifest(root, manifest, episodes)


def commit_raw_instrumentation_payload(
    root: Path,
    episode_index: int,
    instrumentation_payload: dict[str, Any],
) -> None:
    """Rebuild hashes while intentionally retaining an invalid instrumentation schema."""

    manifest = load_manifest(root)
    episode = manifest.episodes[episode_index]
    instrumentation_record = rewrite_json_artifact(
        root,
        episode.privileged_instrumentation,
        instrumentation_payload,
    )
    episodes = list(manifest.episodes)
    episodes[episode_index] = episode.model_copy(
        update={"privileged_instrumentation": instrumentation_record}
    )
    _write_manifest(root, manifest, episodes)


def commit_resolved_config_payload(
    root: Path,
    config_payload: dict[str, Any],
) -> None:
    """Rewrite resolved configuration and every manifest identity that depends on it."""

    manifest = load_manifest(root)
    resolved_config = rewrite_json_artifact(
        root,
        manifest.resolved_config,
        config_payload,
    )
    changed = manifest.model_copy(
        update={
            "resolved_config": resolved_config,
            "config_logical_sha256": sha256_bytes(canonical_json_bytes(config_payload)),
        }
    )
    _write_manifest(root, changed, list(changed.episodes))


def _write_manifest(
    root: Path,
    manifest: DatasetManifest,
    episodes: list[Any],
) -> None:
    provisional = DatasetManifest.model_validate(
        {**manifest.model_dump(mode="python"), "episodes": tuple(episodes)}
    )
    dataset_hash = compute_dataset_logical_hash(provisional)
    updated = DatasetManifest.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "dataset_logical_sha256": dataset_hash,
            "content_provenance_binding_sha256": compute_content_provenance_binding(
                dataset_hash,
                provisional.source_provenance_sha256,
                provisional.renderer_execution_provenance_sha256,
            ),
        }
    )
    write_canonical_json(root / "manifest.json", updated)


def replace_string(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: replace_string(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_string(item, old, new) for item in value]
    if value == old:
        return new
    return value
