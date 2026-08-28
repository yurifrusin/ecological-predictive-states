from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.data import DatasetValidationError, validate_dataset
from epsbench.data.identity import (
    compute_content_provenance_binding,
    compute_dataset_logical_hash,
    compute_renderer_execution_provenance_hash,
)
from epsbench.schema import ArtifactRecord, RendererProvenance
from epsbench.utils.canonical import (
    canonical_json_bytes,
    logical_array_hash,
    write_canonical_json,
)
from tests.dataset_mutations import (
    commit_episode_payloads,
    commit_raw_instrumentation_payload,
    commit_raw_transition_payload,
    load_episode_payloads,
    load_manifest,
    replace_string,
    rewrite_array_artifact,
    rewrite_json_artifact,
)


def _copy_dataset(smoke_dataset: Path, tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(smoke_dataset, target)
    return target


def _artifact(payload: dict[str, Any]) -> ArtifactRecord:
    return ArtifactRecord.model_validate_json(canonical_json_bytes(payload))


@pytest.mark.parametrize(
    "corruption",
    ["reversed", "omitted", "invented", "wrong_frames"],
)
def test_occlusion_relation_corruption_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, f"occlusion-{corruption}")
    transition, instrumentation = load_episode_payloads(broken, 0)
    relation = transition["occlusion"]["relations"][0]
    if corruption == "reversed":
        relation["occluder_surface_id"], relation["occluded_surface_id"] = (
            relation["occluded_surface_id"],
            relation["occluder_surface_id"],
        )
    elif corruption == "omitted":
        transition["occlusion"]["relations"] = []
    elif corruption == "invented":
        existing = {
            (item["occluder_surface_id"], item["occluded_surface_id"])
            for item in transition["occlusion"]["relations"]
        }
        surface_ids = [surface["surface_id"] for surface in transition["surfaces"]]
        invented_owner, invented_affected = next(
            (owner, affected)
            for owner in surface_ids
            for affected in surface_ids
            if owner != affected and (owner, affected) not in existing
        )
        transition["occlusion"]["relations"].append(
            {
                "occluder_surface_id": invented_owner,
                "occluded_surface_id": invented_affected,
                "frame_indices": [0, 1],
            }
        )
        transition["occlusion"]["relations"].sort(
            key=lambda item: (item["occluder_surface_id"], item["occluded_surface_id"])
        )
    else:
        relation["frame_indices"] = [0]
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match="complete oriented-boundary"):
        validate_dataset(broken)


def test_relation_referring_to_frame_without_oracle_evidence_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "occlusion-no-frame-evidence")
    transition, instrumentation = load_episode_payloads(broken, 0)
    frame_evidence = instrumentation["occlusion_oracle"]["frames"][1]
    artifact = _artifact(frame_evidence["counterfactual_segmentation"])
    counterfactual = np.load(broken / artifact.path, allow_pickle=False)
    public_after = np.load(
        broken / transition["after"]["segmentation"]["path"],
        allow_pickle=False,
    )
    labels = {
        surface["surface_id"]: surface["segmentation_label"] for surface in transition["surfaces"]
    }
    ordinary_raw = np.full(counterfactual.shape, -1, dtype=np.int32)
    for raw_id, opaque_id in instrumentation["raw_to_opaque_surface_ids"].items():
        ordinary_raw[public_after == labels[opaque_id]] = int(raw_id)
    oracle = instrumentation["occlusion_oracle"]
    no_reveal = counterfactual.copy()
    reveal = (ordinary_raw == oracle["candidate_occluder_raw_geom_id"]) & (
        counterfactual == oracle["candidate_occluded_raw_geom_id"]
    )
    no_reveal[reveal] = -1
    updated_artifact = rewrite_array_artifact(broken, artifact, no_reveal)
    frame_evidence["counterfactual_segmentation"] = updated_artifact.model_dump(mode="json")
    frame_evidence["revealed_pixel_count"] = 0
    frame_evidence["reveal_mask_logical_sha256"] = logical_array_hash(
        np.zeros(no_reveal.shape, dtype=np.bool_)
    )
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match="counterfactual evidence disagrees"):
        validate_dataset(broken)


@pytest.mark.parametrize(
    "fabrication",
    ["occluder_present", "outside_footprint_change", "fabricated_reveal"],
)
def test_fully_hash_rebuilt_counterfactual_fabrication_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    fabrication: str,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, f"counterfactual-{fabrication}")
    transition, instrumentation = load_episode_payloads(broken, 0)
    oracle = instrumentation["occlusion_oracle"]
    frame_evidence = oracle["frames"][0]
    artifact = _artifact(frame_evidence["counterfactual_segmentation"])
    counterfactual = np.load(broken / artifact.path, allow_pickle=False)
    ordinary_segmentation = np.load(
        broken / transition["before"]["segmentation"]["path"],
        allow_pickle=False,
    )
    labels_by_surface = {
        surface["surface_id"]: surface["segmentation_label"] for surface in transition["surfaces"]
    }
    raw_by_surface = {
        surface_id: int(raw_id)
        for raw_id, surface_id in instrumentation["raw_to_opaque_surface_ids"].items()
    }
    ordinary_raw = np.full(ordinary_segmentation.shape, -1, dtype=np.int32)
    for surface_id, raw_id in raw_by_surface.items():
        ordinary_raw[ordinary_segmentation == labels_by_surface[surface_id]] = raw_id
    occluder_raw_id = oracle["candidate_occluder_raw_geom_id"]
    background_raw_id = oracle["candidate_occluded_raw_geom_id"]
    if fabrication == "occluder_present":
        row, column = np.argwhere(ordinary_raw == occluder_raw_id)[0]
        counterfactual[row, column] = occluder_raw_id
    elif fabrication == "outside_footprint_change":
        row, column = np.argwhere((ordinary_raw != occluder_raw_id) & (ordinary_raw != -1))[0]
        counterfactual[row, column] = -1
    else:
        row, column = np.argwhere(
            (ordinary_raw != occluder_raw_id) & (ordinary_raw != background_raw_id)
        )[0]
        counterfactual[row, column] = background_raw_id

    updated_artifact = rewrite_array_artifact(broken, artifact, counterfactual)
    frame_evidence["counterfactual_segmentation"] = updated_artifact.model_dump(mode="json")
    background_surface_id = instrumentation["raw_to_opaque_surface_ids"][str(background_raw_id)]
    claimed_reveal = (counterfactual == background_raw_id) & (
        ordinary_segmentation != labels_by_surface[background_surface_id]
    )
    frame_evidence["revealed_pixel_count"] = int(np.count_nonzero(claimed_reveal))
    frame_evidence["reveal_mask_logical_sha256"] = logical_array_hash(claimed_reveal)
    commit_episode_payloads(broken, 0, transition, instrumentation)

    expected_error = "excluded occluder" if fabrication == "occluder_present" else "footprint"
    with pytest.raises(DatasetValidationError, match=expected_error):
        validate_dataset(broken)


@pytest.mark.parametrize("corruption", ["extra_surface", "non_bijective"])
def test_hash_rebuilt_inexact_apparatus_mapping_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, f"apparatus-{corruption}")
    _, instrumentation = load_episode_payloads(broken, 0)
    if corruption == "extra_surface":
        instrumentation["raw_geom_ids"]["fabricated_surface"] = 99
        instrumentation["raw_geom_world_positions"]["fabricated_surface"] = [0.0, 0.0, 0.0]
        instrumentation["raw_to_opaque_surface_ids"]["99"] = next(
            iter(instrumentation["raw_to_opaque_surface_ids"].values())
        )
    else:
        mapping_values = list(instrumentation["raw_to_opaque_surface_ids"].values())
        first_key, second_key = tuple(instrumentation["raw_to_opaque_surface_ids"])[:2]
        instrumentation["raw_to_opaque_surface_ids"][first_key] = mapping_values[0]
        instrumentation["raw_to_opaque_surface_ids"][second_key] = mapping_values[0]
    commit_raw_instrumentation_payload(broken, 0, instrumentation)

    with pytest.raises(DatasetValidationError, match="instrumentation failed schema validation"):
        validate_dataset(broken)


@pytest.mark.parametrize("tamper", ["position", "rotation"])
def test_persisted_camera_tampering_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    tamper: str,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, f"camera-{tamper}")
    transition, instrumentation = load_episode_payloads(broken, 0)
    camera_record = _artifact(transition["after"]["camera_world_transform"])
    camera_path = broken / camera_record.path
    camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
    if tamper == "position":
        camera_payload["camera_world_position"][0] += 0.01
    else:
        current = np.asarray(camera_payload["camera_world_rotation_row_major"]).reshape(3, 3)
        quarter_turn = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        camera_payload["camera_world_rotation_row_major"] = (
            (quarter_turn @ current).ravel().tolist()
        )
    updated_camera = rewrite_json_artifact(broken, camera_record, camera_payload)
    transition["after"]["camera_world_transform"] = updated_camera.model_dump(mode="json")
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match="camera"):
        validate_dataset(broken)


def test_cross_episode_surface_identifier_reuse_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "surface-id-reuse")
    first_transition, _ = load_episode_payloads(broken, 0)
    second_transition, second_instrumentation = load_episode_payloads(broken, 1)
    reused_id = first_transition["surfaces"][0]["surface_id"]
    replaced_id = second_transition["surfaces"][0]["surface_id"]
    second_transition = replace_string(second_transition, replaced_id, reused_id)
    second_instrumentation = replace_string(second_instrumentation, replaced_id, reused_id)
    assert isinstance(second_transition, dict)
    assert isinstance(second_instrumentation, dict)
    commit_episode_payloads(broken, 1, second_transition, second_instrumentation)

    with pytest.raises(DatasetValidationError, match="disjoint across episodes"):
        validate_dataset(broken)


def test_tampered_inline_provenance_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "provenance-tamper")
    manifest_path = broken / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_provenance"]["package_version"] = "tampered"
    write_canonical_json(manifest_path, manifest)

    with pytest.raises(DatasetValidationError, match="source provenance hash mismatch"):
        validate_dataset(broken)


@pytest.mark.parametrize("rehash_renderer", [False, True])
def test_renderer_execution_provenance_tampering_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    rehash_renderer: bool,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, f"renderer-tamper-{rehash_renderer}")
    manifest_path = broken / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["renderer_provenance"]["backend"] = "fabricated-backend"
    if rehash_renderer:
        renderer = RendererProvenance.model_validate_json(
            canonical_json_bytes(manifest["renderer_provenance"])
        )
        manifest["renderer_execution_provenance_sha256"] = (
            compute_renderer_execution_provenance_hash(renderer)
        )
    write_canonical_json(manifest_path, manifest)

    expected = "content/provenance binding" if rehash_renderer else "renderer/execution"
    with pytest.raises(DatasetValidationError, match=expected):
        validate_dataset(broken)


def test_renderer_execution_provenance_is_bound_but_not_scientific_content(
    smoke_dataset: Path,
) -> None:
    manifest = load_manifest(smoke_dataset)
    changed_renderer = manifest.renderer_provenance.model_copy(
        update={"backend": "different-execution-backend"}
    )
    changed_manifest = manifest.model_copy(update={"renderer_provenance": changed_renderer})
    assert compute_dataset_logical_hash(changed_manifest) == manifest.dataset_logical_sha256
    changed_execution_hash = compute_renderer_execution_provenance_hash(changed_renderer)
    assert changed_execution_hash != manifest.renderer_execution_provenance_sha256
    assert (
        compute_content_provenance_binding(
            manifest.dataset_logical_sha256,
            manifest.source_provenance_sha256,
            changed_execution_hash,
        )
        != manifest.content_provenance_binding_sha256
    )


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -1.0])
def test_invalid_depth_values_are_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    invalid_value: float,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, f"depth-{invalid_value!r}")
    transition, instrumentation = load_episode_payloads(broken, 0)
    record = _artifact(transition["before"]["depth"])
    depth = np.load(broken / record.path, allow_pickle=False)
    depth[0, 0] = invalid_value
    updated = rewrite_array_artifact(broken, record, depth)
    transition["before"]["depth"] = updated.model_dump(mode="json")
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match="finite and non-negative"):
        validate_dataset(broken)


def test_wrong_depth_dtype_is_rejected_by_role_schema(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "depth-dtype")
    transition, _ = load_episode_payloads(broken, 0)
    record = _artifact(transition["before"]["depth"])
    depth = np.load(broken / record.path, allow_pickle=False).astype(np.float64)
    updated = rewrite_array_artifact(broken, record, depth)
    transition["before"]["depth"] = updated.model_dump(mode="json")
    commit_raw_transition_payload(broken, 0, transition)

    with pytest.raises(DatasetValidationError, match="transition failed schema validation"):
        validate_dataset(broken)


def test_undeclared_segmentation_label_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "segmentation-label")
    transition, instrumentation = load_episode_payloads(broken, 0)
    record = _artifact(transition["before"]["segmentation"])
    segmentation = np.load(broken / record.path, allow_pickle=False)
    segmentation[0, 0] = np.int32(2**31 - 1)
    updated = rewrite_array_artifact(broken, record, segmentation)
    transition["before"]["segmentation"] = updated.model_dump(mode="json")
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match="undeclared surface label"):
        validate_dataset(broken)


def test_duplicate_artifact_path_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "duplicate-artifact")
    transition, instrumentation = load_episode_payloads(broken, 0)
    transition["after"]["depth"] = transition["before"]["depth"]
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match="duplicate artifact path"):
        validate_dataset(broken)


def test_hardlink_artifact_alias_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "hardlink-artifact")
    transition, instrumentation = load_episode_payloads(broken, 0)
    before = _artifact(transition["before"]["depth"])
    after = _artifact(transition["after"]["depth"])
    after_path = broken / after.path
    after_path.unlink()
    os.link(broken / before.path, after_path)
    aliased = before.model_copy(update={"path": after.path})
    transition["after"]["depth"] = aliased.model_dump(mode="json")
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match=r"hard-link alias|aliases another role"):
        validate_dataset(broken)


@pytest.mark.parametrize(
    ("field_path", "new_value"),
    [
        (("before", "frame_index"), 1),
        (("before", "depth", "modality"), "rgb"),
    ],
)
def test_structural_and_role_corruption_is_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
    field_path: tuple[str, ...],
    new_value: Any,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, f"structure-{field_path[-1]}")
    transition, _ = load_episode_payloads(broken, 0)
    target = transition
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = new_value
    commit_raw_transition_payload(broken, 0, transition)

    with pytest.raises(DatasetValidationError, match="transition failed schema validation"):
        validate_dataset(broken)


def test_noncontiguous_episode_indices_are_rejected(
    smoke_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(smoke_dataset, tmp_path, "episode-index")
    manifest_path = broken / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episodes"][1]["episode_index"] = 2
    write_canonical_json(manifest_path, manifest)

    with pytest.raises(DatasetValidationError, match="manifest failed schema validation"):
        validate_dataset(broken)
