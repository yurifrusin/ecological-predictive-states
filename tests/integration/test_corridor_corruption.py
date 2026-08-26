from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from epsbench.data import DatasetValidationError, validate_dataset
from epsbench.data.identity import (
    compute_content_provenance_binding,
    compute_dataset_logical_hash,
)
from epsbench.schema import ArtifactRecord, SceneFamily
from epsbench.utils.canonical import canonical_json_bytes, write_canonical_json
from tests.dataset_mutations import (
    commit_episode_payloads,
    commit_raw_instrumentation_payload,
    commit_raw_transition_payload,
    load_episode_payloads,
    load_manifest,
    rewrite_array_artifact,
    rewrite_json_artifact,
)


def _copy_dataset(corridor_dataset: Path, tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(corridor_dataset, target)
    return target


def _artifact(payload: dict[str, Any]) -> ArtifactRecord:
    return ArtifactRecord.model_validate_json(canonical_json_bytes(payload))


@pytest.mark.parametrize("corruption", ["missing", "extra", "duplicate_raw", "duplicate_opaque"])
def test_inexact_corridor_apparatus_is_rejected(
    corridor_dataset: Path,
    tmp_path: Path,
    corruption: str,
) -> None:
    broken = _copy_dataset(corridor_dataset, tmp_path, f"corridor-apparatus-{corruption}")
    _, instrumentation = load_episode_payloads(broken, 0)
    if corruption == "missing":
        name, raw_id = next(iter(instrumentation["raw_geom_ids"].items()))
        del instrumentation["raw_geom_ids"][name]
        del instrumentation["raw_geom_world_positions"][name]
        del instrumentation["raw_to_opaque_surface_ids"][str(raw_id)]
        instrumentation["apparatus_surface_names"].remove(name)
    elif corruption == "extra":
        instrumentation["raw_geom_ids"]["fabricated_surface"] = 99
        instrumentation["raw_geom_world_positions"]["fabricated_surface"] = [0.0, 0.0, 0.0]
        instrumentation["raw_to_opaque_surface_ids"]["99"] = next(
            iter(instrumentation["raw_to_opaque_surface_ids"].values())
        )
        instrumentation["apparatus_surface_names"].append("fabricated_surface")
    elif corruption == "duplicate_raw":
        names = list(instrumentation["raw_geom_ids"])
        instrumentation["raw_geom_ids"][names[1]] = instrumentation["raw_geom_ids"][names[0]]
    else:
        keys = list(instrumentation["raw_to_opaque_surface_ids"])
        instrumentation["raw_to_opaque_surface_ids"][keys[1]] = instrumentation[
            "raw_to_opaque_surface_ids"
        ][keys[0]]
    commit_raw_instrumentation_payload(broken, 0, instrumentation)

    with pytest.raises(DatasetValidationError, match="instrumentation failed schema validation"):
        validate_dataset(broken)


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    [
        ("lateral", "camera"),
        ("wrong_sign", "camera"),
        ("height", "camera"),
        ("rotation", "camera"),
        ("past_end", "camera"),
    ],
)
def test_corridor_camera_corruption_is_rejected(
    corridor_dataset: Path,
    tmp_path: Path,
    tamper: str,
    expected_error: str,
) -> None:
    broken = _copy_dataset(corridor_dataset, tmp_path, f"corridor-camera-{tamper}")
    transition, instrumentation = load_episode_payloads(broken, 0)
    before_record = _artifact(transition["before"]["camera_world_transform"])
    after_record = _artifact(transition["after"]["camera_world_transform"])
    before_payload = json.loads((broken / before_record.path).read_text(encoding="utf-8"))
    after_payload = json.loads((broken / after_record.path).read_text(encoding="utf-8"))
    if tamper == "lateral":
        after_payload["camera_world_position"][0] += 0.1
    elif tamper == "wrong_sign":
        after_payload["camera_world_position"][1] = (
            before_payload["camera_world_position"][1] - transition["action"]["delta_forward"]
        )
    elif tamper == "height":
        after_payload["camera_world_position"][2] += 0.1
    elif tamper == "rotation":
        current = np.asarray(after_payload["camera_world_rotation_row_major"]).reshape(3, 3)
        quarter_turn = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        after_payload["camera_world_rotation_row_major"] = (quarter_turn @ current).ravel().tolist()
    else:
        after_payload["camera_world_position"][1] = instrumentation["sampled_geometry"]["length"]
    updated = rewrite_json_artifact(broken, after_record, after_payload)
    transition["after"]["camera_world_transform"] = updated.model_dump(mode="json")
    commit_episode_payloads(broken, 0, transition, instrumentation)

    with pytest.raises(DatasetValidationError, match=expected_error):
        validate_dataset(broken)


@pytest.mark.parametrize(
    "action",
    [
        {"name": "forward", "delta_forward": 0.0, "delta_lateral": 0.0, "delta_yaw": 0.0},
        {"name": "forward", "delta_forward": -0.7, "delta_lateral": 0.0, "delta_yaw": 0.0},
        {"name": "forward", "delta_forward": 0.7, "delta_lateral": 0.1, "delta_yaw": 0.0},
        {"name": "forward", "delta_forward": 0.7, "delta_lateral": 0.0, "delta_yaw": 0.1},
    ],
)
def test_invalid_persisted_forward_action_is_rejected(
    corridor_dataset: Path,
    tmp_path: Path,
    action: dict[str, Any],
) -> None:
    broken = _copy_dataset(corridor_dataset, tmp_path, "corridor-invalid-action")
    transition, _ = load_episode_payloads(broken, 0)
    transition["action"] = action
    commit_raw_transition_payload(broken, 0, transition)
    with pytest.raises(DatasetValidationError, match="transition failed schema validation"):
        validate_dataset(broken)


def test_corridor_manifest_scene_family_mismatch_is_rejected(
    corridor_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(corridor_dataset, tmp_path, "corridor-family-mismatch")
    manifest = load_manifest(broken)
    provisional = manifest.model_copy(update={"scene_family": SceneFamily.SINGLE_OCCLUDER})
    dataset_hash = compute_dataset_logical_hash(provisional)
    changed = provisional.model_copy(
        update={
            "dataset_logical_sha256": dataset_hash,
            "content_provenance_binding_sha256": compute_content_provenance_binding(
                dataset_hash,
                provisional.source_provenance_sha256,
                provisional.renderer_execution_provenance_sha256,
            ),
        }
    )
    write_canonical_json(broken / "manifest.json", changed)
    with pytest.raises(DatasetValidationError, match="scene family"):
        validate_dataset(broken)


def test_corridor_sampled_geometry_tampering_is_rejected(
    corridor_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(corridor_dataset, tmp_path, "corridor-geometry")
    transition, instrumentation = load_episode_payloads(broken, 0)
    instrumentation["sampled_geometry"]["width"] += 0.01
    commit_episode_payloads(broken, 0, transition, instrumentation)
    with pytest.raises(DatasetValidationError, match="sampled corridor geometry"):
        validate_dataset(broken)


def test_hash_rebuilt_fabricated_corridor_raw_ids_are_rejected(
    corridor_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(corridor_dataset, tmp_path, "corridor-fabricated-raw-ids")
    transition, instrumentation = load_episode_payloads(broken, 0)
    old_mapping = instrumentation["raw_to_opaque_surface_ids"]
    changed_mapping: dict[str, str] = {}
    for name, old_raw_id in tuple(instrumentation["raw_geom_ids"].items()):
        new_raw_id = old_raw_id + 100
        instrumentation["raw_geom_ids"][name] = new_raw_id
        changed_mapping[str(new_raw_id)] = old_mapping[str(old_raw_id)]
    instrumentation["raw_to_opaque_surface_ids"] = changed_mapping
    commit_episode_payloads(broken, 0, transition, instrumentation)
    with pytest.raises(DatasetValidationError, match="compiled MuJoCo geom identifiers"):
        validate_dataset(broken)


def test_corridor_declared_surface_absent_from_both_frames_is_rejected(
    corridor_dataset: Path,
    tmp_path: Path,
) -> None:
    broken = _copy_dataset(corridor_dataset, tmp_path, "corridor-missing-visible-surface")
    transition, instrumentation = load_episode_payloads(broken, 0)
    missing_label = transition["surfaces"][0]["segmentation_label"]
    for frame_name in ("before", "after"):
        record = _artifact(transition[frame_name]["segmentation"])
        segmentation = np.load(broken / record.path, allow_pickle=False)
        segmentation[segmentation == missing_label] = 0
        updated = rewrite_array_artifact(broken, record, segmentation)
        transition[frame_name]["segmentation"] = updated.model_dump(mode="json")
    commit_episode_payloads(broken, 0, transition, instrumentation)
    with pytest.raises(DatasetValidationError, match="absent from both frames"):
        validate_dataset(broken)


@pytest.mark.parametrize("fabrication", ["available_empty", "available_relation"])
def test_corridor_cannot_claim_available_occlusion_without_oracle(
    corridor_dataset: Path,
    tmp_path: Path,
    fabrication: str,
) -> None:
    broken = _copy_dataset(
        corridor_dataset,
        tmp_path,
        f"corridor-fabricated-occlusion-{fabrication}",
    )
    transition, instrumentation = load_episode_payloads(broken, 0)
    first, second = (surface["surface_id"] for surface in transition["surfaces"][:2])
    relations = []
    if fabrication == "available_relation":
        relations.append(
            {
                "occluder_surface_id": first,
                "occluded_surface_id": second,
                "frame_indices": [0, 1],
            }
        )
    transition["occlusion"] = {
        "status": "available",
        "oracle_rule": "counterfactual_occluder_exclusion_v1",
        "relations": relations,
    }
    commit_episode_payloads(broken, 0, transition, instrumentation)
    with pytest.raises(DatasetValidationError, match="corridor occlusion"):
        validate_dataset(broken)
