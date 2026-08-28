from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import epsbench.audit as audit
from epsbench.appearance import load_appearance_registry, load_evaluation_seed_registry
from epsbench.utils.canonical import write_canonical_json


def _failed_cell(
    dataset: Path,
    repeat: Path,
    packet_root: Path,
    config: Any,
    profile: Any,
    seed_index: int,
    candidate_seed: int,
    registry: Any,
    seeds: Any,
) -> dict[str, Any]:
    del dataset, repeat, packet_root, registry, seeds
    scene = config.scene_family.value
    return {
        "cell_id": f"{scene}--{profile.profile_id}--seed-{seed_index}",
        "scene_family": scene,
        "seed_index": seed_index,
        "candidate_seed": candidate_seed,
        "profile_id": profile.profile_id,
        "matched_control_profile_id": profile.matched_control_profile_id,
        "generation_status": "failed",
        "admission_status": "rejected",
        "rejection_reasons": ["generation_or_validation_failed"],
        "failure_type": "SyntheticFailure",
        "failure_message": "retained negative evidence",
    }


@pytest.fixture()
def synthetic_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(audit, "_dataset_cell", _failed_cell)
    output = tmp_path / "packet"
    audit.create_appearance_audit(
        Path("configs/appearance_candidates_v0.yaml"),
        Path("configs/evaluation_seed_candidates_v0.yaml"),
        Path("configs/benchmark_v0.yaml"),
        Path("configs/corridor_v0.yaml"),
        output,
    )
    return output


def test_audit_packet_is_complete_validated_and_retains_failures(
    synthetic_packet: Path,
) -> None:
    packet = audit.validate_appearance_audit(synthetic_packet)
    for required in (
        "candidate_packet.json",
        "appearance_registry_snapshot.json",
        "seed_registry_snapshot.json",
        "profile_summary.json",
        "seed_matrix.json",
        "negative_evidence.json",
        "representative_contact_sheets",
    ):
        assert (synthetic_packet / required).exists()
    assert packet["freeze_status"] == "candidate_packet_only_not_frozen"
    negatives = json.loads((synthetic_packet / "negative_evidence.json").read_text())
    assert len(negatives["rejected_cells"]) == 10 * 8 * 2


def test_audit_refuses_a_non_empty_output(
    synthetic_packet: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "_dataset_cell", _failed_cell)
    with pytest.raises(FileExistsError):
        audit.create_appearance_audit(
            Path("configs/appearance_candidates_v0.yaml"),
            Path("configs/evaluation_seed_candidates_v0.yaml"),
            Path("configs/benchmark_v0.yaml"),
            Path("configs/corridor_v0.yaml"),
            synthetic_packet,
        )


def test_atomic_directory_publication_does_not_replace_a_racing_target(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    (staging / "packet.json").write_text("staged\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit._atomic_no_replace_directory(staging, destination)
    assert staging.is_dir()
    assert destination.is_dir()
    assert not (destination / "packet.json").exists()


def test_volatile_metadata_does_not_change_packet_root(synthetic_packet: Path) -> None:
    before = audit.validate_appearance_audit(synthetic_packet)["packet_logical_root_sha256"]
    (synthetic_packet / "run.json").write_text('{"arbitrary":"volatile"}\n', encoding="utf-8")
    after = audit.validate_appearance_audit(synthetic_packet)["packet_logical_root_sha256"]
    assert after == before


def test_missing_matrix_cell_and_altered_admission_are_rejected(
    synthetic_packet: Path,
) -> None:
    matrix_path = synthetic_packet / "seed_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    original = json.loads(json.dumps(matrix))
    matrix["cells"].pop()
    write_canonical_json(matrix_path, matrix)
    with pytest.raises(audit.AppearanceAuditError, match="matrix is incomplete"):
        audit.validate_appearance_audit(synthetic_packet)
    write_canonical_json(matrix_path, original)
    matrix = json.loads(json.dumps(original))
    matrix["cells"][0]["admission_status"] = "admitted"
    matrix["cells"][0]["rejection_reasons"] = []
    write_canonical_json(matrix_path, matrix)
    with pytest.raises(audit.AppearanceAuditError, match="altered admission evidence"):
        audit.validate_appearance_audit(synthetic_packet)


def _successful_cell() -> dict[str, Any]:
    return {
        "cell_id": "single_occluder--legacy_solid_base_v1--seed-0",
        "generation_status": "success",
        "appearance_profile_sha256": "profile",
        "appearance_instance_sha256": "instance",
        "appearance_instance": {
            "seeds": {"candidate_schedule_index": 0},
            "assignment_schedule_posture": "candidate_registry_index",
            "style_assignment": {},
            "textures": [],
        },
        "evaluation_seed_registry_sha256": "seed-registry",
        "appearance_assignment_schedule_source": ("snapshotted_evaluation_seed_registry_v1"),
        "scene_content_sha256": "scene",
        "analytic_transport_sha256": "transport",
        "oriented_boundary_sha256": "boundary",
        "visibility_event_sha256": "event",
        "ecological_label_sha256": "label-windows",
        "occlusion_sha256": "occlusion",
        "action_sha256": "action",
        "camera_trajectory_sha256": "camera-windows",
        "geometry_sha256": "geometry-windows",
        "opaque_remapping_sha256": "remapping-windows",
        "admission_checks": {
            "structural_invariance": True,
            "portable_analytic_identity_equality": True,
            "ecological_label_equality": True,
            "depth_segmentation_invariance": True,
            "determinism": True,
        },
        "rgb_logical_sha256": "rgb-windows",
        "renderer_provenance": {"backend": "wgl"},
        "frame_metrics": {},
        "determinism_pass": True,
        "admission_status": "admitted",
    }


def test_renderer_local_labels_can_differ_while_portable_outcome_matches() -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    windows = audit._roots(registry, seeds, [cell])
    ubuntu = deepcopy(cell)
    ubuntu.update(
        {
            "ecological_label_sha256": "label-ubuntu",
            "camera_trajectory_sha256": "camera-ubuntu",
            "geometry_sha256": "geometry-ubuntu",
            "opaque_remapping_sha256": "remapping-ubuntu",
            "rgb_logical_sha256": "rgb-ubuntu",
            "renderer_provenance": {"backend": "osmesa"},
        }
    )
    assert audit._structural_domain(ubuntu) != audit._structural_domain(cell)
    assert ubuntu["admission_checks"]["structural_invariance"] is True
    linux = audit._roots(registry, seeds, [ubuntu])
    assert (
        linux["appearance_invariance_outcome_root_sha256"]
        == windows["appearance_invariance_outcome_root_sha256"]
    )
    assert (
        linux["portable_analytic_identity_root_sha256"]
        == windows["portable_analytic_identity_root_sha256"]
    )
    assert (
        linux["renderer_local_ecological_label_root_sha256"]
        != windows["renderer_local_ecological_label_root_sha256"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "scene_content_sha256",
        "analytic_transport_sha256",
        "oriented_boundary_sha256",
        "visibility_event_sha256",
        "occlusion_sha256",
        "action_sha256",
    ],
)
def test_each_portable_identity_changes_both_portable_roots(field: str) -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    original = audit._roots(registry, seeds, [cell])
    changed = deepcopy(cell)
    changed[field] = f"changed-{field}"
    altered = audit._roots(registry, seeds, [changed])
    for root_name in (
        "portable_analytic_identity_root_sha256",
        "appearance_invariance_outcome_root_sha256",
    ):
        assert altered[root_name] != original[root_name]


def test_raw_label_changes_only_renderer_local_label_root_among_non_rgb_domains() -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    original = audit._roots(registry, seeds, [cell])
    changed = deepcopy(cell)
    changed["ecological_label_sha256"] = "renderer-local-alternate"
    altered = audit._roots(registry, seeds, [changed])
    assert (
        altered["renderer_local_ecological_label_root_sha256"]
        != original["renderer_local_ecological_label_root_sha256"]
    )
    for root_name in (
        "appearance_registry_sha256",
        "seed_registry_sha256",
        "procedural_asset_root_sha256",
        "appearance_assignment_root_sha256",
        "portable_analytic_identity_root_sha256",
        "appearance_invariance_outcome_root_sha256",
        "renderer_specific_audit_root_sha256",
    ):
        assert altered[root_name] == original[root_name]


def test_rgb_and_renderer_provenance_never_enter_portable_roots() -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    cell = _successful_cell()
    original = audit._roots(registry, seeds, [cell])
    changed = deepcopy(cell)
    changed["rgb_logical_sha256"] = "different-rgb"
    changed["renderer_provenance"] = {"backend": "different-renderer"}
    altered = audit._roots(registry, seeds, [changed])
    assert (
        altered["portable_analytic_identity_root_sha256"]
        == original["portable_analytic_identity_root_sha256"]
    )
    assert (
        altered["appearance_invariance_outcome_root_sha256"]
        == original["appearance_invariance_outcome_root_sha256"]
    )


def test_ecological_label_inequality_changes_outcome_and_rejects_cell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = load_appearance_registry(Path("configs/appearance_candidates_v0.yaml"))
    seeds = load_evaluation_seed_registry(Path("configs/evaluation_seed_candidates_v0.yaml"))
    profile = next(
        item for item in registry.profiles if item.profile_id == "balanced_solid_palette_v1"
    )
    cell = _successful_cell()
    control = deepcopy(cell)
    cell["ecological_label_sha256"] = "candidate-label"
    control["ecological_label_sha256"] = "control-label"
    cell["semantic_surface_labels"] = {}
    cell["source_texture_diagnostics"] = []
    control["semantic_surface_labels"] = {}
    control["source_texture_diagnostics"] = []
    monkeypatch.setattr(
        audit,
        "_load_frame",
        lambda *_: (
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 1), dtype=np.int32),
        ),
    )
    monkeypatch.setattr(
        audit,
        "_frame_metrics",
        lambda *_: {
            "material_change_pass": True,
            "controlled_surface_exposure_pass": True,
            "textured_surface_variation_pass": True,
            "surface_diagnostics": [],
        },
    )
    original = audit._roots(registry, seeds, [deepcopy(control)])
    audit._evaluate_cell(tmp_path, cell, control, profile)
    assert cell["admission_checks"]["ecological_label_equality"] is False
    assert cell["admission_status"] == "rejected"
    assert "ecological_label_equality" in cell["rejection_reasons"]
    assert (
        audit._roots(registry, seeds, [cell])["appearance_invariance_outcome_root_sha256"]
        != original["appearance_invariance_outcome_root_sha256"]
    )


@pytest.mark.parametrize(
    "root_name",
    [
        "appearance_invariance_outcome_root_sha256",
        "renderer_local_ecological_label_root_sha256",
    ],
)
def test_fully_rehashed_root_corruption_is_rejected(
    synthetic_packet: Path,
    root_name: str,
) -> None:
    packet_path = synthetic_packet / "candidate_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["roots"][root_name] = "0" * 64
    logical_domain = {
        key: packet[key]
        for key in (
            "schema_version",
            "root_schema_version",
            "freeze_status",
            "source_provenance",
            "governing_document_hashes",
            "roots",
            "report_file_sha256",
        )
    }
    packet["packet_logical_root_sha256"] = audit._hash_json(logical_domain)
    write_canonical_json(packet_path, packet)
    with pytest.raises(audit.AppearanceAuditError, match="root mismatch"):
        audit.validate_appearance_audit(synthetic_packet)


def test_packet_seed_registry_snapshot_is_independently_validated(
    synthetic_packet: Path,
) -> None:
    snapshot = synthetic_packet / "seed_registry_snapshot.json"
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["candidate_episode_seeds"] = list(reversed(payload["candidate_episode_seeds"]))
    write_canonical_json(snapshot, payload)
    with pytest.raises(audit.AppearanceAuditError, match="registry snapshot"):
        audit.validate_appearance_audit(synthetic_packet)
